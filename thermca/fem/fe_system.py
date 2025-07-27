"""Creates executable finite element systems from meshed parts"""

from warnings import warn
import os
from itertools import combinations
from typing import Iterable
from dataclasses import dataclass
import logging

# fmt: off
from numpy import (
    array, linalg, zeros, nonzero, full_like, empty_like, arange, sum as asum, empty,
    hstack, setdiff1d
)
# fmt: on
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy import spatial

from thermca.mesh import Mesh
from thermca.materials import Solid  # for type hints
from thermca._utils.func_tools import disk_cache
from thermca._utils.sparse import coo_diag_data_idxs
from thermca.fem.fe_part import SFC_TO_SFC, SFC_TO_CENTER_PLANE
from thermca.static_bcs import TempBC, HeatBC, flatten_bcs

# skfem is the default FE-system for Thermca
import skfem as fe
from skfem.models.poisson import laplace, mass, unit_load


@dataclass
class Surf:
    name: str


class FESystem:
    """Creates a FEM system from part meshes

    It takes one tetrahedron body mesh and multiple surface meshes as
    interfaces for heat exchange. The meshes have to share the same
    point locations.

    Args:
        body_mesh: Mesh containing body tetrahedrons
        surf_meshes: Mesh containing triangle surfaces
        init_temp: Initial temperature
        matl: Material
        lump_cond_meth: Method to calculate lumped parameters
        temp_dependent: Temperature dependent material
        name: Name for debugging

    Attributes:
        C_surf: Output matrix of IO-system to get the mean coupling
            surface temperature
        C_body: Output matrix of IO-system to get the mean body temperature
    """

    def __init__(
        self,
        body_mesh: Mesh,
        surf_meshes: Mesh,
        init_temp,
        matl: Solid,
        lump_cond_meth,
        temp_dependent,
        name: str = '',
    ):
        self.init_temp = init_temp
        self.matl = matl
        self.surf_names = surf_meshes.block_names
        self.name = name
        # self.surf_to_idx = {name: idx for idx, name in enumerate(self.surf_names)}
        # use: self.surf_names.index(surf_name) instead

        (
            Mx_diag,
            Lx_x,
            Lx_y,
            Lx_z,
            Qs,
            vert_to_dof_idx,
            dof_to_vert_idx,
            dof,
            surf_dof_idxs,
            surf_areas,
        ) = self.fem_matrices_from_skfem(body_mesh, surf_meshes)

        self.Mx_diag = Mx_diag
        self.M_diag = matl.vol_capy_interp(init_temp) * Mx_diag
        # Geometry only; only needed for body properties with mean body temperature
        self.Lx = Lx_x + Lx_y + Lx_z
        if matl._condy is not None:
            self.L = matl.condy_interp(init_temp) * self.Lx
        if matl._orthotropic_condy is not None:
            self.L = (
                matl._orthotropic_condy[0] * Lx_x
                + matl._orthotropic_condy[1] * Lx_y
                + matl._orthotropic_condy[2] * Lx_z
            )

        if temp_dependent:
            if matl._orthotropic_condy is not None:
                raise ValueError(
                    "Temperature dependent orthotropic material is not supported!"
                )
            Lx_base = self.Lx.copy()  # Lx without negated sum of rows on diagonal
            self.Lx_base = Lx_base
            Lx_base_coo = Lx_base.tocoo(copy=False)
            self.Lx_base_coo = Lx_base_coo
            # L_diag_data_idxs = (Lx_base_coo.row == Lx_base_coo.col).nonzero()[0]
            self.L_diag_data_idxs = coo_diag_data_idxs(
                Lx_base.shape, Lx_base.data.shape, Lx_base_coo.row, Lx_base_coo.col
            )
            Lx_base.data[self.L_diag_data_idxs] = 0.0
            # Test for same L like above during update of material data
            # self.L.data = matl.condy_interp(init_temp) * Lx_base.data
            # body_temp = full_like(Mx_diag, init_temp)
            # cond_temp = (body_temp[Lx_base_coo.row] + body_temp[Lx_base_coo.col]) / 2.
            # self.L.data = matl.condy_interp(cond_temp) * Lx_base.data
            # self.L.data[self.L_diag_data_idxs] = -self.L.sum(axis=1).ravel()

        self.Qs = Qs

        # Node-area weighted output matrix for surf mean temps
        C_surf = Qs.copy()
        for c in C_surf.T:
            nz_mask = c != 0.0
            num_dofs = asum(nz_mask)
            c_nz = c[nz_mask]
            c_mean = asum(c_nz) / num_dofs
            c_area_wgt = c_nz / c_mean
            c[nz_mask] = c_area_wgt * 1.0 / num_dofs
        self.C_surf = C_surf

        # Output matrix for volume weighted body mean temps
        m_mean = asum(Mx_diag) / dof
        m_vol_wgt = Mx_diag / m_mean  # relative weight regarding node volume
        self.C_body = m_vol_wgt * 1.0 / dof

        self.vert_to_dof_idx = vert_to_dof_idx
        self.dof_to_vert_idx = dof_to_vert_idx
        self.dof = dof
        self.surf_dof_idxs = surf_dof_idxs
        self.surf_areas = surf_areas

        # lump_surf_conds is calculated with skfem
        if lump_cond_meth is None:
            surf_conds = full_like(surf_areas, 1e-32)
        elif lump_cond_meth == SFC_TO_SFC:
            surf_conds = self.conductance_from_surface_to_surface()
        elif lump_cond_meth == SFC_TO_CENTER_PLANE:
            surf_conds = self.conductance_from_surfaces_to_center(
                body_mesh, surf_meshes, matl, init_temp
            )
        self.lump_surf_conds = surf_conds
        self.lump_capy = self.M_diag.sum()

    def update_material_properties(self, body_temps):
        # TODO: better check all result temps at once
        self.matl.check_limits(body_temps)
        cond_temp = (
            body_temps[self.Lx_base_coo.row] + body_temps[self.Lx_base_coo.col]
        ) / 2.0
        self.L.data = self.matl.condy_interp(cond_temp) * self.Lx_base.data
        self.L.data[self.L_diag_data_idxs] = -self.L.sum(axis=1).ravel()
        self.M_diag = self.matl.vol_capy_interp(body_temps) * self.Mx_diag

    @staticmethod
    def fem_matrices_from_skfem(body_mesh: Mesh, surf_meshes: Mesh):
        """Discretize the heat equation in space

        The returned FE-system contains only geometric information,
        because there are no material properties or heat loads applied.

        Args:
            body_mesh: Mesh containing body tetrahedrons
            surf_meshes: Mesh containing coupling surfaces defined by
                triangles

        Returns:
            Mx_diag: Lumped diagonal of the thermal capacity matrix
            Lx_x: Thermal conductance matrix in x direction
            Lx_y: Thermal conductance matrix in y direction
            Lx_z: Thermal conductance matrix in z direction
            Qs: Load Matrix with columns for coupling surfaces
                containing node related surface area
            vert_to_dof: Mapping from mesh vertex to FE-node index
            dof_to_vert: Mapping from FE-node to mesh vertex index
            dof: Degree of freedom of FE-system
            surf_dof_idxs: Degree of freedom indices of coupling surfaces
            surf_areas: Surface area of coupling surfaces
        """

        def to_skfem_facets(points, skfem_facet_midpts, thermca_facet_cells):
            """Determine, which Skfem mesh facets correspond to the
            given Thermca mesh facet cells. Skfem stores facet
            collections as indices of sorted facet cells. Thermca
            stores facets as cells with indices of points."""
            bound_midpts = points[thermca_facet_cells].mean(1)
            dists, idxs = spatial.KDTree(skfem_facet_midpts).query(bound_midpts)
            return idxs

        @fe.BilinearForm
        def laplace_x(u, v, _):
            return u.grad[0] * v.grad[0]  # ∂u/∂x * ∂v/∂x

        @fe.BilinearForm
        def laplace_y(u, v, _):
            return u.grad[1] * v.grad[1]  # ∂u/∂y * ∂v/∂y

        @fe.BilinearForm
        def laplace_z(u, v, _):
            return u.grad[2] * v.grad[2]  # ∂u/∂z * ∂v/∂z

        # Prevent skfem console logging warnings
        logging.getLogger().setLevel(100)

        fe_mesh = fe.MeshTet(
            body_mesh.points.T,
            body_mesh.cell_blocks[0].T,
        )
        midpts = fe_mesh.p[:, fe_mesh.facets].mean(axis=1).T
        fe_boundaries = {
            name: to_skfem_facets(body_mesh.points, midpts, cells)
            for name, cells in zip(surf_meshes.block_names, surf_meshes.cell_blocks)
        }
        fe_mesh = fe.MeshTet(
            body_mesh.points.T,
            body_mesh.cell_blocks[0].T,
            fe_boundaries,
        )

        body_basis = fe.Basis(fe_mesh, fe.ElementTetP1())
        surf_basis = [
            fe.FacetBasis(fe_mesh, fe.ElementTetP1(), facets=fe_mesh.boundaries[name])
            for name in surf_meshes.block_names
        ]  # InteriorFacetBasis

        # Assemble each directional matrix
        Lx_x = fe.asm(laplace_x, body_basis)
        Lx_y = fe.asm(laplace_y, body_basis)
        Lx_z = fe.asm(laplace_z, body_basis)

        # Mass lumping for linear elements
        Mx = fe.asm(mass, body_basis)
        Mx_diag = Mx.sum(axis=1).A.ravel()

        Qs = array([fe.asm(unit_load, basis) for basis in surf_basis]).T
        surf_areas = Qs.sum(axis=0)
        vert_to_dof_idx = body_basis.nodal_dofs.ravel()
        dof_to_vert_idx = empty_like(vert_to_dof_idx)
        dof_to_vert_idx[vert_to_dof_idx] = arange(len(vert_to_dof_idx))
        dof = len(vert_to_dof_idx)
        surf_dof_idxs = [
            body_basis.get_dofs(name).flatten() for name in surf_meshes.block_names
        ]
        return (
            Mx_diag,
            Lx_x,
            Lx_y,
            Lx_z,
            Qs,
            vert_to_dof_idx,
            dof_to_vert_idx,
            dof,
            surf_dof_idxs,
            surf_areas,
        )

    def stationary_solution(self, bound_condns: Iterable):
        """Stationary (equilibrium) solution for FE-system

        Note:
            Temperature dependent material properties and film
            coefficients are calculated with initial temperatures.

        Args:
            bound_condns: Iterable of boundary conditions on coupling
                surfaces. Boundary conditions may be of type: `TempBC`,
                `HeatBC`, `FluxBC` or `FilmBC`.

        Returns:
            Node temperatures
        """
        surf_names = self.surf_names
        surf_areas = self.surf_areas

        (
            bc_temp_idxs,
            bc_temps,
            bc_heat_idxs,
            bc_heats,
            bc_film_idxs,
            bc_films,
            bc_film_temps,
        ) = flatten_bcs(bound_condns, surf_areas, surf_names)

        # Heat sources
        surf_fluxes = zeros(len(surf_names))
        surf_fluxes[bc_heat_idxs] = bc_heats / surf_areas[bc_heat_idxs]
        # Film coefficients
        if bc_films.size > 0:
            bc_flux = array(
                [film * env_temp for film, env_temp in zip(bc_films, bc_film_temps)]
            )
            surf_fluxes[bc_film_idxs] += bc_flux
            L_films = [diags(q, format='csr') for q in self.Qs.T[bc_film_idxs]]
            L = self.L + asum(
                [film * L_film for film, L_film in zip(bc_films, L_films)]
            )
        else:
            L = self.L
        load_idxs = nonzero(surf_fluxes)[0]
        if load_idxs.size > 0:
            flows = (self.Qs[:, load_idxs] @ surf_fluxes[load_idxs]).ravel()
        else:
            flows = zeros(self.L.shape[0])
        # Bound temperatures
        temps = empty(self.L.shape[0])
        bc_temp_dofs = [self.surf_dof_idxs[i] for i in bc_temp_idxs]  # DOFs at boundary
        for dofs, bc_temp in zip(bc_temp_dofs, bc_temps):
            temps[dofs] = bc_temp
        if bc_temps.size > 0:
            # Transform the system to bind the given boundary temperatures.
            # First, split the system in known k and unknown u temperatures:
            # ⎡L_uu L_uk⎤ ⎧τ_u⎫ = ⎧q̇_u⎫
            # ⎣L_ku L_kk⎦ ⎩τ_k⎭   ⎩q̇_k⎭
            # Second, extract the equation system for the unknown temperatures:
            # L_uu⋅τ_u = q̇_u - L_uk⋅τ_k
            # lhs⋅τ_u  = rhs
            bound_dofs = hstack(bc_temp_dofs)
            solve_dofs = setdiff1d(arange(L.shape[0]), bound_dofs)
            lhs = L[solve_dofs].T[solve_dofs].T
            rhs = flows[solve_dofs] - L[solve_dofs].T[bound_dofs].T @ temps[bound_dofs]
            temps[solve_dofs] = spsolve(lhs, rhs)
        else:
            temps = spsolve(L, flows)
        return temps

    @staticmethod
    @disk_cache
    def conductance_from_surfaces_to_center(
        body_mesh: Mesh, surf_meshes: Mesh, matl, init_temp
    ):
        """Get conductance from body center cut plane to each surface

        This is needed for visualisation purposes only
        """
        meas_surf_meshes = Mesh(
            points=surf_meshes.points,
            cell_blocks=surf_meshes.cell_blocks.copy(),
            block_names=surf_meshes.block_names.copy(),
            block_types=surf_meshes.block_names.copy(),
        )
        # Add center plane for each surf
        body_center = body_mesh.center(0)  # for center plane
        for i, surf_name in enumerate(surf_meshes.block_names):
            surf_center = surf_meshes.center(i)
            norm_vec = surf_center - body_center
            cut_plane = body_mesh.extract_intersecting_plane_faces(
                0, body_center, norm_vec
            )
            meas_surf_meshes.cell_blocks.append(cut_plane)
            meas_surf_meshes.block_names.append(surf_name + '_center_plane')
            meas_surf_meshes.block_types.append('triangle')

        center_plane_part = FESystem(
            body_mesh=body_mesh,
            surf_meshes=meas_surf_meshes,
            init_temp=init_temp,
            matl=matl,
            lump_cond_meth=None,
            temp_dependent=False,
        )

        conds = []
        for i, surf_name in enumerate(surf_meshes.block_names):
            bcs = [
                HeatBC(Surf(surf_name), 1.0),
                TempBC(Surf(surf_name + '_center_plane'), temp=0.0),
            ]
            temps = center_plane_part.stationary_solution(bcs)
            mean_surf_temp = center_plane_part.C_surf_T[i] @ temps
            conds.append(1.0 / mean_surf_temp)

        return array(conds)

    @disk_cache
    def conductance_from_surface_to_surface(self):
        """Get conductance from surface to body center

        First, for all possible combinations the conductance from
        surface-to-surface gets measured. Second, an equation system is
        build, where the measured surface-to-surface conductance is
        computed by a series connection of two conductances which are
        assumed to be connected at the body's center of gravity.
        Third, the conductance values of this overdetermined system of
        equations are determined by an equalization calculus.

        This is needed for visualisation purposes only
        """
        # for all unique combinations of two surface names
        idx0s = []
        idx1s = []
        conds = []
        for comb in combinations(self.surf_names, 2):
            idx0 = self.surf_names.index(comb[0])
            idx1 = self.surf_names.index(comb[1])
            idx0s.append(idx0)
            idx1s.append(idx1)
            bcs = [HeatBC(Surf(comb[0]), 1.0), TempBC(Surf(comb[1]), temp=0.0)]
            temps = self.stationary_solution(bcs)
            mean_surf_temp = self.C_surf.T[idx0] @ temps
            if mean_surf_temp == 0.0:
                # This occurs if all nodes of one surface are located on the line of
                # contact between the two current surfaces.
                warn(
                    f"The computation of the lumped conductance between surfaces {comb}"
                    f" of a part failed."
                    f"Increase the mesh density to prevent inaccurate values."
                )
                mean_surf_temp = 1e-15
            conds.append(1.0 / mean_surf_temp)  # L = Q/delta_T
        L = zeros((len(conds), len(self.surf_names)))
        for i, idxs in enumerate(zip(idx0s, idx1s)):
            L[i, idxs] = 1.0
        b = array(conds)
        l, residuals, rank, s = linalg.lstsq(L, 1 / b, rcond=None)
        return abs(1 / l)
