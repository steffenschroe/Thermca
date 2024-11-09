from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Union
from functools import wraps

from numpy import zeros, hstack, ones, ndarray, add, empty, sum as asum
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sparse import COO

from thermca.lpm.asm import Asm
from thermca.lpm.asm import Surf
from thermca.materials import Solid
from thermca.lpm.lp_construction import assemble
from thermca._utils.sparse import coo_diag_data_idxs, to_csr_data_idxs
from thermca.static_bcs import flatten_bcs, BCs
import thermca.plot.lp_sys as plot


class LPSystem:
    """Creates an LPM part system from lp assemblies

    The same approach as in 'FESystem' is used
    """

    def __init__(
        self,
        asm: Asm,
        init_temp: float,
        matl: Solid,
    ):
        (
            Mx_diag,
            Qs,
            Lx_base,
            posns,
            surf_names,
            surf_to_marg_idxs,
            surf_face_areas,
            dof,
            vol_sum,
        ) = assemble(asm)
        if dof < 2:
            Exception("Assembly must have at least 2 sub-volumes!")

        self.asm = asm
        self.init_temp = init_temp
        self.matl = matl

        self.surf_names = surf_names
        self.posns = posns
        self.Mx_diag = Mx_diag
        self.M_diag = matl.vol_capy_interp(init_temp) * Mx_diag

        self.Lx_base = Lx_base  # Lx without negated sum of rows on diagonal
        Lx_base_coo = Lx_base.tocoo(copy=False)
        self.Lx_base_coo = Lx_base_coo
        self.L_diag_data_idxs = coo_diag_data_idxs(
            Lx_base.shape, Lx_base.data.shape, Lx_base_coo.row, Lx_base_coo.col
        )

        # Conductance matrix containing geometry information only (x)
        self.Lx = Lx_base.copy()
        # self.Lx.data[self.L_diag_data_idxs] = -self.Lx_base[:dof].sum(axis=1).ravel()

        self.L = Lx_base.copy()  # Conductance matrix
        self.L.data = matl.condy_interp(init_temp) * Lx_base.data
        # Placing the sum of the rows on the diagonal is only done for the inner
        # conductances because the margin conductances are added to the film coefficient
        # for temperature field computation.
        self.L.data[self.L_diag_data_idxs] = -self.L[:dof].sum(axis=1).ravel()

        self.surf_face_areas = surf_face_areas
        self.surf_to_marg_data_idxs = [
            to_csr_data_idxs(self.L, smidxs) for smidxs in surf_to_marg_idxs
        ]
        self.Qs = Qs  # Load vector containing surface geometry information only (s)

        # Node-area weighted output matrix for margin mean temperatures
        C_marg = Qs.copy()
        for c in C_marg.T:
            nz_mask = c != 0.0
            num_dofs = asum(nz_mask)
            c_nz = c[nz_mask]
            c_mean = asum(c_nz) / num_dofs
            c_area_wgt = c_nz / c_mean
            c[nz_mask] = c_area_wgt * 1.0 / num_dofs
        self.C_marg = C_marg

        # Output matrix for volume weighted body mean temps
        m_mean = asum(Mx_diag) / dof
        m_vol_wgt = Mx_diag / m_mean  # relative weight regarding node volume
        self.C_body = m_vol_wgt * 1.0 / dof

        # self.C_marg_T = C_marg.T  # Transpose for fast access
        # self.vert_to_dof_idx = vert_to_dof_idx
        # self.dof_to_vert_idx = dof_to_vert_idx
        self.dof = dof  # Number of unknown temperatures
        self.marg_dof_idxs = [marg for surf, marg in surf_to_marg_idxs]
        self.surf_dof_idxs = [
            surf for surf, marg in surf_to_marg_idxs
        ]  # Indices of 'virtual' surface nodes
        self.surf_areas = Qs.sum(axis=0)
        self.vol = vol_sum

    def update_material_properties(self, body_temps):
        self.matl.check_limits(body_temps)
        body_and_surf_temps = empty(self.L.shape[0])
        body_and_surf_temps[: self.dof] = body_temps
        for marg_idxs, surf_idxs in zip(self.marg_dof_idxs, self.surf_dof_idxs):
            body_and_surf_temps[surf_idxs] = body_temps[marg_idxs]
        cond_temp = (
            body_and_surf_temps[self.Lx_base_coo.row]
            + body_and_surf_temps[self.Lx_base_coo.col]
        ) / 2.0
        self.L.data = self.matl.condy_interp(cond_temp) * self.Lx_base.data
        self.L.data[self.L_diag_data_idxs] = -self.L[: self.dof].sum(axis=1).ravel()
        self.M_diag = self.matl.vol_capy_interp(body_temps) * self.Mx_diag

    def solve(self, bound_condns: Iterable[BCs]) -> StaticResult:
        """Stationary (equilibrium) solution for LP-system

            * Temperature dependent material properties and film
              coefficients are calculated with initial temperatures.
            * Heat boundary conditions are applied on margin nodes
              which are inside the part. Note that conductances from
              margin nodes to surface have no effect. This may have a
              significant negative effect on the accuracy and depends
              on the magnitude of the conductances.
            * Temperature boundary conditions are applied on virtual
              nodes directly on the surface. Therefore, conductances
              from margin nodes to surfaces effect the temperature
              field.
            * Film boundary conditions are temperature boundary
              conditions applied on virtual film nodes. These nodes lie
              outside the part and connected to the part surface by a
              film coefficient. The film nodes are connected to the
              margin nodes by a series connection of the conductance of
              the inner part-margin and the outer film-conductance.

        Args:
            bound_condns: Iterable of boundary conditions on coupling
                surfaces. Boundary conditions may be of type: `TempBC`,
                `HeatBC`, `FluxBC` or `FilmBC`.

        Returns:
            Result object
        """
        surf_names = self.surf_names
        surf_areas = self.surf_areas

        flat_bcs = flatten_bcs(bound_condns, surf_areas, surf_names)
        (
            bc_temp_idxs,
            bc_temps,
            bc_heat_idxs,
            bc_heats,
            bc_film_idxs,
            bc_films,
            bc_film_temps,
        ) = flat_bcs

        secondary_bc = set(bc_temp_idxs).intersection(bc_heat_idxs + bc_film_idxs)
        if secondary_bc:
            raise Exception(
                f"The surface(s): {[surf_names[i] for i in secondary_bc]} have secondary "
                f"boundary conditions beneath temperature boundary condition(s). This "
                f"does not make sense physically!"
            )

        # Heat sources
        flows = zeros(self.dof)
        if bc_heats.size:
            surf_fluxes = bc_heats / surf_areas[bc_heat_idxs]
            flows += (self.Qs[:, bc_heat_idxs] @ surf_fluxes).ravel()

        # Flows over margins for result output
        res_marg_flows = zeros((self.dof, len(self.surf_names)))
        for heat_idx, hdofs in zip(
            bc_heat_idxs, [self.marg_dof_idxs[hi] for hi in bc_heat_idxs]
        ):
            res_marg_flows[hdofs, heat_idx] = flows[hdofs]

        # Film coefficients
        if bc_films.size > 0:
            # Series connection of margin and film for every surface
            L_film_margs = zeros((self.dof, len(bc_films)))
            for i, (face_areas, film, smxdxs, marg_idxs) in enumerate(
                zip(
                    [self.surf_face_areas[fi] for fi in bc_film_idxs],
                    bc_films,
                    [self.surf_to_marg_data_idxs[fi] for fi in bc_film_idxs],
                    [self.marg_dof_idxs[fi] for fi in bc_film_idxs],
                )
            ):
                # Film and margin conductance for each surface face
                film_conds = face_areas * film
                marg_conds = -self.L.data[smxdxs]
                # Compute series conductance and sum it up on edges where one margin node
                # may be connected to multiple surface nodes.
                # L_film_margs[marg_idxs, i] += film_conds * marg_conds / (film_conds + marg_conds)
                add.at(
                    L_film_margs[:, i],
                    marg_idxs,
                    film_conds * marg_conds / (film_conds + marg_conds),
                )
            L = self.L[: self.dof] + asum(
                [diags(L_film_marg, format='csr') for L_film_marg in L_film_margs.T]
            )
            flows += (L_film_margs @ bc_film_temps).ravel()
        else:
            L = self.L[: self.dof]

        # Bound temperatures
        if bc_temps.size > 0:
            # temps = empty(self.dof)
            bc_temp_dofs = [
                self.surf_dof_idxs[i] for i in bc_temp_idxs
            ]  # DOFs at boundary
            bc_surf_temps = [
                ones(len(self.surf_dof_idxs[i])) * temp
                for i, temp in zip(bc_temp_idxs, bc_temps)
            ]
            bound_dofs = hstack(bc_temp_dofs)
            bound_temps = hstack(bc_surf_temps)
            # Transform the system to bind the given boundary temperatures.
            # First, split the system in known k and unknown u temperatures:
            # ⎡L_uu L_uk⎤ ⎧τ_u⎫ = ⎧q̇_u⎫
            # ⎣L_ku L_kk⎦ ⎩τ_k⎭   ⎩q̇_k⎭
            # Second, extract the equation system for the unknown temperatures:
            # L_uu⋅τ_u = q̇_u - L_uk⋅τ_k
            # lhs⋅τ_u  = rhs
            # Use 'virtual' surface nodes of lumped parameter models for known boundary
            # temperatures τ_k. These nodes follow the inner nodes in the conductance
            # matrix L. All inner nodes are unknown temperatures τ_u.
            # L_uu⋅τ_u = q̇_u - L_kuᵀ⋅τ_k
            # lhs⋅τ_u  = rhs
            L_ku = self.L[bound_dofs]
            # Add margin conductances stored in last rows to diagonal elements with negated sum
            L.data[self.L_diag_data_idxs] = (
                L.data[self.L_diag_data_idxs] - L_ku.sum(axis=0).data
            )
            rhs = flows - L_ku.T @ bound_temps
            lhs = L
        else:
            rhs = flows
            lhs = L

        # Resulting temperature field of "unknown" inner nodes
        temps = spsolve(lhs, rhs)

        # # Result data
        # Flows over margins into margin nodes
        if bc_temps.size > 0:
            L_marg = zeros((self.dof, len(bc_temp_idxs)))  # Margin conductance matrix
            for i, (midxs, smidxs) in enumerate(
                zip(
                    [self.marg_dof_idxs[ti] for ti in bc_temp_idxs],
                    [self.surf_to_marg_data_idxs[ti] for ti in bc_temp_idxs],
                )
            ):
                # add.at(a, indices, b) is equivalent to a[indices] += b,
                # except results are accumulated for indexed elements more than once.
                # self.L_marg[midxs, i] += self.L.data[smidxs]
                add.at(L_marg[:, i], midxs, -self.L.data[smidxs])

            # L_ku_coo = L_ku.tocoo()
            #  flows[L_ku_coo.row] = L_ku_coo.data * (temps[L_ku_coo.row] - temps[L_ku_coo.col])
            # for marg_idxs, bc_temp_idx in self.marg_dof_idxs, bc_temp_idxs:
            # res_marg_flows[marg_idxs, bc_temp_idx] = flows[marg_idxs]
            delta_temps = bc_temps - temps[:, None]
            res_marg_flows[:, bc_temp_idxs] = L_marg * delta_temps
        if bc_films.size > 0:
            delta_temps = bc_film_temps - temps[:, None]
            res_marg_flows[:, bc_film_idxs] = L_film_margs * delta_temps

        return StaticResult(self, temps, res_marg_flows, flat_bcs)


@dataclass
class StaticResult:
    lp_sys: LPSystem
    temps: ndarray
    marg_flows: ndarray
    flat_bcs: tuple

    def __getitem__(self, elem: Union[Surf, LPSystem]) -> ResultProcessing:
        return ResultProcessing(
            self.lp_sys,
            self.temps,
            self.marg_flows,
            elem,
        )

    @wraps(plot.static_result)
    def plot(self, *args, **kwargs):
        return plot.static_result(self, *args, **kwargs)


@dataclass
class ResultProcessing:
    """Result processing based lumped nodes"""

    # Focus on numerical results for users not visualisation
    lp_sys: LPSystem
    temps: ndarray
    res_marg_flows: ndarray
    elem: Union[Surf, LPSystem]

    def temp(self):
        """Temperature of part element

        This means the temperature of the margin nodes for surfaces
        and the temperature of the inner nodes including the margin
        nodes for the part body."""
        if self.elem == self.lp_sys:
            return self.lp_sys.C_body @ self.temps
        surf_idx = self.lp_sys.surf_names.index(self.elem.name)
        return self.lp_sys.C_marg.T[surf_idx] @ self.temps

    def heat(self):
        """Mean heat flowing over margin into part"""
        surf_idx = self.lp_sys.surf_names.index(self.elem.name)
        return asum(self.res_marg_flows.T[surf_idx])

    def _heat_field(self):
        if self.elem == self.lp_sys:
            # temp = COO.from_numpy(res_temp)
            # dtemp = (temp[:, None] - temp)  # May consume big amount of memory
            # res_heat = cond.to_coo()*dtemp
            cond = COO.from_scipy_sparse(self.lp_sys.L[: self.lp_sys.dof])
            cond_idxs = cond.nonzero()
            dtemp = COO(
                cond_idxs,
                self.temps[cond_idxs[0]] - self.temps[cond_idxs[1]],
                cond.shape,
            )
            return (cond * dtemp).tocsr()
        # surf_idx = self.lp_part.surf_names.index(self.elem.name)
        # return self.lp_part.C_marg.T[surf_idx]
