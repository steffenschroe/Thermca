from __future__ import annotations
from collections import namedtuple
from typing import Optional, NamedTuple
from dataclasses import dataclass, field

from numpy import array

from thermca.mesh import Mesh, triangles_area, tetras_vol, block_center_point
from thermca.materials import Solid  # For type hints
from thermca.baseelements import ModelElement, NetDat, check_name

# Method to calculate lumped conductance parameters for visualisation
SFC_TO_CENTER_PLANE = 'surf_to_center_plane'
SFC_TO_SFC = 'surf_to_surf'


@dataclass
class FEPart(ModelElement):
    """Finite element model of solid part

    Args:
        mesh: Mesh defining the geometry of containing body and
            surfaces. The mesh contains one body cell block with
            tetrahedron cell types and at least one surface cell block
            with triangle cell types. The surfaces are interfaces for
            heat exchange between model elements.
        matl: Defines the material of the part. If the material has
            temperature dependent properties, the properties will be
            set for the initial temperature of the part.
        init_temp: Initial temperature
        posn: Origin position with the x, y and z coordinate
        mor_dof: Controls the degree of freedom (DOF) of the FE-system
            and thus influences its accuracy. If None, a full
            FE-system gets created. Otherwise, a number has to be
            given, defining the DOF of a MOR-system (reduced model
            order). For each coupling surface a MOR-system will be
            generated. Good results have been found for values in the
            range from 7 to 30 while 10 seems to be a good start.
        mor_ref_temp: Reference temperature for MOR simulation; For
            best accuracy, the temperatures should stay near this
            temperature during the simulation. If None, 'init_temp' is
            chosen.
        lump_cond_meth: Method to determine lumped conductance. The
            lumped conductance is used during visualisation. It shows
            the overall conductance of the surfaces to the body center.
            If set to SFC_TO_SFC, the method determines a mean
            conductance from the surfaces to the body center.
            Therefore, all combinations of conductances from surface
            to surface are calculated and an equalization calculus is
            performed assuming the heat is flowing through the body
            center. This can be very time-consuming in case of a higher
            number of surfaces.
            If set to SFC_TO_CENTER_PLANE, the method calculates the
            conductance to a plane that contains the body center and
            has a normal vector going through the particular surface
            center.
        fe_assembly: FEM package for assembly of system matrices; It
            defaults to 'skfem' (scikit-fem). As an alternative,
            'fenics' can be specified if the fenics dolfin package is
            installed.
        temp_dependent: Switch for temperature dependent material
            behavior; If the model has reduced DOF, temperature
            dependent material is not supported.
        name: Name of the part.

    Attributes:
        surf: Coupling surfaces
    """

    mesh: Mesh
    matl: Solid
    init_temp: float = 0.0
    posn: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mor_dof: Optional[int] = None
    mor_ref_temp: Optional[float] = None
    lump_cond_meth: str = SFC_TO_SFC
    fe_assembly: str = 'skfem'
    temp_dependent: bool = False
    surf: NamedTuple[FEPartSurf] = field(init=False)
    name: str = ''
    _net_dat: NetDat = field(init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        check_name(self.name)
        self._body_mesh = self.mesh.extract_type('tetra')
        self._surf_meshes = self.mesh.extract_type('triangle')

        names = self._body_mesh.block_names + self._surf_meshes.block_names
        for name in names:
            check_name(name)
        if len(names) != len(set(names)):
            raise Exception(f"Names of body and surfaces {names} are not unique!")

        if self._body_mesh.block_types != ['tetra'] or set(
            self._surf_meshes.block_types
        ) != {'triangle'}:
            raise Exception(
                "Body cell block must be one body of type 'tetra' the "
                "surface cell blocks must be of type 'triangle'!"
            )

        if self.temp_dependent and self.mor_dof is not None:
            raise Exception(
                "Order reduced part element does not support temperature dependent material properties."
            )

        if self.lump_cond_meth not in (None, SFC_TO_SFC, SFC_TO_CENTER_PLANE):
            raise Exception(
                f"Given lumped_cond_meth '{self.lump_cond_meth}' is not supported!"
            )

        self.init_temp = float(self.init_temp)

        surf = {
            surf_name: FEPartSurf(self, surf_name, self.init_temp, self.posn)
            for surf_name in self._surf_meshes.block_names
        }
        SfcTuple = namedtuple('SfcTuple', surf)
        self.surf = SfcTuple(**surf)

        if self.mor_dof is not None:
            self.mor_dof = int(self.mor_dof)
            if self.mor_dof < 1:
                raise Exception("The degree of freedom must be at least 1.")
            if self.mor_ref_temp is None:
                self.mor_ref_temp = self.init_temp
        if self.mor_ref_temp is not None:
            self.mor_ref_temp = float(self.mor_ref_temp)
        self._net_dat = NetDat(
            posn=array(self.posn),
            dock_lctn=block_center_point(
                self._body_mesh.points, self._body_mesh.cell_blocks[0]
            ),
        )

    def vol(self):
        """Volume of the part"""
        return tetras_vol(self._body_mesh.points, self._body_mesh.cell_blocks[0])

    def __hash__(self):
        return id(self)


@dataclass
class FEPartSurf(ModelElement):
    """Coupling surface as surf for links in Model"""

    part: FEPart
    name: str
    init_temp: float
    posn: tuple[float, float, float]
    _net_dat: NetDat = field(init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        dock_lctn = block_center_point(
            self.part._surf_meshes.points,
            self.part._surf_meshes.cell_blocks[self._idx()],
        )
        self._net_dat = NetDat(
            posn=array(self.posn), dock_lctn=dock_lctn, areas=array([self.area()])
        )

    def _idx(self):
        return self.part._surf_meshes.block_names.index(self.name)

    def area(self):
        """Coupling surface area"""
        points = self.part._surf_meshes.points
        cells = self.part._surf_meshes.cell_blocks[self._idx()]
        return triangles_area(points, cells)

    def __hash__(self):
        return id(self)
