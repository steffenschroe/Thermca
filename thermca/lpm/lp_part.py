from __future__ import annotations
from dataclasses import dataclass, field
from collections import namedtuple
from typing import NamedTuple

from thermca.lpm.asm import Asm
from thermca.lpm.cube import Cube
from thermca.lpm.cyl import Cyl
from thermca.materials import Solid  # for type hints
from thermca.baseelements import ModelElement, check_name, NetDat
from thermca.lpm.lp_construction import create_lp_elems

from numpy import array
from scipy import spatial


@dataclass
class LPPart(ModelElement):
    """Lumped parameter model of solid part

    Args:
        asm: An assembly defines the geometry as connected rectangular
            blocks and cylinders. Additionally, surfaces are defined as
            interfaces for heat exchange between the model elements.
        matl: Defines the material of the part. If the material has
            temperature dependent properties, the properties will be
            set for the initial temperature of the part.
        init_temp: Initial temperature
        posn: Origin position with the x, y and z coordinate
        temp_dependent: Switch for temperature dependent material
            behavior
        name: Name of the part.

    Attributes:
        surf: Coupling surfaces
    """

    asm: Asm
    matl: Solid
    init_temp: float = 0.0
    posn: tuple[float, float, float] = (0.0, 0.0, 0.0)
    temp_dependent: bool = False
    surf: NamedTuple[LPPartSurf] = field(init=False)
    name: str = ''
    _net_dat: NetDat = field(init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        lp_blocks, lp_cyls, lp_link_surfs, force_cont_pairs = create_lp_elems(self.asm)
        surf = {
            link_surf.name: LPPartSurf(
                self,
                link_surf.name,
                self.init_temp,
                self.posn,
                NetDat(
                    posn=array(self.posn),
                    dock_lctn=link_surf.dock_lctn(),
                    areas=array([link_surf.area()]),
                ),
                link_surf.area,
            )
            for link_surf in lp_link_surfs
        }
        SurfTuple = namedtuple('SfcTuple', surf)
        self.surf = SurfTuple(**surf)
        # Body "center" as central node location
        node_lctns = [
            posn
            for body in lp_blocks + lp_cyls
            for posn in body.node_lctns() + body.posn
        ]
        node_lctns = array(node_lctns)
        point_cloud_center = node_lctns.mean(axis=0)
        center_point_idx = spatial.KDTree(node_lctns).query(point_cloud_center)[1]
        dock_lctn = node_lctns[center_point_idx]
        self._net_dat = NetDat(
            posn=array(self.posn),
            dock_lctn=dock_lctn,
        )

    def vol(self):
        """Volume of the part"""
        # The volume is sometimes needed during model creation.
        # For this purpose calculate it using body data.
        block = self.asm._get_all_elems_of_type(Cube)
        cyl = self.asm._get_all_elems_of_type(Cyl)
        return sum(body.vol() for body in block + cyl)

    def __hash__(self):
        return id(self)


@dataclass
class LPPartSurf(ModelElement):
    """Coupling surface as surf for links in Model"""

    part: LPPart
    name: str
    init_temp: float
    posn: tuple[float, float, float]
    _net_dat: NetDat = field(repr=False)
    area: callable  # Coupling surface area

    def __post_init__(self):
        super().__init__()
        check_name(self.name)

    def _idx(self):
        return self.part.surf._fields.index(self.name)

    def __hash__(self):
        return id(self)
