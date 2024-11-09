from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from numpy import array
from scipy import spatial

from thermca.lpm.asm import Surf
from thermca.lpm.cube import Cube
from thermca.lpm.cyl import Cyl

if TYPE_CHECKING:
    from thermca.lpm.lp_cube import LPCube
    from thermca.lpm.lp_cyl import LPCyl


@dataclass
class LPFace:
    name: str
    body: Union[LPCyl, LPCube]

    def node_count(self):
        return len(self.body_marg_node_idxs())

    def __hash__(self):
        return id(self)


@dataclass
class LPLinkSurf:
    """Mirrors link surface but contains LP bodies instead of bodies"""

    faces: list
    name: str
    area: callable

    @classmethod
    def from_link_surf(
        cls,
        link_surf: Surf,
        blocks: list[Cube],
        cyls: list[Cyl],
        lp_blocks: list[LPCube],
        lp_cyls: list[LPCyl],
    ):
        from thermca.lpm.lp_construction import faces_to_lp_faces

        return cls(
            faces_to_lp_faces(link_surf.faces, blocks, cyls, lp_blocks, lp_cyls),
            link_surf.name,
            link_surf.area,
        )

    def dock_lctn(self):
        docks = array([face.dock_lctn() + face.body.posn for face in self.faces])
        center_point_idx = spatial.KDTree(docks).query(docks.mean(axis=0))[1]
        return docks[center_point_idx]
