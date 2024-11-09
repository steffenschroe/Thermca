"""Axis aligned rectangular cuboid geometry element"""

from __future__ import annotations
from typing import NamedTuple
from dataclasses import dataclass, field


from thermca.lpm.asm import AssemblyElement, BodyFace


class Left(BodyFace):
    def area(self):
        return self.body.hgt*self.body.depth


class Right(BodyFace):
    def area(self):
        return self.body.hgt*self.body.depth


class Btm(BodyFace):
    def area(self):
        return self.body.width*self.body.depth


class Top(BodyFace):
    def area(self):
        return self.body.width*self.body.depth


class Back(BodyFace):
    def area(self):
        return self.body.hgt*self.body.width


class Front(BodyFace):
    def area(self):
        return self.body.hgt*self.body.width


class CubeFaces(NamedTuple):
    left: Left
    right: Right
    btm: Btm
    top: Top
    back: Back
    front: Front


@dataclass
class Cube(AssemblyElement):
    """Axis aligned rectangular cuboid.

    The cuboid can be divided in volume-equal sub-blocks.

    Args:
        width: Width in horizontal x-direction
        hgt: Height in vertical y-direction
        depth: Depth in z-direction
        width_div: Number of segments in width direction
        hgt_div: Number of segments in height direction
        depth_div: Number of segments in depth direction
        posn: Origin position
        face: Faces 'lef', 'right', 'btm', 'top', 'back' and 'front'
            of the block.
        name: Block name
    """
    name: str = ''
    width: float = 1.
    hgt: float = 1.
    depth: float = 1.
    width_div: int = 1
    hgt_div: int = 1
    depth_div: int = 1
    posn: tuple[float, float, float] = (0., 0., 0.)
    face: CubeFaces = field(init=False)

    def __post_init__(self):
        AssemblyElement.__init__(self)
        if self.width <= 0. or self.hgt <= 0. or self.depth <= 0.:
            raise ValueError("The arguments `width` and `hgt` and `depth` "
                             "should be > 0.")
        if self.width_div < 1 or self.hgt_div < 1 or self.depth_div < 1:
            raise ValueError("The subdivision in each dimension should be >= 0.")
        self.face = CubeFaces(
            left=Left('left', self),
            right=Right('right', self),
            btm=Btm('btm', self),
            top=Top('top', self),
            back=Back('back', self),
            front=Front('front', self),
        )

    def vol(self):
        """Cube volume"""
        return self.width*self.hgt*self.depth
