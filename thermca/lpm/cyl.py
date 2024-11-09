"""Cylinder geometry element with axis in x-direction"""

from __future__ import annotations
from typing import NamedTuple, Optional
from dataclasses import dataclass, field
from math import tau


from thermca.lpm.asm import AssemblyElement, BodyFace

EQUAL_VOL = 'equal_vol'
EQUAL_RAD = 'equal_rad'


class Inner(BodyFace):
    def area(self):
        return tau*self.body.inner_rad*self.body.lgth


class Outer(BodyFace):
    def area(self):
        return tau*self.body.outer_rad*self.body.lgth


class Base(BodyFace):
    def area(self):
        return tau/2.*(self.body.outer_rad**2 - self.body.inner_rad**2)


class End(BodyFace):
    def area(self):
        return tau/2.*(self.body.outer_rad**2 - self.body.inner_rad**2)


class CylFaces(NamedTuple):
    inner: Optional[Inner]
    outer: Outer
    base: Base
    end: End


@dataclass
class Cyl(AssemblyElement):
    """Open or closed cylinder

    The cylinder represents axis-symmetric cylindrical temperature fields.
    Its axis runs along the horizontal x-axis.
    The body can be horizontally and vertically divided in sub-cylinders.

    Args:
        lgth: Length in axial direction; runs along the
            horizontal x-axis
        inner_rad: Inner radius
        outer_rad: Outer radius
        lgth_div: Number of segments in length direction
        rad_div: Number of segments in radial direction
        posn: Origin position
        face: Faces 'inner', 'outer', 'start' and 'end' of the cylinder.
        name: Name of the cylinder.
        rad_div_mode: It sets the mode to divide the cylinder in
            radial direction. 'EQUAL_VOL' gives equal volumes.
            'EQUAL_RAD' divides in equal radii segments.
    """
    name: str = ''
    lgth: float = 1.
    inner_rad: float = 0.
    outer_rad: float = 1.
    lgth_div: int = 1
    rad_div: int = 1
    posn: tuple[float, float, float] = (0., 0., 0.)
    face: CylFaces = field(init=False)
    rad_div_mode: str = EQUAL_VOL

    def __post_init__(self):
        AssemblyElement.__init__(self)
        if self.lgth_div < 1 or self.rad_div < 1:
            raise ValueError("Argument 'inner_rad', 'lgth_div' should be >= 1!")
        if self.inner_rad < 0. or self.inner_rad >= self.outer_rad < 0:
            raise ValueError("Argument 'inner_rad' should be >= 0. and < as 'inner_rad'!")
        if self.lgth <= 0. or self.outer_rad <= 0.:
            raise ValueError("Argument `lgth` and `outer_rad` should be > 0!")

        self.face = CylFaces(
            inner=(Inner('inner', self) if self.inner_rad > 0. else None),
            outer=Outer('outer', self),
            base=Base('base', self),
            end=End('end', self),
        )

    def vol(self):
        """Cyl volume"""
        return self.face.base.area() * self.lgth


