"""Functionality for cylinder lumped parameter models."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, NamedTuple
from math import tau

from numpy import array, linspace, zeros, ndarray, arange, int64, log
from sparse import DOK

from thermca.lpm.cyl import Cyl, EQUAL_RAD, EQUAL_VOL
from thermca.lpm.lp_link_surf import LPFace


class LPBase(LPFace):
    def center(self):
        return array([0.0, 0.0, 0.0])

    def dock_lctn(self):
        return array(
            [
                0.0,
                self.body.inner_rad + (self.body.outer_rad - self.body.inner_rad) / 2.0,
                0.0,
            ]
        )

    def dock_dirn(self):
        return array([-1.0, 0.0, 0.0])

    def body_marg_node_idxs(self):
        return self.body.node_idxs_2d()[:, 0]

    def node_lctns(self):
        body = self.body
        lctns = zeros((body.rad_div, 3))
        # node_lctns[:, 0] = 0.
        lctns[:, 1] = body.rad_node_lctns()
        # node_lctns[:, 2] = 0.
        return lctns

    def node_areas(self):
        body = self.body
        return tau / 2.0 * (body.rad_borders()[1:] ** 2 - body.rad_borders()[:-1] ** 2)

    def node_geras(self):
        body = self.body
        return self.node_areas() / (
            body.axial_node_lctns()[0] - body.axial_borders()[0]
        )


class LPEnd(LPFace):
    def center(self):
        return array([self.body.lgth, 0.0, 0.0])

    def dock_lctn(self):
        return array(
            [
                self.body.lgth,
                self.body.inner_rad + (self.body.outer_rad - self.body.inner_rad) / 2.0,
                0.0,
            ]
        )

    def dock_dirn(self):
        return array([1.0, 0.0, 0.0])

    def body_marg_node_idxs(self):
        return self.body.node_idxs_2d()[:, -1]

    def node_lctns(self):
        body = self.body
        lctns = zeros((body.rad_div, 3))
        lctns[:, 0] = body.axial_borders()[-1]
        lctns[:, 1] = body.rad_node_lctns()
        return lctns

    def node_areas(self):
        return self.body.face.base.node_areas()

    def node_geras(self):
        return self.body.face.base.node_geras()


class LPInner(LPFace):
    def rad(self):
        return self.body.inner_rad

    def dock_lctn(self):
        return array([self.body.lgth / 2, self.body.inner_rad, 0.0])

    def dock_dirn(self):
        return array([0.0, -1.0, 0.0])

    def body_marg_node_idxs(self):
        return self.body.node_idxs_2d()[0, :]

    def node_lctns(self):
        body = self.body
        lctns = zeros((body.lgth_div, 3))
        lctns[:, 0] = body.axial_node_lctns()
        lctns[:, 1] = body.rad_borders()[0]
        return lctns

    def node_areas(self):
        body = self.body
        return (
            tau
            * body.rad_borders()[0]
            * (body.axial_borders()[1:] - body.axial_borders()[:-1])
        )

    def node_geras(self):
        body = self.body
        return (
            tau
            * (body.axial_borders()[1:] - body.axial_borders()[:-1])
            / log(body.rad_node_lctns()[0] / body.rad_borders()[0])
        )


class LPOuter(LPFace):
    def rad(self):
        return self.body.outer_rad

    def dock_lctn(self):
        return array([self.body.lgth / 2.0, self.body.outer_rad, 0.0])

    def dock_dirn(self):
        return array([0.0, 1.0, 0.0])

    def body_marg_node_idxs(self):
        return self.body.node_idxs_2d()[-1, :]

    def node_lctns(self):
        body = self.body
        lctns = zeros((body.lgth_div, 3))
        lctns[:, 0] = body.axial_node_lctns()
        lctns[:, 1] = body.rad_borders()[-1]
        return lctns

    def node_areas(self):
        body = self.body
        return (
            tau
            * body.rad_borders()[-1]
            * (body.axial_borders()[1:] - body.axial_borders()[:-1])
        )

    def node_geras(self):
        body = self.body
        return (
            tau
            * (body.axial_borders()[1:] - body.axial_borders()[:-1])
            / log(body.rad_borders()[-1] / body.rad_node_lctns()[-1])
        )


class CylFaces(NamedTuple):
    inner: Optional[LPInner]
    outer: LPOuter
    base: LPBase
    end: LPEnd


@dataclass
class LPCyl:
    name: str
    lgth: float
    inner_rad: float
    outer_rad: float
    lgth_div: int
    rad_div: int
    posn: ndarray
    dirn: ndarray
    face: CylFaces = field(init=False)
    rad_div_mode: str

    @classmethod
    def from_cyl(cls, cyl: Cyl):
        lp_cyl = cls(
            name=cyl.name,
            lgth=cyl.lgth,
            inner_rad=cyl.inner_rad,
            outer_rad=cyl.outer_rad,
            lgth_div=cyl.lgth_div,
            rad_div=cyl.rad_div,
            posn=array(cyl.posn),
            dirn=array((1.0, 0.0, 0.0)),
            rad_div_mode=cyl.rad_div_mode,
        )
        lp_cyl.face = CylFaces(
            inner=(LPInner('inner', lp_cyl) if lp_cyl.inner_rad > 0.0 else None),
            outer=LPOuter('outer', lp_cyl),
            base=LPBase('base', lp_cyl),
            end=LPEnd('end', lp_cyl),
        )
        return lp_cyl

    def axial_borders(self):
        """Local x-coordinates of axial sub-cylinder borders"""
        return get_equal_lgth_border_lctns(0.0, self.lgth, self.lgth_div)

    def rad_borders(self):
        if self.rad_div_mode == EQUAL_VOL:
            return get_equal_vol_border_rads(
                self.inner_rad, self.outer_rad, self.rad_div
            )
        elif self.rad_div_mode == EQUAL_RAD:
            return get_equal_lgth_border_lctns(
                self.inner_rad, self.outer_rad, self.rad_div
            )
        else:
            raise ValueError("'rad_div_mode' must be 'EQUAL_RAD' or 'EQUAL_VOL'.")

    def center(self):
        return array(
            [self.lgth / 2, self.inner_rad + (self.outer_rad - self.inner_rad) / 2, 0.0]
        )

    def vol(self):
        return tau / 2.0 * (self.outer_rad**2 - self.inner_rad**2) * self.lgth

    def node_count(self):
        return self.lgth_div * self.rad_div

    # def node_idxs(self):
    #     return arange(self.node_count(), dtype=int64)

    def node_idxs_2d(self):
        # return self.node_idxs().reshape((self.lgth_div, self.rad_div))
        return arange(self.node_count(), dtype=int64).reshape(
            (self.rad_div, self.lgth_div)
        )

    def node_vols(self):
        y_border = self.rad_borders()
        x_border = self.axial_borders()
        base_areas = tau / 2 * (y_border[1:] ** 2 - y_border[:-1] ** 2)
        volus = zeros(self.node_count())
        volus_2d = volus.reshape((self.rad_div, self.lgth_div))
        for rad_idx in range(self.rad_div):
            volus_2d[rad_idx, :] = base_areas[rad_idx] * (x_border[1:] - x_border[:-1])
        return volus

    def node_lctns(self):
        lctns = zeros((self.node_count(), 3))
        lctns_2d = lctns.reshape((self.rad_div, self.lgth_div, 3))
        for rad_idx in range(self.rad_div):
            lctns_2d[rad_idx, :, 0] = self.axial_node_lctns()[:]
            lctns_2d[rad_idx, :, 1] = self.rad_node_lctns()[rad_idx]
            # lctns_2d[di, :, 2] = 0.
        return lctns

    def axial_node_lctns(self):
        return get_equal_lgth_node_lctns(0.0, self.lgth, self.lgth_div)

    def rad_node_lctns(self):
        if self.rad_div_mode == EQUAL_VOL:
            return get_equal_vol_node_rads(self.inner_rad, self.outer_rad, self.rad_div)
        elif self.rad_div_mode == EQUAL_RAD:
            return get_equal_lgth_node_lctns(
                self.inner_rad, self.outer_rad, self.rad_div
            )

    def node_geras(self):
        xdivs, ydivs = self.lgth_div, self.rad_div
        x_node, y_node = self.axial_node_lctns(), self.rad_node_lctns()
        x_border, y_border = self.axial_borders(), self.rad_borders()
        # gera2d = zeros((ydivs*xdivs)**2).reshape(ydivs*xdivs, ydivs*xdivs)
        gera4d = DOK(
            (ydivs, xdivs, ydivs, xdivs)
        )  # Make it accessible over axes-indices
        data4d = (
            gera4d.data
        )  # Speed up DOK.__setitem__, because there are many costly checks
        # Geometric ratios for conductances in x-direction
        for yi in range(ydivs):
            for xi in range(xdivs - 1):
                data4d[(yi, xi, yi, xi + 1)] = (
                    tau
                    / 2.0
                    * (y_border[yi + 1] ** 2 - y_border[yi] ** 2)
                    / (x_node[xi + 1] - x_node[xi])
                )
        # Geometric ratios for conductances in y-direction
        for yi in range(ydivs - 1):
            for xi in range(xdivs):
                data4d[(yi, xi, yi + 1, xi)] = (
                    tau
                    * (x_border[xi + 1] - x_border[xi])
                    / log(y_node[yi + 1] / y_node[yi])
                )
        gera2d = gera4d.to_coo().reshape(
            (ydivs * xdivs, ydivs * xdivs)
        )  # Reshape is only supported by COO
        gera2d = DOK.from_coo(gera2d)
        data = gera2d.data
        # Mirror along diagonal
        for k, v in list(data.items()):
            data[(k[1], k[0])] = v
        return gera2d

    def bounding_box(self):
        """The bounding box of the points of the mesh.

        A 2dimensional array where the first row contains the minimum
        values in each dimension and the second row contains the maximum
        values.
        """
        p, l, o = self.posn, self.lgth, self.outer_rad
        return array([[p[0], p[1] - o, p[2] - o], [p[0] + l, p[1] + o, p[2] + o]])

    def __hash__(self):
        return id(self)


def get_equal_lgth_border_lctns(lctn0, lctn1, num_div):
    """Local coordinates of equal length sub-block borders"""
    return linspace(lctn0, lctn1, num_div + 1)


def get_equal_lgth_node_lctns(lctn0, lctn1, num_div):
    """Local coordinates of equal length sub-block nodes"""
    bl = get_equal_lgth_border_lctns(lctn0, lctn1, num_div)
    dl2 = (bl[1] - bl[0]) / 2
    return linspace(bl[0] + dl2, bl[num_div] - dl2, num_div)


def get_equal_vol_border_rads(inner_rad, outer_rad, num_div):
    """Local coordinates of radial sub-block borders for equal volumes"""
    border_rads = zeros(num_div + 1)
    border_rads[0] = inner_rad  # First border is inner radius
    for i in range(1, num_div, 1):
        pir = border_rads[i - 1]  # Previous inner radius
        ndiv = num_div - i + 1  # Reduce divisions
        border_rads[i] = (
            ndiv * (ndiv * pir**2 + outer_rad**2 - pir**2)
        ) ** 0.5 / ndiv
    border_rads[num_div] = outer_rad
    return border_rads


def get_equal_vol_node_rads(inner_rad, outer_rad, num_div):
    """Local coordinates of radial sub-block borders for equal volumes"""
    node_rads = get_equal_vol_border_rads(inner_rad, outer_rad, num_div * 2)
    return node_rads[1::2]
