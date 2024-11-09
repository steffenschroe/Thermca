"""Functionality for cube lumped parameter models."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple

from numpy import array, linspace, ndarray, arange, int64, zeros, tile, full
from sparse import DOK

from thermca.lpm.cube import Cube
from thermca.lpm.lp_link_surf import LPFace


class LPLeft(LPFace):
    def center(self):
        return array([0., self.body.hgt / 2., self.body.depth / 2.])

    def dock_lctn(self):
        return self.center()

    def dock_dirn(self):
        return array([-1., 0., 0.])

    def verts(self):
        hgt, depth = self.body.hgt, self.body.depth
        return array(
            [[0., 0., 0., 0.],
             [0., 0., hgt, hgt],
             [0., depth, depth, 0.]]
        ).T

    def body_marg_node_idxs(self):
        return self.body.node_idxs_3d()[:, :, 0].ravel()

    def node_lctns(self):
        body = self.body
        lctns = zeros((body.depth_div*body.hgt_div, 3))
        node_lctns_2d = lctns.reshape((body.depth_div, body.hgt_div, 3))
        for zi in range(body.depth_div):
            # node_lctns_2d[zi, :, 0] = 0.
            node_lctns_2d[zi, :, 1] = body.y_node_lctns()[:]
            node_lctns_2d[zi, :, 2] = body.z_node_lctns()[zi]
        return lctns

    def node_areas(self):
        body = self.body
        return tile(
            (body.y_borders()[1:] - body.y_borders()[:-1])
            * (body.z_borders()[1] - body.z_borders()[0]),
            body.depth_div
        )

    def node_geras(self):
        body = self.body
        return self.node_areas() / (body.x_node_lctns()[0] - body.x_borders()[0])


class LPRight(LPFace):
    def center(self):
        return array([self.body.width,  self.body.hgt / 2.,  self.body.depth / 2.])

    def dock_lctn(self):
        return self.center()

    def dock_dirn(self):
        return array([1., 0., 0.])

    def verts(self):
        width, hgt, depth = self.body.width, self.body.hgt, self.body.depth
        return array(
            [[width, width, width, width],
             [0., 0., hgt, hgt],
             [0., depth, depth, 0.]]
        ).T

    def body_marg_node_idxs(self):
        return self.body.node_idxs_3d()[:, :, -1].ravel()

    def node_lctns(self):
        body = self.body
        lctns = zeros((body.depth_div * body.hgt_div, 3))
        node_lctns_2d = lctns.reshape((body.depth_div, body.hgt_div, 3))
        for zi in range(body.depth_div):
            node_lctns_2d[zi, :, 0] = body.x_borders()[-1]
            node_lctns_2d[zi, :, 1] = body.y_node_lctns()[:]
            node_lctns_2d[zi, :, 2] = body.z_node_lctns()[zi]
        return lctns

    def node_areas(self):
        return self.body.face.left.node_areas()

    def node_geras(self):
        return self.body.face.left.node_geras()


class LPBtm(LPFace):
    def center(self):
        return array([self.body.width/2., 0.,  self.body.depth/2.])

    def dock_lctn(self):
        return self.center()

    def dock_dirn(self):
        return array([0., -1., 0.])

    def verts(self):
        width, depth = self.body.width, self.body.depth
        return array(
            [[0., 0., width, width],
             [0., 0.,  0., 0.],
             [0., depth,  depth, 0.]]
        ).T

    def body_marg_node_idxs(self):
        return self.body.node_idxs_3d()[:, 0, :].ravel()

    def node_lctns(self):
        body = self.body
        lctns = zeros((body.depth_div * body.width_div, 3))
        node_lctns_2d = lctns.reshape((body.depth_div, body.width_div, 3))
        for zi in range(body.depth_div):
            node_lctns_2d[zi, :, 0] = body.x_node_lctns()[:]
            # node_lctns_2d[zi, :, 1] = 0.
            node_lctns_2d[zi, :, 2] = body.z_node_lctns()[zi]
        return lctns

    def node_areas(self):
        body = self.body
        return tile(
            (body.x_borders()[1:] - body.x_borders()[:-1])
            * (body.z_borders()[1] - body.z_borders()[0]),
            body.depth_div
        )

    def node_geras(self):
        body = self.body
        return self.node_areas() / (body.y_node_lctns()[0] - body.y_borders()[0])


class LPTop(LPFace):
    def center(self):
        return array([self.body.width/2., self.body.hgt, self.body.depth/2.])

    def dock_lctn(self):
        return self.center()

    def dock_dirn(self):
        return array([0., 1., 0.])

    def verts(self):
        width, hgt, depth = self.body.width, self.body.hgt, self.body.depth
        return array(
            [[0., 0., width, width],
             [hgt, hgt,  hgt, hgt],
             [0., depth,  depth, 0.]]
        ).T

    def body_marg_node_idxs(self):
        return self.body.node_idxs_3d()[:, -1, :].ravel()

    def node_lctns(self):
        body = self.body
        lctns = zeros((body.depth_div * body.width_div, 3))
        node_lctns_2d = lctns.reshape((body.depth_div, body.width_div, 3))
        for zi in range(body.depth_div):
            node_lctns_2d[zi, :, 0] = body.x_node_lctns()[:]
            node_lctns_2d[zi, :, 1] = body.y_borders()[-1]
            node_lctns_2d[zi, :, 2] = body.z_node_lctns()[zi]
        return lctns

    def node_areas(self):
        return self.body.face.btm.node_areas()

    def node_geras(self):
        return self.body.face.btm.node_geras()


class LPBack(LPFace):
    def center(self):
        return array([self.body.width/2., self.body.hgt/2., 0.])

    def dock_lctn(self):
        return self.center()

    def dock_dirn(self):
        return array([0., 0., -1.])

    def verts(self):
        width, hgt, depth = self.body.width, self.body.hgt, self.body.depth
        return array(
            [[0., 0., width, width],
             [0., hgt, hgt, 0.],
             [0., 0., 0., 0.]]
        ).T

    def body_marg_node_idxs(self):
        return self.body.node_idxs_3d()[0, :, :].ravel()

    def node_lctns(self):
        body = self.body
        lctns = zeros((body.hgt_div * body.width_div, 3))
        node_lctns_2d = lctns.reshape((body.hgt_div, body.width_div, 3))
        for yi in range(body.hgt_div):
            node_lctns_2d[yi, :, 0] = body.x_node_lctns()[:]
            node_lctns_2d[yi, :, 1] = body.y_node_lctns()[yi]
            node_lctns_2d[yi, :, 2] = body.z_borders()[-1]
        return lctns

    def node_areas(self):
        body = self.body
        return tile(
            (body.x_borders()[1:] - body.x_borders()[:-1])
            * (body.y_borders()[1] - body.y_borders()[0]),
            body.hgt_div
        )

    def node_geras(self):
        body = self.body
        return self.node_areas() / (body.z_node_lctns()[0] - body.z_borders()[0])


class LPFront(LPFace):
    def center(self):
        return array([self.body.width/2., self.body.hgt/2., self.body.depth])

    def dock_lctn(self):
        return self.center()

    def dock_dirn(self):
        return array([0., 0., 1.])

    def verts(self):
        width, hgt, depth = self.body.width, self.body.hgt, self.body.depth
        return array(
            [[0., 0., width, width],
             [0., hgt, hgt, 0.],
             [depth, depth, depth, depth]]
        ).T

    def body_marg_node_idxs(self):
        return self.body.node_idxs_3d()[-1, :, :].ravel()

    def node_lctns(self):
        body = self.body
        lctns = zeros((body.hgt_div * body.width_div, 3))
        node_lctns_2d = lctns.reshape((body.hgt_div, body.width_div, 3))
        for yi in range(body.hgt_div):
            node_lctns_2d[yi, :, 0] = body.x_node_lctns()[:]
            node_lctns_2d[yi, :, 1] = body.y_node_lctns()[yi]
            # node_lctns_2d[yi, :, 2] = self._z.border[0]
        return lctns

    def node_areas(self):
        return self.body.face.back.node_areas()

    def node_geras(self):
        return self.body.face.back.node_geras()


class CubeFaces(NamedTuple):
    left: LPLeft
    right: LPRight
    btm: LPBtm
    top: LPTop
    back: LPBack
    front: LPFront


@dataclass
class LPCube:
    name: str
    width: float
    hgt: float
    depth: float
    width_div: int
    hgt_div: int
    depth_div: int
    posn: ndarray
    face: CubeFaces = field(init=False)

    @classmethod
    def from_cube(cls, cube: Cube):
        lp_cube = cls(
            name=cube.name,
            width=cube.width,
            hgt=cube.hgt,
            depth=cube.depth,
            width_div=cube.width_div,
            hgt_div=cube.hgt_div,
            depth_div=cube.depth_div,
            posn=array(cube.posn),
        )
        face = CubeFaces(
            left=LPLeft('left', lp_cube),
            right=LPRight('right', lp_cube),
            btm=LPBtm('btm', lp_cube),
            top=LPTop('top', lp_cube),
            back=LPBack('back', lp_cube),
            front=LPFront('front', lp_cube),
        )
        lp_cube.face = face
        return lp_cube

    def x_borders(self):
        """Local x-coordinates of sub-cube borders"""
        return get_border_lctns(0., self.width, self.width_div)

    def y_borders(self):
        """Local y-coordinates of sub-cube borders"""
        return get_border_lctns(0., self.hgt, self.hgt_div)

    def z_borders(self):
        """Local z-coordinates of sub-cube borders"""
        return get_border_lctns(0., self.depth, self.depth_div)

    def x_node_lctns(self):
        return get_node_lctns(0., self.width, self.width_div)

    def y_node_lctns(self):
        return get_node_lctns(0., self.hgt, self.hgt_div)

    def z_node_lctns(self):
        return get_node_lctns(0., self.depth, self.depth_div)

    def center(self):
        """Local center for plot purposes"""
        return array([self.width / 2, self.hgt / 2, self.depth / 2])

    def vol(self):
        return self.width * self.hgt * self.depth

    def node_count(self):
        return self.width_div * self.hgt_div * self.depth_div

    # def node_idxs(self):
    #     return arange(self.node_count(), dtype=int64)

    def node_idxs_3d(self):
        # return self.node_idxs().reshape((self.depth_div, self.hgt_div, self.width_div))
        return arange(self.node_count(), dtype=int64).reshape((self.depth_div, self.hgt_div, self.width_div))

    def node_vols(self):
        return full(self.node_count(), self.vol() / self.node_count())

    def node_lctns(self):
        lctns = zeros((self.node_count(), 3))
        lctns_3d = lctns.reshape((self.depth_div, self.hgt_div, self.width_div, 3))
        for yi in range(self.hgt_div):
            for zi in range(self.depth_div):
                lctns_3d[zi, yi, :, 0] = self.x_node_lctns()[:]
                lctns_3d[zi, yi, :, 1] = self.y_node_lctns()[yi]
                lctns_3d[zi, yi, :, 2] = self.z_node_lctns()[zi]
        return lctns

    def node_geras(self):
        xdivs, ydivs, zdivs = self.width_div, self.hgt_div, self.depth_div
        x_node, y_node, z_node = self.x_node_lctns(), self.y_node_lctns(), self.z_node_lctns()
        x_border, y_border, z_border = self.x_borders(), self.y_borders(), self.z_borders()
        gera6d = DOK((zdivs, ydivs, xdivs, zdivs, ydivs, xdivs))  # Make it accessible over axes-indices
        data6d = gera6d.data  # Speed up DOK.__setitem__, because there are many costly checks
        # Geometric ratios for conductances in x-direction
        for zi in range(zdivs):
            for yi in range(ydivs):
                for xi in range(xdivs - 1):
                    data6d[(zi, yi, xi, zi, yi, xi + 1)] = (
                        (y_border[yi + 1] - y_border[yi]) *
                        (z_border[zi + 1] - z_border[zi]) /
                        (x_node[xi + 1] - x_node[xi]))
        # Geometric ratios for conductances in y-direction
        for zi in range(zdivs):
            for yi in range(ydivs - 1):
                for xi in range(xdivs):
                    data6d[(zi, yi, xi, zi, yi + 1, xi)] = (
                        (x_border[xi + 1] - x_border[xi]) *
                        (z_border[zi + 1] - z_border[zi]) /
                        (y_node[yi + 1] - y_node[yi]))
        # Geometric ratios for conductances in z-direction
        for zi in range(zdivs - 1):
            for yi in range(ydivs):
                for xi in range(xdivs):
                    data6d[(zi, yi, xi, zi + 1, yi, xi)] = (
                            (x_border[xi + 1] - x_border[xi])*
                            (y_border[yi + 1] - y_border[yi])/
                            (z_node[zi + 1] - z_node[zi]))
        gera2d = gera6d.to_coo().reshape((ydivs*xdivs*zdivs, ydivs*xdivs*zdivs))  # Reshape is only supported by COO
        gera2d = DOK.from_coo(gera2d)
        data = gera2d.data
        # Mirror along diagonal
        for k, v in list(data.items()):
            data[(k[1], k[0])] = v
        return gera2d

    def bounding_box(self):
        """The bounding box of the points of the mesh.
        A 2dimensional array where the first row contains the minimum values in each dimension
        and the second row contains the maximum values."""
        p, w, h, d = self.posn, self.width, self.hgt, self.depth
        return array([p,
                     [p[0]+w, p[1]+h, p[2]+d]])         

    def __hash__(self):
        return id(self)


def get_border_lctns(lctn0, lctn1, num_div):
    """Local coordinates of sub-cube borders"""
    return linspace(lctn0, lctn1, num_div + 1)


def get_node_lctns(lctn0, lctn1, num_div):
    """Local coordinates of sub-cube nodes"""
    bl = get_border_lctns(lctn0, lctn1, num_div)
    dl2 = (bl[1] - bl[0]) / 2
    return linspace(bl[0] + dl2, bl[num_div] - dl2, num_div)