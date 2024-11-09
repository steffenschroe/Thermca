"""3D plotting functions to visualise the model elements and the node net."""
from __future__ import annotations
from math import tau
from typing import TYPE_CHECKING, Optional

from numpy import (zeros, array, vstack, hstack, empty, sin, cos, arange, full)
import pyvista as pv

from thermca.plot import primitives as pp
from thermca.plot.primitives import NAME, AXES, EDGE


if TYPE_CHECKING:
    from thermca.lpm.asm import Asm

SURF = 'surf'


def asm_elem(
        asm: Asm,
        conts: Optional[list[tuple]] = None,
        hide: tuple[str, ...] = (pp.AXES, ),
        dpi: float = 95,
        draw2d: bool = False,
        color_theme: str = pp.DEFAULT,
        plotter: Optional[pv.Plotter] = None
):
    """Plots elements of the body assembly

    Args:
        asm: Body assembly
        conts: Body contacts
        hide: Drawing objects to hide, the following are valid
            NAME, AXES, SURF
        dpi: Screen resolution in dots per inch
        draw2d: View of x-y-plane with parallel projection
        color_theme: DEFAULT with grey background,
            BRIGHT with white background
        plotter: PyVista Plotter to draw in, if None a new Plotter will
            be created
    """
    from thermca.lpm.cube import Cube
    from thermca.lpm.cyl import Cyl
    from thermca.lpm.asm import Surf, ForceConts
    from thermca.lpm.lp_cube import LPCube
    from thermca.lpm.lp_cyl import LPCyl, LPBase, LPEnd, LPInner, LPOuter
    from thermca.lpm.lp_construction import create_lp_elems

    # Initialize drawing stuff
    if plotter is None:
        plotter = pv.Plotter()
    clr_theme = pp.color_themes[color_theme]
    plotter.set_background(clr_theme['bg'])
    font_size, small_line, thick_line = pp._adjust_drawings_to_screen_dpi(dpi)

    bodies_and_surfs(asm, hide, dpi, color_theme, plotter)

    # Create lumped parameter objects from geometry objects
    lp_blocks, lp_cyls, lp_link_surfs, force_cont_pairs = create_lp_elems(asm)

    # # Contacts
    link_points = []
    link_conns = []
    idx = 0
    if conts:
        dock_rad = array(
            [cyl.vol()**.333 for cyl in lp_cyls]
            + [block.vol()**.333 for block in lp_blocks]
        ).mean() / 20.
        for cont in conts:
            dock0_posn = cont[0].body.posn + cont[0].dock_lctn()
            pp.dock(
                dock0_posn,
                0.,
                dock_rad,
                cont[0].dock_dirn(),
                clr_theme['link'],
                clr_theme['edge'],
                opacity=1.,
                plotter=plotter
            )
            dock1_posn = cont[1].body.posn + cont[1].dock_lctn()
            pp.dock(
                dock1_posn,
                0.,
                dock_rad,
                cont[1].dock_dirn(),
                clr_theme['link'],
                clr_theme['edge'],
                opacity=1.,
                plotter=plotter
            )
            link_points.append(dock0_posn)
            link_points.append(dock1_posn)
            link_conns.append([idx, idx + 1])
            idx += 2

        if link_points:
            link_points = array(link_points)
            link_conns = array(link_conns)
            pp.points(
                link_points,
                clr_theme['link'],  # ord_node_rad
                thick_line * 3,
                plotter=plotter,
            )
            pp.lines(link_points, link_conns, thick_line * 2, clr_theme['link'], opacity=1, plotter=plotter)

    if AXES not in hide:
        pp.axes(color=clr_theme['fg'], font_size=font_size, use_2d=draw2d, plotter=plotter)

    pp._set_initial_view(plotter, draw2d)
    return plotter


def bodies_and_surfs(
        asm: Asm,
        hide,
        dpi,
        color_theme,
        plotter,
        colored_sfcs: bool = True,
        origin=array([0., 0., 0.]),
):
    from thermca.lpm.asm import Surf, ForceConts
    from thermca.lpm.lp_cube import LPCube
    from thermca.lpm.lp_cyl import LPCyl, LPBase, LPEnd, LPInner, LPOuter
    from thermca.lpm.lp_construction import create_lp_elems

    # Initialize drawing stuff
    clr_theme = pp.color_themes[color_theme]
    font_size, small_line, thick_line = pp._adjust_drawings_to_screen_dpi(dpi)

    # Create lumped parameter objects from geometry objects
    lp_blocks, lp_cyls, lp_link_surfs, force_cont_pairs = create_lp_elems(asm)

    # # Blocks
    if lp_blocks:
        edge_count = 8
        block_points = empty((len(lp_blocks), edge_count, 3))
        block_conn = array(
            [[0, 1], [1, 3], [3, 2], [2, 0],
             [0, 4], [1, 5], [2, 6], [3, 7],
             [4, 5], [5, 7], [7, 6], [6, 4]]
        )
        block_conns = empty((len(lp_blocks), len(block_conn), 2))
        rect_conn = array([[0, 1], [1, 3], [3, 2], [2, 0]])

        block_sub_div_points = []
        block_sub_div_conns = []
        width_norm = array([1, 0, 0])
        hgt_norm = array([0, 1, 0])
        depth_norm = array([0, 0, 1])
        sub_div_rect_idx = 0
        for i, block in enumerate(lp_blocks):
            # Edges
            width, hgt, depth = block.width, block.hgt, block.depth
            lctns = array([[0., 0., 0., 0., width, width, width, width],
                           [0., 0., hgt, hgt, 0., 0., hgt, hgt],
                           [0., depth, 0., depth, 0., depth, 0., depth]]).T
            block_points[i] = block.posn + lctns
            block_conns[i] = block_conn + i * edge_count
            # Subdivisions
            for width_sub_divs in block.x_borders()[1:-1]:
                left_rect_lctns = array([[0., 0., 0., 0.],
                                         [0., 0., hgt, hgt],
                                         [0., depth, 0., depth]]).T
                left_rect = block.posn + left_rect_lctns
                block_sub_div_points.append(left_rect + width_norm * width_sub_divs)
                block_sub_div_conns.append(rect_conn + sub_div_rect_idx * len(left_rect_lctns))
                sub_div_rect_idx += 1
            for hgt_sub_divs in block.y_borders()[1:-1]:
                bott_rect_lctns = array([[0., 0., width, width],
                                         [0., 0., 0., 0.],
                                         [0., depth, 0., depth]]).T
                bott_rect = block.posn + bott_rect_lctns
                block_sub_div_points.append(bott_rect + hgt_norm * hgt_sub_divs)
                block_sub_div_conns.append(rect_conn + sub_div_rect_idx * len(bott_rect_lctns))
                sub_div_rect_idx += 1
            for depth_sub_divs in block.z_borders()[1:-1]:
                back_rect_lctns = array([[0., 0., width, width],
                                         [0., hgt, 0., hgt],
                                         [0., 0., 0., 0.]]).T
                back_rect = block.posn + back_rect_lctns
                block_sub_div_points.append(back_rect + depth_norm * depth_sub_divs)
                block_sub_div_conns.append(rect_conn + sub_div_rect_idx * len(rect_conn))
                sub_div_rect_idx += 1

        # Draw block edges
        if EDGE not in hide:
            block_points = block_points.reshape(-1, 3)
            block_conns = block_conns.reshape(-1, 2)
            pp.lines(
                block_points + origin,
                block_conns,
                thick_line,
                clr_theme['edge'],
                opacity=1,
                plotter=plotter
            )
            # Draw block subdivisions
            if block_sub_div_points:
                block_sub_div_points = vstack(block_sub_div_points)
                block_sub_div_conns = vstack(block_sub_div_conns)
                pp.lines(
                    block_sub_div_points + origin,
                    block_sub_div_conns,
                    thick_line,
                    clr_theme['subdiv'],
                    opacity=1,
                    plotter=plotter,
                )

    # # Cylinders
    if lp_cyls:
        resolution = 20  # circle subdivisions, should be a multiple of 4
        # 4 -> circles (two for outer radius and two for inner radius), 3 -> x, y, z
        cyl_points = zeros((len(lp_cyls), 4, resolution, 3))
        cyl_symm_points = empty((len(lp_cyls) * 4, 3))  # symmetry line, 4 points two lines
        cyl_symm_conns = empty((len(lp_cyls) * 2, 2))
        cyl_symm_con = array([0, 1])

        angle = tau / resolution
        circ_idxs = arange(resolution)
        # normalized vectors in radial and length direction
        rad_norm = array([
            full(resolution, 0),
            cos(angle * circ_idxs),
            - sin(angle * circ_idxs),
        ]).T
        lgth_norm = array([1, 0, 0])

        # connection arrays for cylinder surface
        from_idx = arange(0, resolution - 1)  # first circle
        to_idx = arange(1, resolution)  # first circle
        circ_from_idx = hstack((from_idx, [resolution - 1]))  # close circle
        circ_to_idx = hstack((to_idx, [0]))  # close circle
        circ_conns = array([circ_from_idx, circ_to_idx]).T
        from_idx = hstack((circ_from_idx, circ_from_idx + resolution,
                           [0, resolution / 2]))  # second circle and connection lines between circles
        to_idx = hstack((circ_to_idx, circ_to_idx + resolution, [resolution,
                                                                 resolution + resolution / 2]))  # second circle and connection lines between circles
        cyl_conns = array([from_idx, to_idx]).T

        from_idxs = []
        to_idxs = []
        circ_sub_div_conns = []
        sub_div_circ_idx = 0
        cyl_sub_div_points = []

        for i, cyl in enumerate(lp_cyls):
            # cylinder edges
            left_outer_circ = cyl.posn + rad_norm * cyl.outer_rad
            cyl_points[i, 0] = left_outer_circ
            cyl_points[i, 1] = left_outer_circ + lgth_norm * cyl.lgth
            from_idxs.append(from_idx + i * resolution * 4)
            to_idxs.append(to_idx + i * resolution * 4)
            cyl_symm_points[i * 4] = cyl.posn - lgth_norm * cyl.lgth * .1
            cyl_symm_points[i * 4 + 1] = cyl.posn + lgth_norm * (cyl.lgth / 2 - cyl.lgth * .1)
            cyl_symm_points[i * 4 + 2] = cyl.posn + lgth_norm * (cyl.lgth / 2 + cyl.lgth * .1)
            cyl_symm_points[i * 4 + 3] = cyl.posn + lgth_norm * cyl.lgth * 1.1
            cyl_symm_conns[i * 2] = cyl_symm_con + i * 4
            cyl_symm_conns[i * 2 + 1] = cyl_symm_con + 2 + i * 4
            # cylinder subdivisions in length and radial direction
            for lgth_sub_divs in cyl.axial_borders()[1:-1]:
                cyl_sub_div_points.append(left_outer_circ + lgth_norm * lgth_sub_divs)
                circ_sub_div_conns.append(circ_conns + sub_div_circ_idx * resolution)
                sub_div_circ_idx += 1
            for rad_sub_divs in cyl.rad_borders()[1:-1]:
                left_circ = cyl.posn + rad_norm * rad_sub_divs
                cyl_sub_div_points.append(left_circ)
                cyl_sub_div_points.append(left_circ + lgth_norm * cyl.lgth)
                circ_sub_div_conns.append(cyl_conns + sub_div_circ_idx * resolution)
                sub_div_circ_idx += 2
            if cyl.inner_rad > 0.:
                # cylinder edges
                left_inner_circ = cyl.posn + rad_norm * cyl.inner_rad
                cyl_points[i, 2] = left_inner_circ
                cyl_points[i, 3] = left_inner_circ + lgth_norm * cyl.lgth
                from_idxs.append(from_idx + i * resolution * 4 + 2 * resolution)
                to_idxs.append(to_idx + i * resolution * 4 + 2 * resolution)
                # inner cylinder subdivisions in length direction
                for lgth_sub_divs in cyl.axial_borders()[1:-1]:
                    cyl_sub_div_points.append(left_inner_circ + lgth_norm * lgth_sub_divs)
                    circ_sub_div_conns.append(circ_conns + sub_div_circ_idx * resolution)
                    sub_div_circ_idx += 1
        # cylinder edges
        cyl_points = cyl_points.reshape(-1, 3)
        from_idxs = hstack(from_idxs)
        to_idxs = hstack(to_idxs)
        cyl_conns = array([from_idxs, to_idxs]).T

        # Draw cylinders
        if EDGE not in hide:
            pp.lines(
                cyl_points + origin,
                cyl_conns,
                thick_line,
                clr_theme['edge'],
                opacity=1,
                plotter=plotter
            )
            pp.lines(
                cyl_symm_points + origin,
                cyl_symm_conns,
                thick_line,
                clr_theme['symm'],
                opacity=1,
                plotter=plotter
            )
            # Cylinder subdivisions
            if cyl_sub_div_points:
                cyl_sub_div_conns = vstack(circ_sub_div_conns)
                cyl_sub_div_points = vstack(cyl_sub_div_points)
                pp.lines(
                    cyl_sub_div_points + origin,
                    cyl_sub_div_conns,
                    thick_line,
                    clr_theme['subdiv'],
                    opacity=1,
                    plotter=plotter
                )

    # # LinkSurfs
    if SURF not in hide and lp_link_surfs:
        for link_surf_idx, link_surf in enumerate(lp_link_surfs):
            if colored_sfcs:
                if color_theme == 'default':
                    color = pp.tab10_colors[link_surf_idx % len(pp.tab10_colors)]
                    text_color = color
                elif color_theme == 'bright':
                    color = pp.dark_tab10_colors[link_surf_idx % len(pp.dark_tab10_colors)]
                    text_color = color
            else:
                color = clr_theme['isfc']
                text_color = clr_theme['body_text']
            for face in link_surf.faces:
                body = face.body
                if isinstance(body, LPCyl):
                    if isinstance(face, (LPBase, LPEnd)):
                        pp.cyl_cap(
                            body.posn + face.center() + origin,
                            body.inner_rad,
                            body.outer_rad,
                            body.dirn,
                            color,
                            opacity=.2,
                            plotter=plotter
                        )
                    elif isinstance(face, (LPInner, LPOuter)):
                        pp.cyl(
                            body.posn + body.dirn * body.lgth / 2. + origin,
                            body.dirn,
                            face.rad(),
                            body.lgth,
                            color,
                            opacity=.2,
                            plotter=plotter,
                        )
                elif isinstance(body, LPCube):
                    pp.block_face(body.posn + face.verts() + origin, color, opacity=.2, plotter=plotter)

            if NAME not in hide:
                pp.labels(
                    [link_surf.dock_lctn() + origin],
                    [link_surf.name],
                    text_color,
                    clr_theme['text_bg'],
                    font_size=font_size,
                    plotter=plotter
                )

    if NAME not in hide:
        for body in lp_blocks + lp_cyls:
            if body.name != '':
                pp.labels(
                    [body.center() + body.posn + origin],
                    [body.name],
                    clr_theme['fg'],
                    clr_theme['text_bg'],
                    font_size=font_size,
                    plotter=plotter
                )


if __name__ == '__main__':
    from thermca.lpm.cube import Cube
    from thermca.lpm.cyl import Cyl
    from thermca.lpm.asm import Asm, Surf

    with Asm() as test_asm:
        b0 = Cube(
            name='block0',
            width=1.,
            hgt=1.5,
            depth=2,
            width_div=2,
            hgt_div=3,
            depth_div=4,
        )
        b1 = Cube(
            name='block1',
            posn=(1., 0., 0.),
            width=1.,
            hgt=1.5,
            depth=2,
            width_div=2,
            hgt_div=3,
            depth_div=4,
        )
        c0 = Cyl(
            name='cyl0',
            posn=(2., 0., 0.),
            lgth=2.,
            lgth_div=2,
            inner_rad=1.,
            outer_rad=2.,
            rad_div=2,
        )
        c1 = Cyl(
            name='cyl1',
            posn=(4., 0., 0.),
            lgth=2.,
            lgth_div=2,
            inner_rad=1.,
            outer_rad=2.,
            rad_div=2,
        )
        Surf(
            name='link0',
            faces=[b0.face.left, b0.face.front, b1.face.back]
        )
        Surf(
            name='link1',
            faces=[c0.face.outer, c1.face.inner]
        )

        Surf(
            name='link2',
            faces=[c0.face.base, c1.face.end]
        )

    test_asm.plot(dpi=250).show()


