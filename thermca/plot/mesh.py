from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import pyvista as pv

from thermca.plot import primitives as pp
from thermca.plot.primitives import AXES, DEFAULT, NAME, color_themes

if TYPE_CHECKING:
    from thermca.mesh import Mesh

POINT_IDX = 'point idx'
BODY_POINT_IDX = 'body vtx idx'


def mesh(
        mesh: Mesh,
        wireframe_sfcs: bool = True,
        hide: tuple[str] = (POINT_IDX, ),
        dpi: float = 95,
        color_theme: str = DEFAULT,
        plotter: Optional[pv.Plotter] = None
):
    """Plot mesh

    Args:
        mesh: Mesh containing body tetrahedron and surface triangles
        hide: Tuple containing objects to hide: VTX_IDX for vertex
            indices, BODY_VTX_IDX for inner body vertex indices,
            NAME for cell block names, AXES for coordinate axes
        wireframe_sfcs: If true, surface blocks as wireframe, closed
            surfaces otherwise
        dpi: Screen dpi
        color_theme: DEFAULT with grey background,
            BRIGHT with white background
        plotter: PyVista plotter
    """
    from thermca.mesh import block_center_point

    clr_theme = color_themes[color_theme]
    if plotter is None:
        plotter = pv.Plotter()
    plotter.set_background(clr_theme['bg'])
    text_bg_color = clr_theme['text_bg']
    fg_color = clr_theme['fg']
    # scene = figure.scene
    font_size, small_line, thick_line = pp._adjust_drawings_to_screen_dpi(dpi)
    pp._set_initial_view(plotter, draw2d=False)
    tetra_idx = 0
    triangle_idx = 0
    for cell_type, cells, name in zip(mesh.block_types, mesh.cell_blocks, mesh.block_names):
        if cell_type == 'tetra':
            pp.tetra_mesh(
                mesh.points,
                cells,
                sfc_color=None,
                edge_color=clr_theme['body_edges'],
                line_width=small_line,
                opacity=1.,
                plotter=plotter,
            )
            if NAME not in hide:
                lctn = block_center_point(mesh.points, cells)
                pp.labels([lctn], [name], clr_theme['body_text'], shape_color=text_bg_color, font_size=font_size, plotter=plotter)
            tetra_idx += 1
        elif cell_type == 'triangle':
            if color_theme == 'default':
                color = pp.tab10_colors[triangle_idx % len(pp.tab10_colors)]
            elif color_theme == 'bright':
                color = pp.dark_tab10_colors[triangle_idx % len(pp.dark_tab10_colors)]
            sfc_color = None if wireframe_sfcs else color
            pp.triangle_mesh(
                mesh.points,
                cells,
                sfc_color=sfc_color,
                edge_color=color,
                line_width=small_line,
                opacity=1.,
                plotter=plotter,
            )
            if NAME not in hide:
                lctn = block_center_point(mesh.points, cells)
                pp.labels([lctn], [name], color, shape_color=text_bg_color, font_size=font_size, plotter=plotter)
            triangle_idx += 1
    if POINT_IDX not in hide:
        if BODY_POINT_IDX not in hide:
            point_idxs = {point for cells in mesh.cell_blocks
                          for point in cells.ravel()}  # each point only once
        else:
            sfc_cell_blocks = mesh.extract_type('triangle').cell_blocks
            point_idxs = {point for cells in sfc_cell_blocks
                          for point in cells.ravel()}  # each point only once
        pp.labels(mesh.points[list(point_idxs)], point_idxs, fg_color, shape_color=text_bg_color, font_size=font_size, plotter=plotter)
    if AXES not in hide:
        pp.axes(color=clr_theme['fg'], font_size=font_size, use_2d=False, plotter=plotter)
    pp._set_initial_view(plotter, draw2d=False)
    return plotter