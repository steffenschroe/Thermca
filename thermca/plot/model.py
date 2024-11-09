"""3D plotting functions to visualise the model elements and the node net."""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from numpy import empty, array
import pyvista as pv

from thermca.plot import primitives as pp
from thermca.plot.primitives import AXES, DEFAULT, NAME, color_themes


if TYPE_CHECKING:
    from thermca.model import Model

# Plot entities
# Element specific
LINK = 'link'
HEAT_SRC_ELEM = 'heat src elem'
# Parameter specific
CAPY = 'capy'
HEAT = 'heat'
COND = 'cond'
CONN = 'conn'
TEMP = 'temp'
LUMP_NODE_IDX = 'lump node idx'
COLOR_BAR = 'color bar'
HEAT_SRC = 'heat src'
VOL = 'vol'
AREA = 'area'

# time scales
SIM_SPAN = 'sim span'
POINT_IN_TIME = 'point in time'


def elements(
        model: Model,
        hide: tuple[str, ...] = (AXES, ),
        dpi: float = 95,
        draw_2d: bool = False,
        color_theme: str = DEFAULT,
        colored_sfcs: bool = True,
        plotter: Optional[pv.Plotter] = None):
    """Plots elements of the model

    Args:
        model: The model containing the elements
        hide: Drawing objects to hide, the following are valid
            NAME, AXES, LINK, HEAT_SRC_ELEM
        dpi: Screen resolution in dots per inch
        draw_2d: View of x-y-plane with parallel projection
        color_theme: DEFAULT with grey background,
            BRIGHT with white background
        colored_sfcs: Draws Part surfaces in individual colors.
        plotter: PyVista Plotter to draw in, if None a new Plotter will
            be created
    """
    from thermca.pointnodes import Node, MatlNode, StatNode, BoundNode
    from thermca.links import CondLink, FilmLink, FlowLink
    from thermca.fem.fe_part import FEPart
    from thermca.lpm.lp_part import LPPart
    from thermca.source import HeatSource, FluxSource
    from thermca.mesh import block_center_point
    from thermca.plot.asm import bodies_and_surfs
    from thermca.lpm.lp_construction import assemble

    # # Drawing stuff
    if plotter is None:
        plotter = pv.Plotter()
    clr_theme = color_themes[color_theme]
    plotter.set_background(clr_theme['bg'])
    font_size, small_line, thick_line = pp._adjust_drawings_to_screen_dpi(dpi)

    # # Links
    if LINK not in hide:
        links = model._get_all_elems_of_types([CondLink, FilmLink, FlowLink])
        points, conns = [], []
        idx = 0
        for link in links:
            elem0, elem1 = link.elem0._net_dat, link.elem1._net_dat
            points.append(elem0.posn + elem0.dock_lctn)
            points.append(elem1.posn + elem1.dock_lctn)
            conns.append([idx, idx + 1])
            idx += 2
        points = array(points)
        conns = array(conns)

        pp.points(points, clr_theme['link'], thick_line * 3, plotter=plotter)
        pp.lines(points, conns, thick_line * 2, clr_theme['link'], opacity=1, plotter=plotter)

    # # Sources
    source_posns = []
    if HEAT_SRC_ELEM not in hide:
        for source in model._get_all_elems_of_type(HeatSource):
            elem = source.net_elem._net_dat
            source_posns.append(elem.posn + elem.dock_lctn)
        for source in model._get_all_elems_of_type(FluxSource):
            elem = source.surf._net_dat
            source_posns.append(elem.posn + elem.dock_lctn)
        pp.points(source_posns, pp.tab10_colors[3], thick_line * 6, plotter=plotter)

    # # Nodes
    ord_nodes = model._get_all_elems_of_types((Node, MatlNode))
    stat_nodes = model._get_all_elems_of_type(StatNode)
    bound_nodes = model._get_all_elems_of_type(BoundNode)

    node_posns = empty((len(stat_nodes), 3))
    for i, node in enumerate(stat_nodes):
        node_posns[i] = node._net_dat.posn
    pp.points(node_posns, clr_theme['text_bg'], thick_line * 4, plotter=plotter)
    pp.points(node_posns, clr_theme['fg'], thick_line * 2, plotter=plotter)

    node_posns = empty((len(bound_nodes), 3))
    for i, node in enumerate(bound_nodes):
        node_posns[i] = node._net_dat.posn
    pp.points(node_posns, clr_theme['fg'], thick_line * 4, plotter=plotter)
    pp.points(node_posns, clr_theme['text_bg'], thick_line * 2, plotter=plotter)

    node_posns = empty((len(ord_nodes), 3))
    for i, node in enumerate(ord_nodes):
        node_posns[i] = node._net_dat.posn
    pp.points(node_posns, clr_theme['fg'], thick_line * 4, plotter=plotter)

    # # LPParts
    lp_parts = model._get_all_elems_of_types(LPPart)
    for part in lp_parts:
        hide_incl_axes = hide if AXES in hide else (*hide, AXES)
        bodies_and_surfs(part.asm, hide_incl_axes, dpi, color_theme, plotter, colored_sfcs, array(part.posn))
    # # FEParts
    fe_parts = model._get_all_elems_of_types(FEPart)
    triangle_idx = 0
    for part in fe_parts:
        # plot only surfaces
        mesh = part._surf_meshes
        points = mesh.points + part._net_dat.posn
        for name, cell_type, cells, surf in zip(
                mesh.block_names, mesh.block_types, mesh.cell_blocks, part.surf):
            if colored_sfcs:
                sfc_color = pp.tab10_colors[triangle_idx % len(pp.tab10_colors)]
                edge_color = sfc_color
                text_color = sfc_color
            else:
                sfc_color = clr_theme['isfc']
                edge_color = None
                text_color = clr_theme['body_text']
            pp.triangle_mesh(
                points,
                cells,
                sfc_color=sfc_color,
                edge_color=edge_color,
                line_width=small_line,
                opacity=.2,
                plotter=plotter,
            )
            # TODO: check: triangles.actor.property.backface_culling = True
            if NAME not in hide:
                posn = surf._net_dat.posn + surf._net_dat.dock_lctn
                pp.labels([posn], [name], text_color, shape_color=clr_theme['text_bg'], font_size=font_size, plotter=plotter)
            triangle_idx += 1

    # # Names
    if NAME not in hide:
        for elem in ord_nodes + stat_nodes + bound_nodes + lp_parts + fe_parts:
            if isinstance(elem, (LPPart, FEPart)):
                posn = elem._net_dat.posn + elem._net_dat.dock_lctn
            else:
                posn = elem._net_dat.posn
            pp.labels([posn], [elem.name], clr_theme['fg'], clr_theme['text_bg'], font_size=font_size, plotter=plotter)

    if AXES not in hide:
        pp.axes(color=clr_theme['fg'], font_size=font_size, use_2d=draw_2d, plotter=plotter)
        # plotter.show_bounds(color=clr_theme['fg'], font_size=font_size, use_2d=draw2d)  # same as show_grid  # padding=.1,

    pp._set_initial_view(plotter, draw_2d)
    return plotter