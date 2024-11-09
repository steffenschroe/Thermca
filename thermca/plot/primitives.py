"""Basic plotting primitives, utility functions and data"""

from math import tau, acos
from typing import Optional, Union, Iterable

import pyvista
from numpy import (
    column_stack,
    zeros_like,
    ndarray,
    full,
    empty,
    uint8,
    array,
    arange,
    cos,
    sin,
    vstack,
    tile,
    repeat,
)
from scipy import interpolate
import pyvista as pv
from pyvista.core.filters import _get_output, _update_alg
import vtk

from thermca import vector3d as v3d

# color themes
DEFAULT = 'default'
BRIGHT = 'bright'

AXES = 'axes'

# Element and Parameter specific
NAME = 'name'

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
EDGE = 'edge'

tab10_colors = [
    (.121, .4648, .7031),
    (.996, .496, .0546),
    (.1718, .625, .1718),
    (.8359, .1523, .1562),
    (.5781, .4023, .7382),
    (.5468, .3359, .2929),
    (.8867, .4648, .7578),
    (.496, .496, .496),
    (.7343, .7382, .1328),
    (.0898, .7421, .8085),
]

dark_tab10_colors = array(tab10_colors) * .7


color_themes = {
    DEFAULT: {
        'bg': (.5, .5, .5),
        'fg': (1., 1., 1.),
        'text_bg': (0., 0., 0.),
        # params
        'temp_clamp': (.3, .3, .3),
        'cond_line': (.8, .8, .8),
        'cond_cyl': (.3, .3, .3),
        'heat_cone': (1., 1., 1.),
        'heat_cone_stripe': (.3, .3, .3),
        'forced_heat': (.9, .9, .9),
        # elements
        'edge': (1., 1., 1.),
        'subdiv': (.6, .6, .6),
        'symm': (1., 1., 1.),
        'link': (.2, .2, .2),
        # mesh
        'body_edges': (.6, .6, .6),
        'body_text': (.8, .8, .8),
        'isfc': (.3, .3, .3),
    },
    BRIGHT: {
        'bg': (1., 1., 1.),
        'fg': (0., 0., 0.),
        'text_bg': (.8, .8, .8),
        # params
        'temp_clamp': (.3, .3, .3),
        'cond_line': (.2, .2, .2),
        'cond_cyl': (.5, .5, .5),
        'heat_cone': (.8, .8, .8),
        'heat_cone_stripe': (.2, .2, .2),
        'forced_heat': (.1, .1, .1),
        # elements
        'edge': (0., 0., 0.),
        'subdiv': (.8, .8, .8),
        'symm': (0., 0., 0.),
        'link': (.2, .2, .2),
        # mesh
        'body_edges': (.8, .8, .8),
        'body_text': (.2, .2, .2),
        'isfc': (.3, .3, .3),
    },
}


def _adjust_drawings_to_screen_dpi(dpi):
    font_size = int(
        interpolate.interp1d([95, 230], [14, 24], fill_value='extrapolate')(dpi)
    )  # font size depending on screen dpi
    small_line = 1 if dpi < 140 else 2
    thick_line = 2 if dpi < 140 else 3
    return font_size, small_line, thick_line


def _set_initial_view(plotter: pv.Plotter, draw2d: bool):
    """Set initial view of the scene

    Intuitive front heavy view in 3D-mode
    View to x-y-plane with parallel projection in 2D-mode
    """
    # Thermca uses y-axis up because this allows good compatibility
    # between 2d and 3d view mode
    plotter.view_xy()
    if draw2d:
        plotter.enable_parallel_projection()
    else:
        # Intuitive front heavy 3d view
        plotter.camera.azimuth = (
            15  # Move center viewing camera around horizontal latitude
        )
        plotter.camera.elevation = 15  # Move center viewing camera around longitude


def axes(color, font_size, use_2d, plotter):
    # Use own implementation because of: https://github.com/pyvista/pyvista/issues/147
    # Scaling doesn't work at the moment because of odd Vtk behaviour.
    # It scales text of vtkCubeAxesActor2D, but it does not scale text
    # of vtkLabelPlacementMapper during change of the size of drawing
    # window.
    axes = vtk.vtkCubeAxesActor2D()
    axes.SetBounds(plotter.renderer.bounds)
    axes.SetCamera(plotter.camera)
    axes.SetFlyModeToClosestTriad()
    # axes.SetNumberOfLabels(2)
    # axes.SetFlyModeToOuterEdges()
    axes.SetCornerOffset(0.)
    # axes.UseRangesOn()
    # axes.UseBoundsOn()
    axes.ScalingOff()
    if use_2d:
        axes.ZAxisVisibilityOff()
    axes.SetFontFactor(1.)  #

    tprop = vtk.vtkTextProperty()
    tprop.SetColor(color)
    tprop.ShadowOff()
    tprop.SetFontSize(font_size)
    tprop.SetBold(False)

    axes.SetAxisTitleTextProperty(tprop)
    axes.SetAxisLabelTextProperty(tprop)
    plotter.add_actor(axes)


# # # Basic primitives # # #


def labels(
    posns: Union[list, ndarray],
    text: Iterable,
    text_color: tuple[float, float, float],
    shape_color: tuple[float, float, float],
    font_size: float,
    plotter: pyvista.Plotter,
):
    """Draw multiple text labels
    Args:
        posns: Positions, format [[x0, y0, z0], [x1, y1, z1], ...]
        text: Labels, if not given as strings, they get internally converted
        text_color: Text color (one for all labels)
        shape_color: Text background color (one for all labels)
        font_size: Font size in pixels
        plotter: Plotter to plot in
    """
    if len(posns) == 0:
        return
    plotter.add_point_labels(
        posns,
        text,
        text_color=text_color,
        font_size=font_size,
        point_size=0,
        shape_color=shape_color,
        margin=0,
        show_points=False,
        always_visible=True,
    )


def points(
    posns: Union[list, ndarray],
    color: tuple[float, float, float],
    size: float,
    plotter: pyvista.Plotter,
):
    """Draw multiple pixel points
    Args:
        posns: Midpoints, format [[x0, y0, z0], [x1, y1, z1], ...]
        color: Color (one for all spheres)
        size: Size of points in pixels
        plotter: Plotter to plot in
    """
    if len(posns) == 0:
        return
    point_mesh = pv.PolyData(posns)
    plotter.add_mesh(point_mesh, point_size=size, color=color)


def colored_points(
    posns: ndarray,
    color_intsts: ndarray,
    size: float,
    clim: tuple[float, float],
    show_color_bar: bool,
    plotter: pyvista.Plotter,
):
    """Draw multiple pixel points with individual color
    Args:
        posns: Midpoints, format [[x0, y0, z0], [x1, y1, z1], ...]
        color_intsts: Intensity for colors
        size: Size of points in pixels
        clim: Color bar limits
        show_color_bar: Turn color bar on/off
        plotter: Plotter to plot in
    """
    if len(posns) == 0:
        return
    point_mesh = pv.PolyData(posns)
    point_mesh.point_data['colors'] = color_intsts
    plotter.add_mesh(
        point_mesh,
        point_size=size,
        cmap="plasma",
        clim=clim,
        show_scalar_bar=show_color_bar,
    )


def lines(
    points: ndarray,
    conns: ndarray,
    width: float,
    color: tuple[float, float, float],
    opacity: float,
    plotter: pyvista.Plotter,
):
    """Draw multiple pixel lines

    Args:
        points: Supporting points, format [[x0, y0, z0], [x1, y1, z1], ...]
        conns: Pairs of point indices that define line start and end.
            e.g. [[0, 1], [1, 2], ...]
        width: Width of the lines in pixels
        color: Color (one for all lines)
        opacity: Opacity
        plotter: Plotter to plot in
    """
    if len(conns) == 0:
        return
    line_mesh = pv.PolyData()
    line_mesh.points = points
    cells = full((len(conns), 3), 2, dtype=pv.ID_TYPE)
    cells[:, 1] = conns[:, 0]
    cells[:, 2] = conns[:, 1]
    line_mesh.lines = cells
    plotter.add_mesh(line_mesh, color=color, line_width=width, opacity=opacity)


def scaled_spheres(
    posns: ndarray,
    radii: ndarray,
    color: tuple[float, float, float],
    plotter: pyvista.Plotter,
):
    """Draw multiple spheres with individual size
    Args:
        posns: Midpoints, format [[x0, y0, z0], [x1, y1, z1], ...]
        radii: Radii of the spheres
        color: Color (one for all spheres)
        plotter: Plotter to plot in
    """
    if len(posns) == 0:
        return
    point_mesh = pv.PolyData(posns)
    # Scalars given as diameter seems to work but documentation suggests radius
    point_mesh.point_data['radii'] = radii * 2.

    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(20)
    sphere.SetPhiResolution(20)
    sphere.Update()
    geom_dataset = sphere.GetOutput()

    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(point_mesh)
    glyphs.SetSourceData(geom_dataset)

    _update_alg(glyphs, message='Computing scaled_spheres')
    pv_mesh = _get_output(glyphs)
    plotter.add_mesh(pv_mesh, color=color)


def glowing_scaled_spheres(
    posns: ndarray,
    radii: ndarray,
    color_intsts: ndarray,
    clim: tuple[float, float],
    show_color_bar: bool,
    plotter: pyvista.Plotter,
):
    """Draw multiple spheres with individual size and color
    Args:
        posns: Midpoints, format [[x0, y0, z0], [x1, y1, z1], ...]
        radii: Radii of the spheres
        color_intsts: Intensity for colors
        clim: Color bar limits
        show_color_bar: Turn color bar on/off
        plotter: Plotter to plot in
    """
    if len(posns) == 0:
        return
    point_mesh = pv.PolyData(posns)
    # sphere radius as norm of vector
    zero_lgths = zeros_like(radii)
    # Vectors given as diameter seems to work but documentation suggests radius
    point_mesh.point_data['radii'] = column_stack((radii * 2., zero_lgths, zero_lgths))
    point_mesh.set_active_vectors('radii')

    point_mesh.point_data['colors'] = color_intsts
    point_mesh.set_active_scalars('colors')

    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(20)
    sphere.SetPhiResolution(20)
    sphere.Update()
    geom_dataset = sphere.GetOutput()

    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(point_mesh)
    glyphs.SetSourceData(geom_dataset)
    # glyphs.SetScaleFactor(1.15)
    # glyphs.SetVectorModeToUseNormal()
    glyphs.SetColorModeToColorByScalar()
    glyphs.SetScaleModeToScaleByVector()
    _update_alg(glyphs, message='Computing colored_and_scaled_spheres')
    pv_mesh = _get_output(glyphs)

    plotter.add_mesh(
        pv_mesh,
        cmap="plasma",
        clim=clim,
        show_scalar_bar=show_color_bar,
    )


def color_bar(title: str, font_size: float, clim: tuple[float, float], plotter):
    point_mesh = pv.PolyData(array([[0., 0., 0.]]))
    point_mesh.point_data[title] = array([0.])
    scalar_bar_args = {
        'vertical': True,
        'width': .05,
        'height': .9,
        'position_x': .92,
        'position_y': .01,
        # Text
        'title_font_size': font_size,
        'label_font_size': font_size,
    }
    plotter.add_mesh(
        point_mesh,
        cmap="plasma",
        # point_size=0,
        opacity=0.,
        clim=clim,
        scalar_bar_args=scalar_bar_args,
        show_scalar_bar=True,
    )


def scaled_tubes(
    points: ndarray,
    conns: ndarray,
    radii: ndarray,
    color: tuple[float, float, float],
    opacity: float,
    capped: bool,
    plotter: pyvista.Plotter,
):
    """Draw multiple tube surfaces
    Args:
        points: Supporting points, format [[x0, y0, z0], [x1, y1, z1], ...]
        conns: Pairs of point indices that define tube start and end.
            e.g. [[0, 1], [1, 2], ...]
        radii: Radii on each of the supporting points
        color: Color
        opacity: Opacity
        capped: Turn on/off whether the tube ends are closed by polygons
        plotter: Figure to plot in
    """
    if len(conns) == 0:
        return
    line_mesh = pv.PolyData()
    line_mesh.points = points
    cells = full((len(conns), 3), 2, dtype=pv.ID_TYPE)
    cells[:, 1] = conns[:, 0]
    cells[:, 2] = conns[:, 1]
    line_mesh.lines = cells
    line_mesh.point_data['radii'] = radii

    tube = vtk.vtkTubeFilter()
    tube.SetInputDataObject(line_mesh)
    tube.SetNumberOfSides(20)
    field = line_mesh.get_array_association('radii', preference='point')
    tube.SetInputArrayToProcess(0, 0, 0, field.value, 'radii')
    # tube.SetVaryRadiusToVaryRadiusByScalar()
    tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tube.SetCapping(capped)  # transparency seems to hide the capped ends
    # tube.CappingOn()
    tube.Update()

    pv_mesh = _get_output(tube)
    plotter.add_mesh(pv_mesh, color=color, opacity=opacity)


def colored_and_scaled_tubes(
    points: ndarray,
    conns: ndarray,
    radii: ndarray,
    colors: ndarray,
    opacity: float,
    capped: bool,
    plotter: pyvista.Plotter,
):
    """Draw tube surfaces with individual size and color
    Args:
        points: Supporting points, format [[x0, y0, z0], [x1, y1, z1], ...]
        conns: Pairs of point indices that define tube start and end.
            e.g. [[0, 1], [1, 2], ...]
        radii: Radii on each of the supporting points
        colors: Color of each tube as rgb triple
        opacity: Opacity
        capped: Turn on/off whether the tube ends are closed by polygons
        plotter: Figure to plot in
    """
    if len(conns) == 0:
        return
    line_mesh = pv.PolyData()
    line_mesh.points = points
    cells = full((len(conns), 3), 2, dtype=pv.ID_TYPE)
    cells[:, 1] = conns[:, 0]
    cells[:, 2] = conns[:, 1]
    line_mesh.lines = cells
    line_mesh.point_data['radii'] = radii
    line_mesh.cell_data['colors'] = colors
    # print(line_mesh)
    """
    # Set colors
    from vtk.util.numpy_support import numpy_to_vtk
    vtk_colors = numpy_to_vtk(colors.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)  # create vtkDoubleArray
    vtk_colors.SetName('colors')
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetNumberOfTuples(len(points))
    line_mesh.GetPointData().AddArray(vtk_colors)
    """

    tube = vtk.vtkTubeFilter()
    # Set radii
    tube.SetInputDataObject(line_mesh)
    tube.SetNumberOfSides(20)
    field = line_mesh.get_array_association('radii', preference='point')
    tube.SetInputArrayToProcess(0, 0, 0, field.value, 'radii')
    # tube.SetVaryRadiusToVaryRadiusByScalar()
    tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()

    """
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray('colors')
    """

    tube.SetCapping(capped)  # transparency seems to hide the capped ends
    # tube.CappingOn()
    tube.Update()

    pv_mesh = _get_output(tube)
    plotter.add_mesh(pv_mesh, opacity=opacity, scalars='colors', rgb=True)


def glowing_scaled_tubes(
    points: ndarray,
    conns: ndarray,
    radii: ndarray,
    glow: ndarray,
    clim: tuple[float, float],
    opacity: float,
    capped: bool,
    plotter: pyvista.Plotter,
):
    """Draw tube surfaces with individual size and glow
    Args:
        points: Supporting points, format [[x0, y0, z0], [x1, y1, z1], ...]
        conns: Pairs of point indices that define tube start and end.
            e.g. [[0, 1], [1, 2], ...]
        radii: Radii on each of the supporting points
        glow: Glow intensity on points
        opacity: Opacity
        capped: Turn on/off whether the tube ends are closed by polygons
        plotter: Figure to plot in
    """
    if len(conns) == 0:
        return
    line_mesh = pv.PolyData()
    line_mesh.points = points
    cells = full((len(conns), 3), 2, dtype=pv.ID_TYPE)
    cells[:, 1] = conns[:, 0]
    cells[:, 2] = conns[:, 1]
    line_mesh.lines = cells
    line_mesh.point_data['radii'] = radii
    line_mesh.point_data['colors'] = glow
    # print(line_mesh)
    """
    # Set colors
    from vtk.util.numpy_support import numpy_to_vtk
    vtk_colors = numpy_to_vtk(colors.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)  # create vtkDoubleArray
    vtk_colors.SetName('colors')
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetNumberOfTuples(len(points))
    line_mesh.GetPointData().AddArray(vtk_colors)
    """

    tube = vtk.vtkTubeFilter()
    # Set radii
    tube.SetInputDataObject(line_mesh)
    tube.SetNumberOfSides(20)
    field = line_mesh.get_array_association('radii', preference='point')
    tube.SetInputArrayToProcess(0, 0, 0, field.value, 'radii')
    # tube.SetVaryRadiusToVaryRadiusByScalar()
    tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()

    """
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray('colors')
    """

    tube.SetCapping(capped)  # transparency seems to hide the capped ends
    # tube.CappingOn()
    tube.Update()

    pv_mesh = _get_output(tube)
    # plotter.add_mesh(pv_mesh, opacity=opacity, scalars='colors', rgb=True)
    plotter.add_mesh(pv_mesh, clim=clim, opacity=opacity, scalars='colors', cmap="plasma", show_scalar_bar=False)


def triangle_mesh(
    points: ndarray,
    triangles: ndarray,
    sfc_color: Optional[tuple[float, float, float]],
    edge_color: Optional[tuple[float, float, float]],
    line_width: float,
    opacity: float,
    plotter: pyvista.Plotter,
):
    """Draws triangle surface mesh

    Args:
        points: Supporting points, format [[x0, y0, z0], [x1, y1, z1], ...]
        triangles: Triangle connections as indices to the points array.
            e.g. [[0, 1, 2], [3, 4, 5], ...]
        sfc_color: Surface color
        edge_color: Edge color
        line_width: Line width in pixels
        opacity: Opacity of all elements
        plotter: Figure to plot in
    """
    if len(triangles) == 0:
        return
    cells = empty((triangles.shape[0], 4), dtype=pv.ID_TYPE)
    cells[:, -3:] = triangles
    cells[:, 0] = 3
    pv_mesh = pv.PolyData(points, cells, deep=False)
    # style may be 'surface', 'wireframe' or 'points'
    if sfc_color is None:
        style = 'wireframe'
        color = edge_color
        plotter.add_mesh(
            pv_mesh,
            show_edges=True,
            color=color,
            edge_color=edge_color,
            line_width=line_width,
            style=style,
            opacity=opacity,
            ambient=1.,  # prevent color flicker of wireframe lines
            specular=0.,
            diffuse=0.,
        )
    else:
        style = 'surface'
        color = sfc_color
        show_edges = False if edge_color is None else True
        plotter.add_mesh(
            pv_mesh,
            show_edges=show_edges,
            color=color,
            edge_color=edge_color,
            line_width=line_width,
            style=style,
            opacity=opacity,
        )


def tetra_mesh(
    points: ndarray,
    tetras: ndarray,
    sfc_color: Optional[tuple[float, float, float]],
    edge_color: Optional[tuple[float, float, float]],
    line_width: float,
    opacity: float,
    plotter: pyvista.Plotter,
):
    """Draws tetrahedron mesh

    If sfc_color is None a wireframe with internal edges are drawn,
    otherwise only the outer surfaces

    Args:
        points: Supporting points, format [[x0, y0, z0], [x1, y1, z1], ...]
        tetras: Tetrahedron connections as indices to the points array
           e.g. [[0, 1, 2, 3], [4, 5, 6, 7], ...]
        sfc_color: Surface color
        edge_color: Edge color
        line_width: Line width in pixels
        opacity: Opacity of all elements
        plotter: Figure to plot in
    """
    if len(tetras) == 0:
        return
    cells = empty((tetras.shape[0], 5), dtype=pv.ID_TYPE)
    cells[:, -4:] = tetras
    cells[:, 0] = 4
    cell_type = full(tetras.shape[0], 10, dtype=uint8)
    pv_mesh = pv.UnstructuredGrid(cells, cell_type, points, deep=False)  # needs vtk > 9
    # style may be 'surface', 'wireframe' or 'points'
    # p.add_mesh(mesh.contour([160]).extract_all_edges(), color="grey", opacity=.25)
    if sfc_color is None:
        style = 'wireframe'
        color = edge_color
        # to show also internal edges we need an "edge mesh"
        pv_edge_mesh = pv_mesh.extract_all_edges()  # vtkExtractEdges
        plotter.add_mesh(
            pv_edge_mesh,
            show_edges=True,
            color=color,
            edge_color=edge_color,
            line_width=line_width,
            style=style,
            opacity=opacity,
        )
    else:
        style = 'surface'
        color = sfc_color
        plotter.add_mesh(
            pv_mesh,
            show_edges=True,
            color=color,
            edge_color=edge_color,
            line_width=line_width,
            style=style,
            opacity=opacity,
        )


def colored_tetra_mesh(
    points: ndarray,
    tetras: ndarray,
    colors: ndarray,
    plotter: pyvista.Plotter,
):
    """Draws triangle surface mesh

    If sfc_color is None a wireframe with internal edges are drawn,
    otherwise only the outer surfaces

    Args:
        points: Supporting points, format [[x0, y0, z0], [x1, y1, z1], ...]
        tetras: Tetrahedron connections as indices to the points array
           e.g. [[0, 1, 2, 3], [4, 5, 6, 7], ...]
        colors: Colors of points
        plotter: Figure to plot in
    """
    if len(tetras) == 0:
        return
    cells = empty((tetras.shape[0], 5), dtype=pv.ID_TYPE)
    cells[:, -4:] = tetras
    cells[:, 0] = 4
    cell_type = full(tetras.shape[0], 10, dtype=uint8)
    pv_mesh = pv.UnstructuredGrid(cells, cell_type, points, deep=False)  # needs vtk > 9
    pv_mesh['scalars'] = colors
    plotter.add_mesh(pv_mesh)


# # # Assembled primitives # # #


def tube_cones(
    posns: ndarray,
    radii: ndarray,
    lgths: ndarray,
    axes: ndarray,
    color,
    opacity,
    plotter,
):
    """Draw multiple cones with tube surfaces
    Args:
        posns: Position as midpoint of cone base,
            format [[x0, y0, z0], [x1, y1, z1], ...]
        radii: Radii of cone base
        lgths: Lengths from base to tip
        axes: Normalized directions from base to tip,
            format [[dx0, dy0, dz0], [dx1, dy1, dz1], ...]
        color: Color
        opacity: Opacity
        plotter: Figure to plot in
    """
    if len(posns) == 0:
        return
    cone_count = len(radii)
    point_count = cone_count * 2
    p = empty((point_count, 3))
    r = empty(point_count)
    p[0::2] = posns
    p[1::2] = posns + axes * lgths[:, None]
    very_small = 1e-16
    r[0::2] = radii
    r[1::2] = very_small
    connections = array([arange(0, point_count - 1, 2), arange(1, point_count, 2)]).T
    scaled_tubes(
        p, connections, r, color, opacity=opacity, capped=True, plotter=plotter
    )


def glowing_tube_cones(
    posns: ndarray,
    radii: ndarray,
    lgths: ndarray,
    axes: ndarray,
    glow: ndarray,
    clim: tuple[float, float],
    opacity,
    plotter,
):
    """Draw multiple cones with tube surfaces and glow
    Args:
        posns: Position as midpoint of cone base,
            format [[x0, y0, z0], [x1, y1, z1], ...]
        radii: Radii of cone base
        lgths: Lengths from base to tip
        axes: Normalized directions from base to tip,
            format [[dx0, dy0, dz0], [dx1, dy1, dz1], ...]
        glow: Glow intensities of cones
        opacity: Opacity
        plotter: Figure to plot in
    """
    if len(posns) == 0:
        return
    cone_count = len(radii)
    point_count = cone_count * 2
    p = empty((point_count, 3))
    r = empty(point_count)
    p[0::2] = posns
    p[1::2] = posns + axes * lgths[:, None]
    very_small = 1e-16
    r[0::2] = radii
    r[1::2] = very_small
    connections = array([arange(0, point_count - 1, 2), arange(1, point_count, 2)]).T
    glowing_scaled_tubes(
        p,
        connections,
        r,
        repeat(glow, 2),
        clim,
        opacity=opacity,
        capped=True,
        plotter=plotter,
    )


def cond_cylinders(
    posns: ndarray,
    radii: ndarray,
    lgths: ndarray,
    axis: ndarray,
    color: tuple[float, float, float],
    opacity,
    plotter,
):
    """Draw multiple conductance cylinders with tube surfaces
    Args:
        posns: Position as midpoint of cylinder side,
            format [[x0, y0, z0], [x1, y1, z1], ...]
        radii: Radii
        lgths: Lengths from cap to cap
        axis: Normalized directions of axes,
            format [[dx0, dy0, dz0], [dx1, dy1, dz1], ...]
        color: Color
        plotter: Figure to plot in
    """
    if len(posns) == 0:
        return
    cone_count = len(radii)
    point_count = cone_count * 2
    p = empty((point_count, 3))
    r = empty(point_count)
    p[0::2] = posns
    p[1::2] = posns + axis * lgths[:, None]
    r[0::2] = radii
    r[1::2] = radii
    connections = array([arange(0, point_count - 1, 2), arange(1, point_count, 2)]).T
    scaled_tubes(
        p, connections, r, color, opacity=opacity, capped=True, plotter=plotter
    )


def colored_cond_cylinders(
    posns: ndarray,
    radii: ndarray,
    lgths: ndarray,
    axis: ndarray,
    colors: ndarray,
    opacity,
    plotter,
):
    """Draw multiple colored conductance cylinders with tube surfaces
    Args:
        posns: Position as midpoint of cylinder side,
            format [[x0, y0, z0], [x1, y1, z1], ...]
        radii: Radii
        lgths: Lengths from cap to cap
        axis: Normalized directions of axes,
            format [[dx0, dy0, dz0], [dx1, dy1, dz1], ...]
        colors: Colors of individual cylinders
        plotter: Figure to plot in
    """
    if len(posns) == 0:
        return
    cone_count = len(radii)
    point_count = cone_count * 2
    p = empty((point_count, 3))
    r = empty(point_count)
    p[0::2] = posns
    p[1::2] = posns + axis * lgths[:, None]
    r[0::2] = radii
    r[1::2] = radii
    connections = array([arange(0, point_count - 1, 2), arange(1, point_count, 2)]).T
    colored_and_scaled_tubes(
        p, connections, r, colors, opacity=opacity, capped=True, plotter=plotter
    )


def forced_flow_arrows(
    posns: ndarray,
    radii: ndarray,
    lengs: ndarray,
    axes: ndarray,
    color: tuple[float, float, float],
    opacity: float,
    plotter: pyvista.Plotter,
):
    """Draw multiple cylinders with tube surfaces
    Args:
        posns: Position of conductance cylinder as midpoint of cylinder
            side, format [[x0, y0, z0], [x1, y1, z1], ...]
        radii: Conductance cylinder radii
        lengs: Cylinder lengths from cap to cap
        axes: Normalized directions of axes,
            format [[dx0, dy0, dz0], [dx1, dy1, dz1], ...]
        color: Color
        opacity: Opacity
        plotter: Figure to plot in
    """
    if len(posns) == 0:
        return
    # normalized vectors for points in radial and length direction
    leng_norm = array([1, 0, 0])
    base_angles = arange(4) * tau / 4  # 4 points for 2 bases with 2 points each
    base_rad_norm = array(
        [
            full(4, 0),
            cos(base_angles),
            -sin(base_angles),
        ]
    ).T
    # connection arrays for points: 4 base points followed by 1 tip point
    conn = array([[0, 2], [2, 4], [4, 0], [1, 3], [3, 4], [4, 1]])
    points = []
    conns = []
    idx = 0
    for ps, r, l, a in zip(posns, radii, lengs, axes):
        # create points in base coord system, cylinder axis in x-direction
        base_points = base_rad_norm * r + leng_norm * l
        tip_point = array([[0, 0, 0]])
        arr_points = vstack((base_points, tip_point))
        # rotate points towards axis
        # rotation axis = cross_product(unrotated vec, cylinder axis)
        rot_axis = v3d.cross(leng_norm, a)
        # inverse cosine to get the rotation amount or angle
        # a*b = |a|*|b|*cos(angle)
        # if |a| and |b| == 1 (normalized) than: angle = acos(a*b)
        rot_angle = acos(v3d.dot(a, leng_norm))
        rot_mat = v3d.rot_mat(rot_angle, rot_axis)
        # vectorized matrix-vector multiplication,
        # where M is rotation matrix and vv is array of 3d vectors
        # vec_dot = vectorize(dot, signature='(m,n),(n)->(m)')
        # vec_dot(M, vv)
        # or better: (M@vv.T).T
        arr_points = (rot_mat @ arr_points.T).T
        # translate points toward posns
        arr_points += ps
        # collect results
        points.append(arr_points)
        conns.append(conn + idx * len(arr_points))
        idx += 1
    if points:
        lines(
            vstack(points),
            vstack(conns),
            1,
            opacity=opacity,
            color=color,
            plotter=plotter,
        )


def tube_heat_cones(
    posns: ndarray,
    radii: ndarray,
    lengs: ndarray,
    axes: ndarray,
    cone_color,
    stripe_color,
    opacity,
    plotter,
):
    """Draw multiple striped cones with tube surfaces
    Args:
        posns: Position as midpoint of cone base,
            format [[x0, y0, z0], [x1, y1, z1], ...]
        radii: Radii of cone base
        lengs: Lengths from base to tip
        axes: Normalized directions from base to tip,
            format [[dx0, dy0, dz0], [dx1, dy1, dz1], ...]
        color: Color
    """
    if len(posns) == 0:
        return
    # TODO: use colored_and_scaled_tubes to include stripe instead of stripe overlay
    #  see following commented code:
    '''
    cone_count = len(radii)
    point_count = cone_count*4
    p = empty((point_count, 3))
    r = empty(point_count)
    p[0::4] = posns  # base point
    p[1::4] = posns + axes*(lengs*.4)[:, None]  # stripe begin
    p[2::4] = posns + axes*(lengs*.6)[:, None]  # stripe end
    p[3::4] = posns + axes*lengs[:, None]  # tip
    very_small = 1e-32
    r[0::4] = radii
    r[1::4] = radii*.6
    r[2::4] = radii*.4
    r[3::4] = very_small
    connections = empty((cone_count*3, 2))
    connections[0::3, 0] = arange(0, point_count - 3, 4)
    connections[1::3, 0] = arange(1, point_count - 2, 4)
    connections[2::3, 0] = arange(2, point_count - 1, 4)
    connections[0::3, 1] = arange(1, point_count - 2, 4)
    connections[1::3, 1] = arange(2, point_count - 1, 4)
    connections[2::3, 1] = arange(3, point_count, 4)
    _colored_tubes(p, r, color, connections, capping=1)
    '''

    tube_cones(posns, radii, lengs, axes, cone_color, opacity=opacity, plotter=plotter)
    # stripe
    cone_count = len(radii)
    point_count = cone_count * 2
    p = empty((point_count, 3))
    r = empty(point_count)
    p[0::2] = posns + axes * (lengs * .3)[:, None]
    p[1::2] = posns + axes * (lengs * .5)[:, None]
    r[0::2] = radii * .701
    r[1::2] = radii * .501
    connections = array([arange(0, point_count - 1, 2), arange(1, point_count, 2)]).T
    scaled_tubes(
        p, connections, r, stripe_color, opacity=opacity, capped=True, plotter=plotter
    )


def temp_clamping(
    capy_posns,
    capy_rads,
    color,
    plotter,
):
    """Draw 6 cones around the bound temperature capacity sphere"""
    if len(capy_posns) == 0:
        return
    cone_rad = capy_rads * .4
    cone_hgt = cone_rad * 1.732 * .5
    ds = array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    p = empty((len(capy_rads) * len(ds), 3))
    axis = empty((len(capy_rads) * len(ds), 3))
    rad = empty(len(capy_rads) * len(ds))
    hgt = empty(len(capy_rads) * len(ds))
    for i, xd in enumerate(ds):
        pp = xd * (capy_rads + cone_hgt)[:, None]
        p[i::6] = (capy_posns + pp)[:]
        axis[i::6] = tile(-xd, (len(capy_rads), 1))
        rad[i::6] = cone_rad
        hgt[i::6] = cone_hgt
    tube_cones(p, rad, hgt, axis, color, opacity=1., plotter=plotter)


def glowing_temp_clamping(
    capy_posns,
    cone_hgts,
    glows: ndarray,
    clim: tuple[float, float],
    plotter,
):
    """Draw 6 cones around the bound temperature capacity point
    Args:
        capy_posns: Positions of each clamping
        cone_hgts: Cone hight of each clamping
        glows: Temperature intensity of each clamping
    """
    num_clamps = len(capy_posns)
    if num_clamps == 0:
        return
    # cone_rad = capy_rads*.4
    cone_rads = cone_hgts / (1.732 * 2.)
    ds = array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    p = empty((num_clamps * len(ds), 3))
    axis = empty((num_clamps * len(ds), 3))
    rad = empty(num_clamps * len(ds))
    hgt = empty(num_clamps * len(ds))
    for i, xd in enumerate(ds):
        pp = xd * (cone_hgts)[:, None]
        p[i::6] = (capy_posns + pp)[:]
        axis[i::6] = tile(-xd, (num_clamps, 1))
        rad[i::6] = cone_rads
        hgt[i::6] = cone_hgts
    glowing_tube_cones(
        p, rad, hgt, axis, repeat(glows, 6), clim, opacity=1., plotter=plotter
    )


def heat_sources(
    capy_posns,
    capy_rads,
    heat_rads,
    dirn,
    line_size,
    heat_color,
    line_color,
    stripe_color,
    opacity,
    plotter,
):
    if len(capy_posns) == 0:
        return
    cone_hgts = heat_rads * 1.732
    cone_posn = capy_posns + dirn * (capy_rads + cone_hgts)[:, None]
    axis = tile(-dirn, (len(capy_rads), 1))
    tube_heat_cones(
        cone_posn,
        heat_rads,
        cone_hgts,
        axis,
        heat_color,
        stripe_color,
        opacity=opacity,
        plotter=plotter,
    )
    '''
    # radius estimation doesnt work well
    tube_cylinders(
        cone_posn,
        full(len(heat_rads), line_rad),
        cone_hgts,
        -axis,
        line_color,
        opacity=opacity,
        figure=figure,
        name=name + ' tail')
    '''
    point_count = len(capy_posns) * 2
    line_points = empty((point_count, 3))
    line_points[:-1:2] = cone_posn
    line_points[1::2] = cone_posn - axis * cone_hgts[:, None]
    from_idx = arange(0, point_count - 1, 2)
    to_idx = arange(1, point_count, 2)
    conns = array([from_idx, to_idx]).T
    lines(
        line_points,
        conns,
        line_size,
        line_color,
        opacity,
        plotter,
    )


def cyl(
    posn,
    dirn,
    rad,
    lght,
    color,
    opacity,
    plotter,
):
    plotter.add_mesh(
        pv.Cylinder(
            posn,
            dirn,
            rad,
            lght,
            capping=False,
            resolution=20,
        ),
        opacity=opacity,
        show_edges=False,
        color=color,
    )


def cyl_cap(
    posn,
    inner_rad,
    outer_rad,
    normal,
    color,
    opacity,
    plotter,
):
    plotter.add_mesh(
        pv.Disc(
            posn,
            inner_rad,
            outer_rad,
            normal,
            c_res=20,
        ),
        opacity=opacity,
        show_edges=False,
        color=color,
    )


def block_face(
    points,
    color,
    opacity,
    plotter,
):
    plotter.add_mesh(
        pv.Quadrilateral(
            points,
        ),
        opacity=opacity,
        show_edges=False,
        # line_width=10,
        # edge_color=color,
        color=color,
        ambient=1.,  # prevent color flicker of wireframe lines
        specular=0.,
        diffuse=0.,
    )


def dock(
    posn,
    inner_rad,
    outer_rad,
    normal,
    color,
    edge_color,
    opacity,
    plotter,
):
    plotter.add_mesh(
        pv.Disc(
            posn,
            inner_rad,
            outer_rad,
            normal,
            c_res=20,
        ),
        opacity=opacity,
        show_edges=False,
        color=color,
        edge_color=edge_color,
    )


if __name__ == '__main__':
    from thermca import Mesh

    capy_posns = array([[0., 0., 0.], [1., 1., 1.], [4., 1., 1.]])
    capy_rads = array([1., .5, .2])
    conns = array([[0, 1], [1, 2]])
    temps = array([20., 22., 24.])
    simple_block = Mesh(
        points=array(
            [
                [.8, .8, .8, .8, 0., 0., 0., 0., .4, .4, .4, .4, .4, .4],
                [.6, .6, 0., 0., .6, .6, 0., 0., .6, 0., .6, 0., .3, .3],
                [0., .4, 0., .4, 0., .4, 0., .4, .4, .4, 0., 0., .4, 0.],
            ]
        ).T,
        cell_blocks=[
            array(
                [
                    [2, 13, 2, 8, 12, 13, 2, 8, 8, 13, 12, 8, 13, 12, 12, 13, 12, 8],
                    [12, 5, 12, 13, 11, 7, 12, 13, 1, 7, 11, 13, 7, 1, 11, 5, 11, 13],
                    [13, 4, 0, 0, 13, 5, 3, 5, 0, 11, 7, 10, 6, 0, 9, 6, 2, 12],
                    [0, 10, 3, 10, 7, 12, 9, 12, 12, 6, 9, 5, 5, 3, 2, 4, 13, 0],
                ]
            ).T,
            array([[6, 3, 9, 11], [11, 9, 7, 2], [7, 2, 11, 9]]).T,
            array(
                [[6, 11, 4, 10, 2, 13], [4, 6, 10, 0, 11, 0], [13, 13, 13, 13, 13, 2]]
            ).T,
            array([[3, 1], [2, 3], [0, 0]]).T,
            array([[10, 10, 5, 1], [4, 8, 8, 0], [5, 0, 10, 8]]).T,
            array([[6, 5], [7, 4], [5, 6]]).T,
            array([[5, 8, 7, 1, 9, 3], [7, 5, 9, 8, 3, 1], [12, 12, 12, 12, 12, 12]]).T,
        ],
        block_types=[
            'tetra',
            'triangle',
            'triangle',
            'triangle',
            'triangle',
            'triangle',
            'triangle',
        ],
        block_names=['noname1', 'bottom', 'back', 'right', 'top', 'left', 'front'],
    )

    pl = pv.Plotter(lighting='three lights')
    # clr_theme = color_themes['bright']
    font_size, small_line, thick_line = _adjust_drawings_to_screen_dpi(195)

    # colored_and_scaled_spheres(capy_posns, capy_rads, temps, (min(temps), max(temps)), True, pl)
    # scaled_spheres(capy_posns, capy_rads, color=(.5, .5, .5), plotter=pl)
    # pp.colored_points(capy_posns, temps, size=30., plotter=p)
    # pp.points(capy_posns, color=(.5, .5, .5), size=30., plotter=p)
    # pp.lines(capy_posns, conns=conns, width=15, color=(.5, .5, .5), opacity=.5, plotter=p)
    # pp.scaled_tubes(capy_posns, conns, capy_rads, color=(.5, .5, .5), opacity=.5, capped=False, plotter=pl)
    # labels(capy_posns, [str(posn) for posn in capy_posns], text_color=(1., 1., 1.), shape_color=(.5, .5, .5), font_size=30, plotter=pl)
    # pp.triangle_mesh(points, bottom_triangles, sfc_color=(1., 0., 1.), edge_color=(0., 0., 0.), plotter=pl)
    # pp.triangle_mesh(points, bottom_triangles, sfc_color=None, edge_color=(1., 0., 0.), plotter=pl)
    # pp.tetra_mesh(simple_block.points, simple_block.cell_blocks[0], sfc_color=None, edge_color=(0., 0., 0.), opacity=.5, plotter=pl)
    # plot_mesh(simple_block, wireframe_sfcs=False, hide=(), dpi=195, color_theme='bright', plotter=pl)
    # pl.add_axes(color=clr_theme['fg'])
    # https://github.com/pyvista/pyvista/issues/147
    # pl.show_bounds(padding=.1, font_size=font_size)  # same as show_grid
    # colored_and_scaled_tubes(capy_posns, conns, capy_rads, colors=array([[1., 0., 0.], [0., 1., 0.],]), opacity=1., capped=True, plotter=pl)
    # colored_tetra_mesh(simple_block.points, simple_block.cell_blocks[0], colors=simple_block.points[:, 1], plotter=pl,)
    # colored_and_scaled_tubes(capy_posns, conns, capy_rads, colors=array([[1., 0., 0.], [0., 1., 0.], ]), opacity=1., capped=True, plotter=pl)
    # glowing_scaled_tubes(capy_posns, conns, capy_rads, glow=array([.2, .4, .6]), opacity=1., capped=True, plotter=pl)
    glowing_temp_clamping(
        capy_posns,
        cone_hgts=array([.2, .4, .6]),
        glows=array([.2, .4, .6]),
        plotter=pl,
    )
    axes(color=(1., 1., 1.), font_size=font_size, use_2d=False, plotter=pl)
    _set_initial_view(pl, draw2d=True)

    pl.show()
