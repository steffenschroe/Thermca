import pathlib
import warnings
from textwrap import indent
import pprint
from functools import wraps

# fmt: off
from numpy import (
    array, vstack, hstack, full, printoptions, asarray, unique, int64, float64, arange,
    min as amin, max as amax, sqrt as asqrt, dot, sort, newaxis, bool_, ones, empty
)
# fmt: on
from numba import jit
import meshio
from scipy import spatial

from thermca import vector3d as v3d
import thermca.plot.mesh as plot

TRIANGLE = 'triangle'  # Triangle cells
TETRA = 'tetra'  # Tetrahedron cells


@jit(nopython=True)
def _extract_surface_fast(sorted_tetras):
    surface = set()
    for tetra in sorted_tetras:
        for face in (
            (tetra[0], tetra[1], tetra[2]),
            (tetra[0], tetra[1], tetra[3]),
            (tetra[0], tetra[2], tetra[3]),
            (tetra[1], tetra[2], tetra[3]),
        ):
            if face in surface:
                surface.remove(face)
            else:
                surface.add(face)
    if surface:
        return array(list(surface), dtype=int64)
    else:
        return asarray(sorted_tetras, dtype=int64)


def extract_surface(tetra_cells):
    """Extract surface triangles from tetrahedron cells"""
    # https://stackoverflow.com/questions/66607716/how-to-extract-surface-triangles-from-a-tetrahedral-mesh
    sorted_tetra_cells = sort(tetra_cells, axis=1)
    return _extract_surface_fast(sorted_tetra_cells)


def inner_point_cells_of_tetras(tetra_cells):
    """ "Inner point cells of tetrahedron cells (without surface)"""
    surface = extract_surface(tetra_cells)
    surface_point_idxs = set(surface.ravel())
    inner_point_idxs = set(tetra_cells.ravel()) - surface_point_idxs
    return array(list(inner_point_idxs), dtype=int64).reshape((-1, 1))


@jit(nopython=True)
def _extract_border_fast(sorted_triangles):
    border = set()
    for triangle in sorted_triangles:
        for line in (
            (triangle[0], triangle[1]),
            (triangle[0], triangle[2]),
            (triangle[1], triangle[2]),
        ):
            if line in border:
                border.remove(line)
            else:
                border.add(line)
    if border:
        return array(list(border), dtype=int64)
    else:
        return asarray(sorted_triangles, dtype=int64)


def extract_border(triangle_cells):
    """Extract border line cells from triangle cells"""
    sorted_triangles = sort(triangle_cells, axis=1)
    return _extract_border_fast(sorted_triangles)


def inner_point_cells_of_triangles(triangle_cells):
    """ "Inner point cells of triangle cells (without border)"""
    boarder_lines = extract_border(triangle_cells)
    border_point_idxs = set(boarder_lines.ravel())
    inner_point_idxs = set(triangle_cells.ravel()) - border_point_idxs
    return array(list(inner_point_idxs), dtype=int64).reshape((-1, 1))


@jit(nopython=True)
def _intersecting_plane(body_tetras, partial_body_tetras):
    """Return triangles near intersecting plane from surface of two
    precalculated tetrahedron bodies. The first body contains the full
    volume. The second contains one of the partial bodies cut by an
    intersecting plane. The function returns the surface triangles of
    the cutting surface of the partial body.
    """

    def extract_surface_fast(sorted_tetras):
        surface = set()
        for tetra in sorted_tetras:
            for face in (
                (tetra[0], tetra[1], tetra[2]),
                (tetra[0], tetra[1], tetra[3]),
                (tetra[0], tetra[2], tetra[3]),
                (tetra[1], tetra[2], tetra[3]),
            ):
                if face in surface:
                    surface.remove(face)
                else:
                    surface.add(face)
        return surface

    body_surf = extract_surface_fast(body_tetras)
    partial_body_surf = extract_surface_fast(partial_body_tetras)
    plane_triangles = partial_body_surf - body_surf
    return array(list(plane_triangles), dtype=int64)


def extract_intersecting_plane_faces(points, tetra_cells, cut_point, cut_vec):
    """Get faces of tetrahedron cell block near intersecting plane

    Args:
        points: Vertices locations of tetrahedron cells
        tetra_cells: Tetrahedron cell block
        cut_point: Point in cut plane
        cut_vec: Normal vector of cut plane
    """

    # Approach: (1) get triangle surface of entire body, (2) get triangle
    # surface of one part of divided body, (3) get triangles of divided
    # body that are not in surface of entire body
    # TODO: Typically the plane is calculated for every triangle
    #  surface. This includes the costly search of the outer
    #  surface for every triangle surface. To reduce this effort,
    #  the loop could be done over all surfaces and the outer
    #  surface triangle set could be maintained over the loop.

    # Create cutting plane
    point_in_plane = asarray(cut_point)
    norm_vec_plane = asarray(cut_vec)
    # Plane is a*x + b*y + c*z + d = 0 with normal [a, b, c]
    d = -point_in_plane.dot(norm_vec_plane)

    # Get cell centers
    sorted_tetras = sort(tetra_cells, axis=1)  # Needed for surface detection
    cell_centers = points[sorted_tetras].mean(1)

    # Extract cells of one partial body
    distance = cell_centers.dot(norm_vec_plane) + d
    mask = distance < 0
    partial_tetras = sorted_tetras[mask, :]

    return _intersecting_plane(sorted_tetras, partial_tetras)


@jit(nopython=True)
def _inner_tetras_mask(tetra_cells, surf_triangle_cells):
    """ "Remove outer surface layer of tetrahedrons"""
    surf_points = set(surf_triangle_cells.ravel())
    inner_tet_mask = ones(len(tetra_cells), dtype=bool_)
    for i, tet in enumerate(tetra_cells):
        for tet_vert in tet:
            if tet_vert in surf_points:
                inner_tet_mask[i] = False
    return inner_tet_mask


def inner_tetras(tetra_cells):
    """ "Remove outer surface layer of tetrahedron cells"""
    surf_cells = extract_surface(tetra_cells)
    return tetra_cells[_inner_tetras_mask(tetra_cells, surf_cells)]


@jit(nopython=True)
def tetra_vols(points, tetras):
    """Volumes of tetrahedron cells"""

    def tetra_vol(verts):
        """Volume of tetrahedron"""

        def determinant_3x3(vec0, vec1, vec2):
            return (
                vec0[0] * (vec1[1] * vec2[2] - vec2[1] * vec1[2])
                - vec0[1] * (vec1[0] * vec2[2] - vec2[0] * vec1[2])
                + vec0[2] * (vec1[0] * vec2[1] - vec2[0] * vec1[1])
            )

        return (
            determinant_3x3(
                v3d.sub(verts[0], verts[1]),
                v3d.sub(verts[1], verts[2]),
                v3d.sub(verts[2], verts[3]),
            )
            / 6.0
        )

    num_tets = len(tetras)
    volus = empty(num_tets)
    for i in range(num_tets):
        volus[i] = tetra_vol(points[tetras[i]])
    return volus


def tetras_vol(points, tetra_cells):
    """Summed volume of tetrahedron cells"""
    return sum(abs(tetra_vols(points, tetra_cells)))


@jit(nopython=True)
def triangle_areas(points, triangle_cells):
    """Areas of triangle cells"""

    def triangle_area(verts):
        return (
            v3d.norm(
                v3d.cross(
                    v3d.sub(verts[0], verts[1]),
                    v3d.sub(verts[1], verts[2]),
                )
            )
            / 2.0
        )

    num_triangles = len(triangle_cells)
    areas = empty(num_triangles)
    for i in range(num_triangles):
        areas[i] = triangle_area(points[triangle_cells[i]])
    return areas


def triangles_area(points, triangle_cells):
    """Summed surface area of triangle cells"""
    return sum(triangle_areas(points, triangle_cells))


def tetras_center(points, tetra_cells):
    """Center of tetrahedron cells overall volume"""
    cell_centers = points[tetra_cells].mean(1)
    volus = tetra_vols(points, tetra_cells)
    return sum(volus[:, newaxis] * cell_centers) / sum(volus)


def triangles_center(points, tri_cells):
    """Center of triangle cells overall surface area"""
    cell_centers = points[tri_cells].mean(1)
    areas = triangle_areas(points, tri_cells)
    return sum(areas[:, newaxis] * cell_centers) / sum(areas)


def block_point_near_center(points, cell_block):
    """Index of the nearest cell vertex to geometric center of vertices"""
    cell_points = points[cell_block.ravel()]
    point_cloud_center = cell_points.mean(axis=0)
    # Nearest point to center
    center_point_idx = spatial.KDTree(cell_points).query(point_cloud_center)[1]
    # Index of mesh point in geometric center of surfaces
    # Translate index of surface cell points to index of mesh points
    cell_size = cell_block.shape[1]
    nearest_point = cell_block[
        center_point_idx // cell_size, center_point_idx % cell_size
    ]
    return nearest_point


def block_center_point(points, cell_block):
    """Get mesh point near center of cells and try to avoid points on
    border of triangles and surface of tetrahedrons
    """
    if cell_block.shape[1] == 4:  # Tetrahedrons
        point_cells = inner_point_cells_of_tetras(cell_block)
        cells = cell_block if point_cells.size == 0 else point_cells
    elif cell_block.shape[1] == 3:  # Triangles
        point_cells = inner_point_cells_of_triangles(cell_block)
        cells = cell_block if point_cells.size == 0 else point_cells
    else:
        raise Exception("Cell type not supported!")
    center_point_idx = block_point_near_center(points, cells)
    return points[center_point_idx]


class Mesh:
    """Geometry of bodies and its surfaces as connected mesh cells

    Args:
        points: Locations of the mesh points as a two-dimensional
            list or ndarray of shape (number of points, 3). The second
            dimension holds the x, y and z coordinates.
        cell_blocks: List of cell blocks (list or ndarray) containing
            indices pointing to the points array.
        block_types: List containing the type of each of the cell
            blocks: `TETRA` for cells of tetrahedrons and `TRIANGLE`
            for triangle surfaces.
        block_names: List containing the name of each of the cell
            blocks. The names must be valid python identifiers.

    Example::

        >>> # A simple pyramid with tetrahedron body and 5 triangle surfaces
        >>> points = [
        ...     (0.0, 0.0, 0.0),
        ...     (1.0, 0.0, 0.0),
        ...     (1.0, 0.0, 1.0),
        ...     (0.0, 0.0, 1.0),
        ...     (0.5, 1.0, 0.5),
        ... ]
        >>> cell_blocks = [
        ...     # pyramid body
        ...     [(0, 1, 3, 4), (1, 2, 3, 4)],
        ...     # pyramid surfaces
        ...     [(3, 2, 4)],
        ...     [(0, 1, 4)],
        ...     [(0, 3, 4)],
        ...     [(1, 2, 4)],
        ...     [(0, 1, 3), (1, 2, 3)],
        ... ]
        >>> block_types = [TETRA, TRIANGLE, TRIANGLE, TRIANGLE, TRIANGLE, TRIANGLE]
        >>> block_names = ['pyramid', 'front', 'back', 'left', 'right', 'bottom']
        >>> mesh = Mesh(points, cell_blocks, block_types, block_names)
    """

    # In contrast to meshio, data and meta information regarding the data
    # is separated. In meshio each cell block contains the types as well
    # as the actual geometric information. During processing, both the
    # meta information and the data have oftentimes to be stripped apart.
    # Also, the data layout has to be changed, if additional meta
    # information has to be added. With the current data layout, this
    # can be avoided.

    def __init__(self, points, cell_blocks, block_types, block_names):
        self.points = asarray(points)
        self.cell_blocks = [asarray(cells) for cells in cell_blocks]
        self.block_types = list(block_types)
        self.block_names = list(block_names)

    def extract_type(self, type_name):
        """Return mesh with cell blocks of given type"""
        idxs = [
            idx
            for idx, block_type in enumerate(self.block_types)
            if block_type == type_name
        ]
        return Mesh(
            self.points,
            [self.cell_blocks[idx] for idx in idxs],
            [self.block_types[idx] for idx in idxs],
            [self.block_names[idx] for idx in idxs],
        )

    def write(self, file_name, file_format=None):
        """Save mesh to file

        Gmsh file_format specification is mandatory because .msh is
        used by Gmsh and Ansys.

        Args:
            file_name: Name of the file
            file_format:
                'xdmf' for Fenics format; doesn't support names for
                    tags

                'med' for Salome format

                'gmsh' for Gmsh format 4.1 binary,
                'gmsh22' for Gmsh format 2.2 binary,
                'gmsh_txt' for Gmsh format 4.1 text,
                'gmsh22_txt' for Gmsh format 2.2 text,
                Gmsh formats can't save multiple cell types for now
        """
        path = pathlib.Path(file_name)
        if not file_format:
            file_format = path.suffix[1:]
        file_format = file_format.lower()

        # Flatten cell blocks in to one cell block per cell format
        # because this format is supported by mesh formats used in
        # Thermca. E.g. xdmf doesn't support multiple cell blocks of
        # the same cell type. The surfaces get distinguished by cell
        # tags. Each Thermca cell block gets an individual integer tag
        # which is mapped to its cells.
        triangles = []
        triangle_idxs = []
        triangle_names = []
        tetras = []
        tetra_idxs = []
        tetra_names = []
        cell_sets = {}
        curr_cell_idx = 0
        for i, (cells, block_type, name) in enumerate(
            zip(self.cell_blocks, self.block_types, self.block_names)
        ):
            cell_count = len(cells)
            cell_block_idxs = full(cell_count, i, dtype=int64)
            cell_set_idxs = arange(curr_cell_idx, curr_cell_idx + cell_count, dtype=int64)
            cell_sets[name] = [cell_set_idxs]
            curr_cell_idx += cell_count
            if block_type == TRIANGLE:
                triangles.append(cells)
                triangle_idxs.append(cell_block_idxs)
                triangle_names.append(name)
            elif block_type == TETRA:
                tetras.append(cells)
                tetra_idxs.append(cell_block_idxs)
                tetra_names.append(name)
            else:
                raise Exception("Cell type not supported!")
        flat_cells = []
        flat_cell_idxs = []
        if triangles:
            flat_cells.append(('triangle', vstack(triangles)))
            flat_cell_idxs.append(hstack(triangle_idxs))
        if tetras:
            flat_cells.append(('tetra', vstack(tetras)))
            flat_cell_idxs.append(hstack(tetra_idxs))

        if file_format == 'xdmf':
            # XDMF indices start at 0; Names are not supported
            meshio.write_points_cells(
                file_name,
                points=self.points,
                cells=flat_cells,
                cell_data={'cell_tags': flat_cell_idxs},
                file_format="xdmf",
            )

        elif file_format == 'med':
            # Salome .med saves int tags negative: 0 seems to mean no
            # tag, but positive indices also work for importing meshes
            # into salome.
            cell_tags = {
                idx[0] + 1: [name]
                for idx, name in zip(
                    triangle_idxs + tetra_idxs, triangle_names + tetra_names
                )
            }
            flat_cell_tags = [idxs + 1 for idxs in flat_cell_idxs]
            med_mesh = meshio.Mesh(
                points=self.points,
                cells=flat_cells,
                cell_data={'cell_tags': flat_cell_tags},
            )
            # Hack: Writing of `mesh.cell_tags` works but the attribute
            # can only be added outside `__init__`.
            med_mesh.cell_tags = cell_tags
            med_mesh.write(file_name, file_format='med')

        elif file_format in ('gmsh', 'gmsh22', 'gmsh_txt', 'gmsh22_txt'):
            # Meshio can only write one cell type for now.
            # Because `cell_sets` don't get written, gmsh:physical and
            # field_data is used to define surfaces with names.
            flat_cell_tags = [idxs + 1 for idxs in flat_cell_idxs]
            field_data = {}
            # Gmsh tags are from 1 upwards
            for name, idxs in zip(triangle_names, triangle_idxs):
                field_data[name] = array(
                    [idxs[0] + 1, 2], dtype=int64
                )  # 2 == dimension
            for name, idxs in zip(tetra_names, tetra_idxs):
                field_data[name] = array(
                    [idxs[0] + 1, 3], dtype=int64
                )  # 3 == dimension

            if file_format in ('gmsh', 'gmsh22'):
                binary = True
            else:
                binary = False
                file_format = file_format[:-4]  # Strip "_txt"

            meshio.write_points_cells(
                file_name,
                self.points,
                cells=flat_cells,
                cell_data={
                    'gmsh:physical': flat_cell_tags,
                },
                field_data=field_data,
                file_format=file_format,
                # cell_sets=cell_sets,  # cell_sets don't get written
                binary=binary,
            )
        else:
            raise Exception(
                f"Can't write file format '{file_format}' of file '{file_name}'"
            )

    @classmethod
    def read(cls, file_name, file_format=None):
        """Read mesh from file

        Args:
            file_name: Name of the file
            file_format:
                'med' for Salome format,
                'msh' or 'gmsh' for Gmsh
        """
        path = pathlib.Path(file_name)
        if not path.exists():
            raise Exception(f"File {file_name} not found.")
        if not file_format:
            file_format = path.suffix[1:]
        file_format = file_format.lower()
        if file_format == 'cgns':
            node_lctns, cell_blocks, block_types, block_names = cls._read_ansys_cgns(
                file_name
            )
            return Mesh(node_lctns, cell_blocks, block_types, block_names)
        elif file_format == 'med':
            # Salome-mesh
            points, cell_blocks, block_types, block_names = cls._read_salome_med(
                file_name
            )
            return Mesh(points, cell_blocks, block_types, block_names)
        elif file_format in ['msh', 'gmsh']:
            # Gmsh
            node_lctns, cell_blocks, block_types, block_names = cls._read_gmsh_msh(
                file_name
            )
            return Mesh(node_lctns, cell_blocks, block_types, block_names)
        else:
            raise Exception(
                f'Can not read file format "{file_format}" of file "{file_name}"'
            )

    @staticmethod
    def _read_gmsh_msh(file_name):
        # Gmsh supports multiple cell blocks of the same cell type.
        # Surface names can only be distinguished by cell tags.
        # Cell tags can be in gmsh:physical or gmsh:geometrical,
        # whereby gmsh:physical is mostly used. Tags spreading over
        # multiple cell blocks are not supported.
        io_mesh = meshio.read(file_name, file_format="gmsh")
        cell_blocks = []
        block_types = []
        block_names = []
        cell_tags = {tag: name for name, (tag, dim) in io_mesh.field_data.items()}
        for cell_type in ['tetra', 'triangle']:
            for idx, cell_block in enumerate(io_mesh.cells):
                if cell_block.type == cell_type:
                    # Integers as cell block tags for each cell
                    cells, tags = (
                        cell_block.data,
                        io_mesh.cell_data['gmsh:physical'][idx],
                    )
                    for tag in unique(tags):
                        tag_mask = tags == tag
                        cell_blocks.append(asarray(cells[tag_mask], dtype=int64))
                        block_names.append(
                            cell_tags.get(tag, f'noname{len(cell_blocks)}')
                        )
                        block_types.append(cell_type)
        return (
            asarray(io_mesh.points, dtype=float64),
            cell_blocks,
            block_types,
            block_names,
        )

    @staticmethod
    def _read_salome_med(file_name):
        # Salome med doesn't support multiple cell blocks of same cell type.
        # Therefore, surfaces are distinguished by cell tags.
        io_mesh = meshio.read(file_name, file_format="med")
        cell_blocks = []
        block_types = []
        block_names = []
        for cell_type in ['tetra', 'triangle']:
            for idx, cell_block in enumerate(io_mesh.cells):
                if cell_block.type == cell_type:
                    # Ints as cell block tags for each cell
                    cells, tags = cell_block.data, io_mesh.cell_data["cell_tags"][idx]
                    for tag in unique(tags):
                        tag_mask = tags == tag
                        cell_blocks.append(asarray(cells[tag_mask], dtype=int64))
                        tag_names = io_mesh.cell_tags.get(
                            tag, [f'noname{len(cell_blocks)}']
                        )
                        if len(tag_names) > 1:
                            tag_name = "_".join(tag_names)
                            warnings.warn(
                                f"Surface groups: {tag_names} are overlapping!"
                                f"Created group {tag_name} at overlapping region."
                            )
                        else:
                            tag_name = tag_names[0]
                        block_names.append(tag_name)
                        block_types.append(cell_type)
        return (
            asarray(io_mesh.points, dtype=float64),
            cell_blocks,
            block_types,
            block_names,
        )

    def bounding_box(self):
        """The bounding box of the points of the mesh.
        A 2dimensional array where the first row contains the minimum values in each dimension
        and the second row contains the maximum values."""
        bbox = empty((2, 3))
        bbox[0] = amin(self.points, axis=0)
        bbox[1] = amax(self.points, axis=0)
        return bbox

    def bounding_box_diag_lgth(self):
        """Length of the bounding box diagonal"""
        bbox = self.bounding_box()
        diag = bbox[1] - bbox[0]
        return asqrt(dot(diag, diag))

    @wraps(plot.mesh)
    def plot(self, **kwargs):
        return plot.mesh(self, **kwargs)

    def __str__(self):
        with printoptions(threshold=2, precision=4):
            lines = [
                self.__class__.__name__ + '(',
                "    points=(",
                indent(str(self.points), "        ") + ")",
                "    cell_blocks=(",
                indent(pprint.pformat(self.cell_blocks), "        ") + ")",
                "    block_types=(",
                indent(pprint.pformat(self.block_types), "        ") + ")",
                "    block_names=(",
                indent(pprint.pformat(self.block_names), "        ") + ")",
            ]
        return '\n'.join(lines)
