#!/usr/bin/python
"""Creates high quality tetrahedron mesh from a triangle surface mesh
and maps the contained triangle surface groups to the new surface.

The tetrahedron mesh is created with fTetWild. The accuracy of the
resulting surface geometry is controlled by a given tolerance
regarding the input surface. This allows removing of small surface
features. The mesh density determines the computing load during
finite-element-calculations. It is controlled by a given target
tetrahedron edge length. This edge length is aimed for, if the geometry
permits it. Additionally, the mesh optimisation energy can be adjusted.

The surfaces triangles of the input surface mesh don't match the
surface of the created tetrahedron mesh. Therefore, the triangles of the
input mesh get mapped to the new triangle surface of the tetrahedron
mesh. For each surface triangle of the 3D mesh the nearest triangle of
the input mesh is evaluated and its triangle group is determined.

Input is a Salome .med mesh file containing named face-groups. For mesh
cells without a group assigned, Salome automatically adds a nameless
surface group. Therefore, the surface is closed and the file is suitable
as input for fTetWild.

The output is a Salome .med mesh that contains the tetrahedron volume
and the mapped triangle surface groups.
"""

import tempfile
import os
import sys
from pathlib import Path
import argparse
import warnings
import time

import meshio
from numpy import int64, full, hstack, vstack, unique
from trimesh import Trimesh, proximity

from thermca.mesh import Mesh


def _map_face_groups(body_mesh, hull_mesh_groups):
    """Map hull-mesh face groups to body mesh surface

    The hull-mesh is a high resolution surface mesh containing surface
    groups. The body-mesh is a low resolution mesh without surface
    specific information. This function maps the hull face groups to
    the body-mesh surface. This is done by finding the closest hull
    surface triangle to each body surface triangle center.
    Finally, the surface groups are added to the body-mesh.
    """
    body_surf = body_mesh.extract_surface(0)  # returns triangle indices
    body_face_centers = body_mesh.points[body_surf].mean(1)
    # Flatten input surface mesh groups
    hull_mesh = hull_mesh_groups.extract_type('triangle')
    hull_cells = []
    hull_group_tags = []
    for i, cells in enumerate(hull_mesh.cell_blocks):
        hull_cells.append(cells)
        hull_group_tags.append(full(len(cells), i, dtype=int64))
    hull_cells = vstack(hull_cells)
    # hull_cell_centers = hull_mesh.points[hull_cells].mean(1)
    hull_group_tags = hstack(hull_group_tags)
    # Find the nearest hull face for each body face center
    hull_trimesh = Trimesh(vertices=hull_mesh.points, faces=hull_cells)
    point_on_triangles, distances, hull_cell_idxs = proximity.closest_point(hull_trimesh, body_face_centers)
    # hull_cell_idxs = spatial.KDTree(hull_cell_centers).query(face_centers)[1]
    group_tags = hull_group_tags[hull_cell_idxs]
    # Create nested mesh cell blocks
    for tag in unique(group_tags):
        tag_mask = group_tags == tag
        body_mesh.cell_blocks.append(body_surf[tag_mask])
        body_mesh.block_names.append(hull_mesh.block_names[tag])
        body_mesh.block_types.append('triangle')
    # Test, if all face groups are in resulting mesh
    for name in hull_mesh.block_names:
        if name not in body_mesh.block_names:
            warnings.warn(f"Triangle group: '{name}' is not included in resulting mesh! "
                          f"Try a finer surface mesh with a smaller edge length factor L.")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "input_file",
    help="Input mesh file with closed surface of triangles and groups of triangles for various boundary conditions in Salome .med format"
)
parser.add_argument(
    "output_file",
    help="Output mesh file with volume of tetrahedrons and groups of surface triangles for various boundary conditions in Salome .med format"
)
parser.add_argument(
    "-l",
    help="ideal_edge_length = diagonal_of_bounding_box * L. (float, default: 0.05)"
)
parser.add_argument(
    "-l_abs",
    help="Ideal edge length as an absolute value (float, default determined via relative value L)"
)
parser.add_argument(
    "-e",
    help="surface_deviation_tolerance = diagonal_of_bounding_box * E. (float, default: 1e-3)"
)
parser.add_argument(
    "-e_abs",
    help="Surface deviation tolerance as an absolute value (float, default: determined via relative value E)"
)
parser.add_argument(
    "--stop_energy",
    help="Stop optimization when max. energy is lower than this. (float, default: 10.)"
)
default_tetwild_path = "fTetWild/build"
parser.add_argument(
    "--tetwild_path",
    help="Path to FloatTetWild_bin executable (default: {path of tetwild.py}/" + default_tetwild_path + " )"
)

args = parser.parse_args()

path = Path().absolute()  # Path of this python file
if args.tetwild_path is not None:
    tetwild_exe = args.tetwild_path + "/FloatTetwild_bin"
else:
    tetwild_exe = str(path/default_tetwild_path/"FloatTetwild_bin")


hull_mesh_name = args.input_file
hull_mesh = Mesh.read(hull_mesh_name)

tetwild_args = ""
if args.l is not None and args.l_abs is not None:
    raise Exception("Ideal edge length must be given as relative or as absolute value but not both!")
if args.l is not None:
    tetwild_args += " -l " + args.l
if args.l_abs is not None:
    tetwild_args += " -l " + str(float(args.l_abs)/hull_mesh.bounding_box_diag_lgth())
if args.e is not None and args.e_abs is not None:
    raise Exception("Surface deviation tolerance (envelope) must be given as relative or as absolute value but not both!")
if args.e is not None:
    tetwild_args += " -e " + args.e
if args.e_abs is not None:
    tetwild_args += " -e " + str(float(args.e_abs)/hull_mesh.bounding_box_diag_lgth())
if args.stop_energy is not None:
    tetwild_args += " --stop-energy " + args.stop_energy

with tempfile.TemporaryDirectory() as tmp_dir:
    orig_dir = os.getcwd()
    os.chdir(tmp_dir)
    # Salome exported .stl file also works as surface input
    hull_mesh.write('face_groups_mesh.msh', file_format='gmsh22_txt')
    print("### Creating 3D tetrahedron mesh")
    tet_wild_cmd = f"{tetwild_exe} -i face_groups_mesh.msh -o body.msh" + tetwild_args
    print(tet_wild_cmd)
    start_time = time.time()
    os.system(tet_wild_cmd)
    end_time = time.time()
    tetwild_time = end_time - start_time

    io_mesh = meshio.read("body.msh", file_format="gmsh")
    body_mesh = Mesh(io_mesh.points, [io_mesh.cells[0].data], [io_mesh.cells[0].type], ['body'])

    start_time = time.time()
    _map_face_groups(body_mesh, hull_mesh)
    merge_time = time.time() - start_time
    os.chdir(orig_dir)

body_mesh.plot().show()
print(f"### Writing mapped and merged mesh: {args.output_file}")
body_mesh.write(args.output_file)
print(f"Time in s for meshing: {tetwild_time}, for merging: {merge_time}")
body_mesh.write(args.output_file)
print(f"{len(body_mesh.points)=}")

