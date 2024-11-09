"""Test mesh implementation"""

import os

from numpy import genfromtxt, allclose, asarray, unique
import meshio

from thermca import *


# create a simple pyramid with tetrahedron body and 5 triangle surfaces
point_locns = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1), (.5, 1, .5)]
cell_blocks = [
    # pyramid body
    [(0, 1, 3, 4), (1, 2, 3, 4)],
    # pyramid surfaces
    [(3, 2, 4)], [(0, 1, 4)], [(0, 3, 4)], [(1, 2, 4)], [(0, 1, 3), (1, 2, 3)],
]
cell_block_types = ['tetra', 'triangle', 'triangle', 'triangle', 'triangle', 'triangle']
cell_block_names = ['pyramid', 'front', 'back', 'left', 'right', 'bottom']
mesh = Mesh(point_locns, cell_blocks, cell_block_types, cell_block_names)


def read_xdmf(file_name):
    # xdmf doesn't support multiple cell blocks of same cell type
    # distinguish surfaces by cell tags
    mesh = meshio.read(file_name, file_format="xdmf")
    cell_blocks = []
    block_types = []
    # block_names = []
    for cell_type in ['tetra', 'triangle']:
        for idx, cell_block in enumerate(mesh.cells):
            if cell_block.type == cell_type:
                # ints as cell block tags for each cell
                cells, tags = cell_block.data, mesh.cell_data["cell_tags"][idx]
                tag_set, first_idx = unique(tags, return_index=True)
                sorted_tag_set = [tag for _, tag in sorted(zip(first_idx, tag_set))]
                for tag in sorted_tag_set:
                    tag_mask = tags == tag
                    cell_blocks.append(cells[tag_mask])
                    # block_names.append(mesh.cell_tags.get(tag, ["noname"])[0])
                    block_types.append(cell_type)
    return mesh.points, cell_blocks, block_types


def test_init():
    assert allclose(mesh.points, asarray(point_locns))
    for block, block_ref in zip(mesh.cell_blocks, cell_blocks):
        assert allclose(block, asarray(block_ref))
    assert allclose(mesh.cell_blocks[0], asarray(cell_blocks[0]))
    assert allclose(mesh.cell_blocks[1], asarray(cell_blocks[1]))


def test_write_and_read():
    # Salome .med
    mesh.write("test.med")
    mesh_read = Mesh.read("test.med")
    os.remove("test.med")
    assert allclose(mesh.points, mesh_read.points)
    for block, block_read in zip(mesh.cell_blocks, mesh_read.cell_blocks):
        assert allclose(block, block_read)
    assert mesh.block_names == mesh_read.block_names
    assert mesh.block_types == mesh_read.block_types
    # Fenics .xdmf (only write without names supported, read is not supported)
    mesh.write("test.xdmf")
    points_read, cell_blocks_read, block_types_read = read_xdmf("test.xdmf")
    os.remove("test.xdmf")
    os.remove("test.h5")
    assert allclose(mesh.points, points_read)
    for block, block_read in zip(mesh.cell_blocks, cell_blocks_read):
        assert allclose(block, block_read)
    assert mesh.block_types == block_types_read







