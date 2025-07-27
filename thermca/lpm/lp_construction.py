"""Construction of the lumped parameter system from body assemblies"""

from itertools import combinations, product
from numpy import arange, int64, array, hstack, zeros, repeat, add
from scipy.sparse import diags
from sparse import DOK

from thermca.lpm.asm import Asm, ForceConts
from thermca.lpm.cube import Cube, Left, Right, Btm, Top, Back, Front
from thermca.lpm.cyl import Cyl, Base, End, Inner, Outer
from thermca.lpm.asm import Surf
from thermca.lpm.lp_cube import LPCube
from thermca.lpm.lp_cyl import LPCyl
from thermca.lpm.lp_link_surf import LPLinkSurf

ABS_TOL = 0.000000001
MIN_OVLP = 0.000000001  # Required minimum overlap

X = 0
Y = 1
Z = 2


def faces_to_lp_faces(
    faces,
    blocks: list[Cube],
    cyls: list[Cyl],
    lp_blocks: list[LPCube],
    lp_cyls: list[LPCyl],
):
    lp_faces = []
    for face in faces:
        if face is None:
            raise Exception(
                "Got `None` instead of a face. May be it's a cylinder without inner face "
                "because the inner radius is 0."
            )
        match face.body:
            case Cube():
                lp_body = lp_blocks[blocks.index(face.body)]
            case Cyl():
                lp_body = lp_cyls[cyls.index(face.body)]
        match face:
            case Left():
                lp_faces.append(lp_body.face.left)
            case Right():
                lp_faces.append(lp_body.face.right)
            case Btm():
                lp_faces.append(lp_body.face.btm)
            case Top():
                lp_faces.append(lp_body.face.top)
            case Back():
                lp_faces.append(lp_body.face.back)
            case Front():
                lp_faces.append(lp_body.face.front)
            case Base():
                lp_faces.append(lp_body.face.base)
            case End():
                lp_faces.append(lp_body.face.end)
            case Inner():
                lp_faces.append(lp_body.face.inner)
            case Outer():
                lp_faces.append(lp_body.face.outer)
    return lp_faces


def create_lp_elems(asm):
    """Create lumped parameter elements from geometry elements"""

    blocks = asm._get_all_elems_of_type(Cube)
    cyls = asm._get_all_elems_of_type(Cyl)
    link_surfs = asm._get_all_elems_of_type(Surf)
    force_conts_list = asm._get_all_elems_of_type(ForceConts)

    lp_blocks = [LPCube.from_cube(block) for block in blocks]
    lp_cyls = [LPCyl.from_cyl(cyl) for cyl in cyls]
    lp_link_surfs = [
        LPLinkSurf.from_link_surf(link_surf, blocks, cyls, lp_blocks, lp_cyls)
        for link_surf in link_surfs
    ]
    force_cont_pairs = [
        tuple(faces_to_lp_faces(pair, blocks, cyls, lp_blocks, lp_cyls))
        for force_conts in force_conts_list
        for pair in force_conts.face_pairs
    ]
    return lp_blocks, lp_cyls, lp_link_surfs, force_cont_pairs


def is_close(a, b):
    diff = abs(b - a)
    return diff <= ABS_TOL


def greater_than(a, b):  # With overlap because edge to edge should be no contact
    return a - MIN_OVLP > b


def circ_rect_overlaps(
    circ_xposn, circ_yposn, circ_rad, rect_xposn, rect_yposn, rect_width, rect_hgt
):
    """Test, if circle and rectangle overlaps"""
    dist_x = abs(circ_xposn - rect_xposn - rect_width / 2.0)
    dist_y = abs(circ_yposn - rect_yposn - rect_hgt / 2.0)
    if dist_x > rect_width / 2.0 + circ_rad - MIN_OVLP:
        return False
    if dist_y > rect_hgt / 2.0 + circ_rad - MIN_OVLP:
        return False
    if dist_x <= rect_width / 2.0 - MIN_OVLP:
        return True
    if dist_y <= rect_hgt / 2.0 - MIN_OVLP:
        return True
    dx = dist_x - rect_width / 2.0
    dy = dist_y - rect_hgt / 2.0
    return dx**2 + dy**2 <= (circ_rad - MIN_OVLP) ** 2


def circ_contain_rect(
    circ_xposn, circ_yposn, circ_rad, rect_xposn, rect_yposn, rect_width, rect_hgt
):
    """Test, if rectangle is inside circle and doesn't overlap circle"""
    dx = max(abs(circ_xposn - rect_xposn), abs(rect_xposn + rect_width - circ_xposn))
    dy = max(abs(circ_yposn - rect_yposn), abs(rect_yposn + rect_hgt - circ_yposn))
    return (circ_rad + MIN_OVLP) ** 2 >= dx**2 + dy**2


def annuli_overlaps(
    xposn0, yposn0, inner_rad0, outer_rad0, xposn1, yposn1, inner_rad1, outer_rad1
):
    """Test, if two annuli overlap"""
    dist = ((xposn0 - xposn1) ** 2 + (yposn0 - yposn1) ** 2) ** 0.5
    # Outer circle overlap
    if dist <= outer_rad0 + outer_rad1 - MIN_OVLP:
        # Annulus 1 inside inner circle of annulus 0
        if inner_rad0 + MIN_OVLP > (dist + outer_rad1):
            return False
        # Annulus 0 inside inner circle of annulus 1
        if inner_rad1 + MIN_OVLP > (dist + outer_rad0):
            return False
        return True
    return False


def find_body_contacts(blocks, cyls):
    """Find contacts of body faces"""

    # # Block-block contacts with surface overlap
    block_block_conts = []
    for block0, block1 in combinations(blocks, 2):
        xmin0 = block0.posn[0]
        xmin1 = block1.posn[0]
        xmax0 = block0.posn[0] + block0.width
        xmax1 = block1.posn[0] + block1.width
        ymin0 = block0.posn[1]
        ymin1 = block1.posn[1]
        ymax0 = block0.posn[1] + block0.hgt
        ymax1 = block1.posn[1] + block1.hgt
        zmin0 = block0.posn[2]
        zmin1 = block1.posn[2]
        zmax0 = block0.posn[2] + block0.depth
        zmax1 = block1.posn[2] + block1.depth
        # On top or bottom
        if greater_than(xmax1, xmin0) and greater_than(xmax0, xmin1):
            if greater_than(zmax1, zmin0) and greater_than(zmax0, zmin1):
                if is_close(ymax1, ymin0):
                    block_block_conts.append((block1.face.top, block0.face.btm))
                elif is_close(ymax0, ymin1):
                    block_block_conts.append((block0.face.top, block1.face.btm))
            # On front or back
            elif greater_than(ymax1, ymin0) and greater_than(ymax0, ymin1):
                if is_close(zmax1, zmin0):
                    block_block_conts.append((block1.face.front, block0.face.back))
                elif is_close(zmax0, zmin1):
                    block_block_conts.append((block0.face.front, block1.face.back))
        # On left or right
        elif greater_than(ymax1, ymin0) and greater_than(ymax0, ymin1):
            if greater_than(zmax1, zmin0) and greater_than(zmax0, zmin1):
                if is_close(xmax1, xmin0):
                    block_block_conts.append((block1.face.right, block0.face.left))
                elif is_close(xmax0, xmin1):
                    block_block_conts.append((block0.face.right, block1.face.left))

    # # Cylinder-cylinder contacts
    cyl_cyl_conts = []
    for cyl0, cyl1 in combinations(cyls, 2):
        xmin0 = cyl0.posn[0]
        xmin1 = cyl1.posn[0]
        xmax0 = cyl0.posn[0] + cyl0.lgth
        xmax1 = cyl1.posn[0] + cyl1.lgth
        ymin0 = cyl0.posn[1] + cyl0.inner_rad
        ymin1 = cyl1.posn[1] + cyl1.inner_rad
        ymax0 = cyl0.posn[1] + cyl0.outer_rad
        ymax1 = cyl1.posn[1] + cyl1.outer_rad
        zmin0 = cyl0.posn[2]
        zmin1 = cyl1.posn[2]
        # Same axis
        if is_close(cyl0.posn[1], cyl1.posn[1]) and is_close(zmin1, zmin0):
            # Radial stacked with axial overlap
            if greater_than(xmax1, xmin0) and greater_than(xmax0, xmin1):
                if is_close(ymax1, ymin0):
                    cyl_cyl_conts.append((cyl1.face.outer, cyl0.face.inner))
                elif is_close(ymax0, ymin1):
                    cyl_cyl_conts.append((cyl0.face.outer, cyl1.face.inner))
        # Annuli overlap
        if is_close(xmin0, xmax1):
            if annuli_overlaps(
                cyl0.posn[1],
                cyl0.posn[2],
                cyl0.inner_rad,
                cyl0.outer_rad,
                cyl1.posn[1],
                cyl1.posn[2],
                cyl1.inner_rad,
                cyl1.outer_rad,
            ):
                cyl_cyl_conts.append((cyl0.face.base, cyl1.face.end))
        if is_close(xmin1, xmax0):
            if annuli_overlaps(
                cyl0.posn[1],
                cyl0.posn[2],
                cyl0.inner_rad,
                cyl0.outer_rad,
                cyl1.posn[1],
                cyl1.posn[2],
                cyl1.inner_rad,
                cyl1.outer_rad,
            ):
                cyl_cyl_conts.append((cyl1.face.base, cyl0.face.end))

    # # Block-cylinder contacts
    block_cyl_conts = []
    for block, cyl in product(blocks, cyls):
        block_xmin = block.posn[0]
        block_xmax = block.posn[0] + block.width
        cyl_xmin = cyl.posn[0]
        cyl_xmax = cyl.posn[0] + cyl.lgth

        if is_close(block_xmax, cyl_xmin):
            if circ_rect_overlaps(
                cyl.posn[1],
                cyl.posn[2],
                cyl.outer_rad,
                block.posn[1],
                block.posn[2],
                block.hgt,
                block.depth,
            ):
                if not circ_contain_rect(
                    cyl.posn[1],
                    cyl.posn[2],
                    cyl.inner_rad,
                    block.posn[1],
                    block.posn[2],
                    block.hgt,
                    block.depth,
                ):
                    block_cyl_conts.append((block.face.right, cyl.face.base))

        if is_close(block_xmin, cyl_xmax):
            if circ_rect_overlaps(
                cyl.posn[1],
                cyl.posn[2],
                cyl.outer_rad,
                block.posn[1],
                block.posn[2],
                block.hgt,
                block.depth,
            ):
                if not circ_contain_rect(
                    cyl.posn[1],
                    cyl.posn[2],
                    cyl.inner_rad,
                    block.posn[1],
                    block.posn[2],
                    block.hgt,
                    block.depth,
                ):
                    block_cyl_conts.append((block.face.left, cyl.face.end))

    return block_block_conts, cyl_cyl_conts, block_cyl_conts


def add_force_cont_pairs_to_grouped_conts(
    block_block_conts,
    cyl_cyl_conts,
    block_cyl_conts,
    force_cont_pairs,
):
    for cont_pair in force_cont_pairs:
        match (cont_pair[0].body, cont_pair[1].body):
            case (LPCube(), LPCube()):
                block_block_conts.append(cont_pair)
            case (LPCyl(), LPCyl()):
                cyl_cyl_conts.append(cont_pair)
            case (LPCube(), LPCyl()) | (LPCyl(), LPCube()):
                block_cyl_conts.append(cont_pair)
    return block_block_conts, cyl_cyl_conts, block_cyl_conts


def assemble(asm: Asm):

    lp_blocks, lp_cyls, lp_link_surfs, force_cont_pairs = create_lp_elems(asm)

    block_block_conts, cyl_cyl_conts, block_cyl_conts = find_body_contacts(
        lp_blocks, lp_cyls
    )

    block_block_conts, cyl_cyl_conts, block_cyl_conts = (
        add_force_cont_pairs_to_grouped_conts(
            block_block_conts,
            cyl_cyl_conts,
            block_cyl_conts,
            force_cont_pairs,
        )
    )

    elem_idxs = {}
    elem_slice = {}
    vols = []  # Node volumes (Mx_diag)
    posns = []  # Node positions
    vol_sum = 0.0
    node_count = 0
    for body in lp_blocks + lp_cyls:
        vols.extend(body.node_vols())
        posns.extend(body.node_lctns() + body.posn)
        elem_idxs[body] = arange(body.node_count(), dtype=int64) + node_count
        elem_slice[body] = slice(node_count, node_count + body.node_count())
        node_count += body.node_count()
        vol_sum += body.vol()
    num_body_nodes = node_count

    body_marg_idxs = {}
    surf_names = []
    for surf in lp_link_surfs:
        surf_names.append(surf.name)
        for face in surf.faces:
            posns.extend(face.node_lctns() + face.body.posn)
            idxs = arange(face.node_count(), dtype=int64) + node_count
            elem_idxs[face] = idxs
            body_marg_idxs[face] = (
                face.body_marg_node_idxs() + elem_slice[face.body].start
            )
            elem_slice[face] = slice(node_count, node_count + face.node_count())
            node_count += face.node_count()
    if len(lp_link_surfs) != len(set(surf_names)):
        raise Exception("Names of surfaces must be unique!")

    # Geometric ratio of conductances (Lx)
    geom_ratio_x = DOK((node_count, num_body_nodes))
    geom_ratio_y = DOK((node_count, num_body_nodes))
    geom_ratio_z = DOK((node_count, num_body_nodes))
    lpm_gera = [geom_ratio_x, geom_ratio_y, geom_ratio_z]
    for body in lp_blocks + lp_cyls:
        sl = elem_slice[body]
        body_gera = body.node_geras()
        for lpm_gera_dirn, body_gera_dirn in zip(lpm_gera, body_gera):
            gera_idxs = body_gera_dirn.to_coo().nonzero()
            lpm_gera_idxs = (gera_idxs[0] + sl.start, gera_idxs[1] + sl.start)
            lpm_gera_dirn.aix[lpm_gera_idxs] = body_gera_dirn.aix[gera_idxs]

    for cont in block_block_conts + cyl_cyl_conts + block_cyl_conts + force_cont_pairs:
        face0, face1 = cont
        idxs0 = face0.body_marg_node_idxs() + elem_slice[face0.body].start
        idxs1 = face1.body_marg_node_idxs() + elem_slice[face1.body].start
        face0_node_geras = face0.node_geras()
        face1_node_geras = face1.node_geras()
        if face0_node_geras[X] is not None:
            geras0, geras1 = face0_node_geras[X], face1_node_geras[X]
            lpm_gera_dirn = lpm_gera[X]
        elif face0_node_geras[Y] is not None:
            geras0, geras1 = face0_node_geras[Y], face1_node_geras[Y]
            lpm_gera_dirn = lpm_gera[Y]
        elif face0_node_geras[Z] is not None:
            geras0, geras1 = face0_node_geras[Z], face1_node_geras[Z]
            lpm_gera_dirn = lpm_gera[Z]
        if len(geras0) == 1 or len(geras1) == 1 or len(geras0) == len(geras1):
            if len(geras0) == 1:
                idxs0 = repeat(idxs0, len(geras1))
            if len(geras1) == 1:
                idxs1 = repeat(idxs1, len(geras0))
            geras = 1.0 / (1.0 / geras0 + 1.0 / geras1)
            gera_idxs = (hstack((idxs0, idxs1)), hstack((idxs1, idxs0)))
            lpm_gera_dirn.aix[gera_idxs] = hstack((geras, geras))

    # Conductances from margin nodes to surfaces are needed to compute conductances
    # for temperature boundary conditions on real surfaces and combined conductances of
    # body margins and film coefficients for heat transfer to neighboring lp-systems.
    # Margin nodes are the closest nodes to the body surfaces and part of the nodes of
    # unknown temperatures (T_u).
    # The surface nodes are considered as 'virtual' nodes because they don't exist in
    # lp-systems and therefore their temperatures T_s do not need to be calculated.
    # The conductance's to surfaces L_su are only saved in the direction from surface (s)
    # to the margin nodes. In this way they can be cheaply cut off from CSR matrices by
    # row slicing. This is needed because they are not needed in some computations.
    # ⎡M_u⎤ ⎧Ṫ_u⎫ ⎡L_uu⎤ ⎧T_u⎫ = ⎧q̇_u⎫
    # ⎣M_s⎦ ⎩Ṫ_s⎭ ⎣L_su⎦ ⎩T_s⎭   ⎩q̇_s⎭

    surf_to_marg_idxs = []
    # surf_idxs = []
    # marg_idxs = []
    surf_areas = []
    for surf in lp_link_surfs:
        surf_to_marg_idxs.append(([], []))
        # surf_idxs.append([])
        # marg_idxs.append([])
        surf_areas.append([])
        for face in surf.faces:
            face_marg_idxs, face_idxs = body_marg_idxs[face], elem_idxs[face]
            # surf_idxs[-1].extend(face_idxs)
            # marg_idxs[-1].extend(face_marg_idxs)
            # gera_idxs = (hstack((face_marg_idxs, face_idxs)),
            #              hstack((face_idxs, face_marg_idxs)))
            surf_to_marg_idxs[-1][0].extend(face_idxs)
            surf_to_marg_idxs[-1][1].extend(face_marg_idxs)
            face_node_geras = face.node_geras()
            if face_node_geras[X] is not None:
                lpm_gera[X].aix[(face_idxs, face_marg_idxs)] = face_node_geras[X]
            elif face_node_geras[Y] is not None:
                lpm_gera[Y].aix[(face_idxs, face_marg_idxs)] = face_node_geras[Y]
            elif face_node_geras[Z] is not None:
                lpm_gera[Z].aix[(face_idxs, face_marg_idxs)] = face_node_geras[Z]
            surf_areas[-1].extend(face.node_areas())

    # Build conductance matrix for x, y and z direction:
    # Negate conductance values and negative sum of rows on diagonal
    # Placing the sum of the rows on the diagonal is only done for the inner
    # conductances because the margin conductances are added to the film coefficient
    # for temperature field computation.

    run_geom_ratio_x = lpm_gera[X].to_coo().tocsr()
    run_geom_ratio_x.data[:] = -run_geom_ratio_x.data
    row_sums = run_geom_ratio_x[:num_body_nodes].sum(axis=1).A.ravel()
    run_geom_ratio_x -= diags(row_sums, shape=run_geom_ratio_x.shape)

    run_geom_ratio_y = lpm_gera[Y].to_coo().tocsr()
    run_geom_ratio_y.data[:] = -run_geom_ratio_y.data
    row_sums = run_geom_ratio_y[:num_body_nodes].sum(axis=1).A.ravel()
    run_geom_ratio_y -= diags(row_sums, shape=run_geom_ratio_y.shape)

    run_geom_ratio_z = lpm_gera[Z].to_coo().tocsr()
    run_geom_ratio_z.data[:] = -run_geom_ratio_z.data
    row_sums = run_geom_ratio_z[:num_body_nodes].sum(axis=1).A.ravel()
    run_geom_ratio_z -= diags(row_sums, shape=run_geom_ratio_z.shape)

    # Surfaces per margin node for mean temperature computation
    # Notice: some margin nodes have multiple connected surfaces whose areas are added up
    Qs = zeros((num_body_nodes, len(lp_link_surfs)))  # Node surface areas
    for i, (smidxs, areas) in enumerate(zip(surf_to_marg_idxs, surf_areas)):
        # Qs[idxs, i] += areas
        marg_idxs = smidxs[1]
        add.at(Qs[:, i], marg_idxs, areas)

    return (
        array(vols),
        Qs,
        run_geom_ratio_x,
        run_geom_ratio_y,
        run_geom_ratio_z,
        array(posns),
        surf_names,
        [
            (array(idxs0, dtype=int64), array(idxs1, dtype=int64))
            for idxs0, idxs1 in surf_to_marg_idxs
        ],
        [(array(areas)) for areas in surf_areas],
        num_body_nodes,
        vol_sum,
    )


if __name__ == '__main__':
    from numpy.random import random_sample
    import pyvista as pv
    from thermca.plot.primitives import axes

    ann_xposn = 0.5
    ann_yposn = 0.5
    inner_rad = 0.3
    outer_rad = 0.4

    plotter = pv.Plotter()
    plotter.add_mesh(
        pv.Disc(
            (ann_xposn, ann_yposn, 0.0),
            inner_rad,
            outer_rad,
            (0.0, 0.0, 1.0),
            c_res=80,
        ),
        opacity=0.1,
        show_edges=True,
        color=(0.0, 1.0, 0.0),
    )

    rand = lambda a, b: (b - a) * random_sample((10,)) + a
    for rect_xposn, rect_yposn, rect_width, rect_hgt in zip(
        rand(0, 1),
        rand(0, 1),
        rand(0, 1),
        rand(0, 1),
    ):
        color = (0.0, 0.0, 1.0)

        if circ_rect_overlaps(
            ann_xposn,
            ann_yposn,
            outer_rad,
            rect_xposn,
            rect_yposn,
            rect_width,
            rect_hgt,
        ):
            color = (1.0, 0.0, 0.0)
            # print(f"yes {cx1=}, {cy1=}, {cir1=}, {cor1=}")

        if circ_contain_rect(
            ann_xposn,
            ann_yposn,
            inner_rad,
            rect_xposn,
            rect_yposn,
            rect_width,
            rect_hgt,
        ):
            color = (1.0, 1.0, 0.0)
            # print(f"no  {cx1=}, {cy1=}, {cir1=}, {cor1=}")
        pointa = [rect_xposn + rect_width, rect_yposn, 0.0]
        pointb = [rect_xposn + rect_width, rect_yposn + rect_hgt, 0.0]
        pointc = [rect_xposn, rect_yposn + rect_hgt, 0.0]
        pointd = [rect_xposn, rect_yposn, 0.0]
        plotter.add_mesh(
            pv.Rectangle([pointa, pointb, pointc, pointd]),
            opacity=0.1,
            show_edges=True,
            color=color,
        )
        print(f"{color}  {rect_xposn=}, {rect_yposn=}, {rect_hgt=}, {rect_width=}")
    axes((1.0, 1.0, 1.0), 20, True, plotter)
    plotter.show()
