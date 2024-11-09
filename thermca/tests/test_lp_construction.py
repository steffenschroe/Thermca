from itertools import product

from thermca.lpm.asm import Asm
from thermca.lpm.cube import Cube
from thermca.lpm.cyl import Cyl
from thermca.lpm.asm import Surf, ForceConts
from thermca.lpm.lp_construction import (
    create_lp_elems, find_body_contacts, add_force_cont_pairs_to_grouped_conts
)


def sorted_contact_names(conts):
    return sorted([
        tuple(sorted(
            [f"{cont[0].body.name}.{cont[0].name}", f"{cont[1].body.name}.{cont[1].name}"]
        ))
        for cont in conts
    ])


def test_find_body_contacts():
    """Build a Rubik's Cube and test, if all inner
    face contacts are found exactly once."""

    # # Test block to block contacts

    with Asm() as asm:
        for posn, name in zip(
                product([-1., 0., 1], [-1., 0., 1], [-1., 0., 1]),
                product(['left', 'mid', 'rgt'], ['lwr', 'mid', 'upr'], ['back', 'mid', 'fnt'])
        ):
            Cube(posn=posn, name='_'.join(name))

    # Create lumped parameter objects from geometry objects
    lp_blocks, lp_cyls, lp_link_surfs, force_cont_pairs = create_lp_elems(asm)
    block_block_conts, cyl_cyl_conts, block_cyl_conts = find_body_contacts(lp_blocks, [])

    contact_names = sorted_contact_names(block_block_conts)

    expected_contacts = [
        ('left_lwr_back.front', 'left_lwr_mid.back'),
        ('left_lwr_back.top', 'left_mid_back.btm'),
        ('left_lwr_back.right', 'mid_lwr_back.left'),
        ('left_lwr_mid.front', 'left_lwr_fnt.back'),
        ('left_lwr_mid.top', 'left_mid_mid.btm'),
        ('left_lwr_mid.right', 'mid_lwr_mid.left'),
        ('left_lwr_fnt.top', 'left_mid_fnt.btm'),
        ('left_lwr_fnt.right', 'mid_lwr_fnt.left'),
        ('left_mid_back.front', 'left_mid_mid.back'),
        ('left_mid_back.top', 'left_upr_back.btm'),
        ('left_mid_back.right', 'mid_mid_back.left'),
        ('left_mid_mid.front', 'left_mid_fnt.back'),
        ('left_mid_mid.top', 'left_upr_mid.btm'),
        ('left_mid_mid.right', 'mid_mid_mid.left'),
        ('left_mid_fnt.top', 'left_upr_fnt.btm'),
        ('left_mid_fnt.right', 'mid_mid_fnt.left'),
        ('left_upr_back.front', 'left_upr_mid.back'),
        ('left_upr_back.right', 'mid_upr_back.left'),
        ('left_upr_mid.front', 'left_upr_fnt.back'),
        ('left_upr_mid.right', 'mid_upr_mid.left'),
        ('left_upr_fnt.right', 'mid_upr_fnt.left'),
        ('mid_lwr_back.front', 'mid_lwr_mid.back'),
        ('mid_lwr_back.top', 'mid_mid_back.btm'),
        ('mid_lwr_back.right', 'rgt_lwr_back.left'),
        ('mid_lwr_mid.front', 'mid_lwr_fnt.back'),
        ('mid_lwr_mid.top', 'mid_mid_mid.btm'),
        ('mid_lwr_mid.right', 'rgt_lwr_mid.left'),
        ('mid_lwr_fnt.top', 'mid_mid_fnt.btm'),
        ('mid_lwr_fnt.right', 'rgt_lwr_fnt.left'),
        ('mid_mid_back.front', 'mid_mid_mid.back'),
        ('mid_mid_back.top', 'mid_upr_back.btm'),
        ('mid_mid_back.right', 'rgt_mid_back.left'),
        ('mid_mid_mid.front', 'mid_mid_fnt.back'),
        ('mid_mid_mid.top', 'mid_upr_mid.btm'),
        ('mid_mid_mid.right', 'rgt_mid_mid.left'),
        ('mid_mid_fnt.top', 'mid_upr_fnt.btm'),
        ('mid_mid_fnt.right', 'rgt_mid_fnt.left'),
        ('mid_upr_back.front', 'mid_upr_mid.back'),
        ('mid_upr_back.right', 'rgt_upr_back.left'),
        ('mid_upr_mid.front', 'mid_upr_fnt.back'),
        ('mid_upr_mid.right', 'rgt_upr_mid.left'),
        ('mid_upr_fnt.right', 'rgt_upr_fnt.left'),
        ('rgt_lwr_back.front', 'rgt_lwr_mid.back'),
        ('rgt_lwr_back.top', 'rgt_mid_back.btm'),
        ('rgt_lwr_mid.front', 'rgt_lwr_fnt.back'),
        ('rgt_lwr_mid.top', 'rgt_mid_mid.btm'),
        ('rgt_lwr_fnt.top', 'rgt_mid_fnt.btm'),
        ('rgt_mid_back.front', 'rgt_mid_mid.back'),
        ('rgt_mid_back.top', 'rgt_upr_back.btm'),
        ('rgt_mid_mid.front', 'rgt_mid_fnt.back'),
        ('rgt_mid_mid.top', 'rgt_upr_mid.btm'),
        ('rgt_mid_fnt.top', 'rgt_upr_fnt.btm'),
        ('rgt_upr_back.front', 'rgt_upr_mid.back'),
        ('rgt_upr_mid.front', 'rgt_upr_fnt.back'),
    ]
    expected_contacts = sorted([tuple(sorted(cont)) for cont in expected_contacts])
    assert len(contact_names) == len(set(contact_names))  # Check for double entries
    assert contact_names == expected_contacts
    # for cont in contacts:
    #     print(f"('{cont[0].body.name}.{cont[0].name}', '{cont[1].body.name}.{cont[1].name}'),")
    # from thermca.plot.asm import asm as plot_asm
    # plot_asm(asm, contacts, dpi=200).show()

    # # Test cylinder to cylinder contacts

    with Asm() as asm:
        Cyl('left_out', posn=(-1., 0., 0.), inner_rad=3., outer_rad=4., lgth=1.)
        Cyl('left_mid', posn=(-1., 0., 0.), inner_rad=2., outer_rad=3., lgth=1.)
        Cyl('left_inr', posn=(-1., 0., 0.), inner_rad=1., outer_rad=2., lgth=1.)
        Cyl('mid_out', posn=(0., 0., 0.), inner_rad=3., outer_rad=4., lgth=1.)
        Cyl('mid_mid', posn=(0., 0., 0.), inner_rad=2., outer_rad=3., lgth=1.)
        Cyl('mid_inr', posn=(0., 0., 0.), inner_rad=1., outer_rad=2., lgth=1.)
        Cyl('rgt_out', posn=(1., 0., 0.), inner_rad=3., outer_rad=4., lgth=1.)
        Cyl('rgt_mid', posn=(1., 0., 0.), inner_rad=2., outer_rad=3., lgth=1.)
        Cyl('rgt_inr', posn=(1., 0., 0.), inner_rad=1., outer_rad=2., lgth=1.)
        # Create lumped parameter objects from geometry objects

    lp_blocks, lp_cyls, lp_link_surfs, force_cont_pairs = create_lp_elems(asm)

    block_block_conts, cyl_cyl_conts, block_cyl_conts = find_body_contacts([], lp_cyls)

    contact_names = sorted_contact_names(cyl_cyl_conts)

    expected_contacts = [
        ('left_inr.end', 'mid_inr.base'),
        ('left_inr.outer', 'left_mid.inner'),
        ('left_mid.end', 'mid_mid.base'),
        ('left_mid.outer', 'left_out.inner'),
        ('left_out.end', 'mid_out.base'),
        ('mid_inr.end', 'rgt_inr.base'),
        ('mid_inr.outer', 'mid_mid.inner'),
        ('mid_mid.end', 'rgt_mid.base'),
        ('mid_mid.outer', 'mid_out.inner'),
        ('mid_out.end', 'rgt_out.base'),
        ('rgt_inr.outer', 'rgt_mid.inner'),
        ('rgt_mid.outer', 'rgt_out.inner')
    ]

    # print(contact_names)
    assert len(contact_names) == len(set(contact_names))  # Check for double entries
    assert contact_names == expected_contacts

    # # Test block to cylinder contacts

    with Asm() as asm:
        Cyl('inr_cyl', posn=(0., 0., 0.), inner_rad=0., outer_rad=1., lgth=1.)
        Cyl('out_cyl', posn=(0., 0., 0.), inner_rad=1., outer_rad=2., lgth=1.)
        width2 = 1./(2.**.5)
        Cube('left_inr', posn=(-1., -width2, -width2), width=1., hgt=width2 * 2, depth=width2 * 2)
        Cube('left_out', posn=(-1., 1., -.5), width=1., hgt=1., depth=1.)
        Cube('left_out', posn=(-1., 2., -.5), width=1., hgt=1., depth=1.)
        Cube('right_inr', posn=(1., -width2, -width2), width=1., hgt=width2 * 2, depth=width2 * 2)
        Cube('right_out', posn=(1., 1., -.5), width=1., hgt=1., depth=1.)
        Cube('right_out', posn=(1., 2., -.5), width=1., hgt=1., depth=1.)

    lp_blocks, lp_cyls, lp_link_surfs, force_cont_pairs = create_lp_elems(asm)
    block_block_conts, cyl_cyl_conts, block_cyl_conts = find_body_contacts(lp_blocks, lp_cyls)

    contact_names = sorted_contact_names(block_cyl_conts)
    # print(contact_names)
    expected_contacts = [
        ('inr_cyl.base', 'left_inr.right'),
        ('inr_cyl.end', 'right_inr.left'),
        ('left_out.right', 'out_cyl.base'),
        ('out_cyl.end', 'right_out.left')
    ]
    assert len(contact_names) == len(set(contact_names))  # Check for double entries
    assert contact_names == expected_contacts

    # # Test grouping of forced contact pairs

    with Asm() as asm:
        bl = Cube('block_left', posn=(-1., 0., 0.))
        br = Cube('block_right', posn=(0., 0., 0.))
        cl = Cyl('cyl_left',  posn=(1., 0., 0.), inner_rad=.1)
        cr = Cyl('cyl_right', posn=(2., 0., 0.), inner_rad=.1)
        ForceConts([(bl.face.btm, br.face.btm)])
        ForceConts([(br.face.top, cl.face.outer)])
        ForceConts([(cl.face.inner, cr.face.inner)])

    lp_blocks, lp_cyls, lp_link_surfs, force_cont_pairs = create_lp_elems(asm)

    block_block_conts, cyl_cyl_conts, block_cyl_conts = (
        find_body_contacts(lp_blocks, lp_cyls)
    )

    block_block_conts, cyl_cyl_conts, block_cyl_conts = add_force_cont_pairs_to_grouped_conts(
        block_block_conts,
        cyl_cyl_conts,
        block_cyl_conts,
        force_cont_pairs,
    )
    block_block_names = sorted_contact_names(block_block_conts)
    # print(block_block_names)
    assert block_block_names, [('block_left.btm', 'block_right.btm'), ('block_left.right', 'block_right.left')]
    cyl_cyl_names = sorted_contact_names(cyl_cyl_conts)
    # print(cyl_cyl_names)
    assert cyl_cyl_names, [('cyl_left.end', 'cyl_right.base'), ('cyl_left.inner', 'cyl_right.inner')]
    block_cyl_names = sorted_contact_names(block_cyl_conts)
    # print(block_cyl_names)
    assert block_cyl_names, [('block_right.right', 'cyl_left.base'), ('block_right.top', 'cyl_left.outer')]

    # from thermca.plot.asm import asm as plot_asm
    # plot_asm(asm, block_block_conts + cyl_cyl_conts + block_cyl_conts, dpi=200).show()

