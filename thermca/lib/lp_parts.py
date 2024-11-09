"""Functions to create basic lumped parameter parts"""

from thermca.lpm.asm import Asm
from thermca.lpm.lp_part import LPPart
from thermca.lpm.cyl import Cyl, EQUAL_VOL
from thermca.lpm.cube import Cube
from thermca.lpm.asm import Surf


def block(
    matl,
    init_temp: float = 0.,
    width: float = 1.,
    hgt: float = 1.,
    depth: float = 1.,
    width_div: int = 1,
    hgt_div: int = 1,
    depth_div: int = 1,
    posn: tuple[float, float, float] = (0., 0., 0.),
    name: str = '',
):
    """Creates a block with surfaces 'left', 'right', 'btm', 'top', 'back, 'front'."""
    with Asm() as asm:
        cube_rod = Cube(
            width=width,
            hgt=hgt,
            depth=depth,
            width_div=width_div,
            hgt_div=hgt_div,
            depth_div=depth_div,
            name=name,
        )
        for face in cube_rod.face:
            Surf(name=face.name, faces=[face])
    block_part = LPPart(
        asm=asm,
        matl=matl,
        init_temp=init_temp,
        posn=posn,
        name='block',
    )
    return block_part


def pipe(
        matl,
        init_temp: float = 0.,
        lgth: float = 1.,
        inner_rad: float = .2,
        outer_rad: float = .25,
        lgth_div: int = 1,
        rad_div: int = 1,
        posn: tuple[float, float, float] = (0., 0., 0.),
        rad_div_mode: str = EQUAL_VOL,
        name: str = ''
):
    """Creates a pipe with surfaces 'base', 'end', 'outer', 'inner'."""
    with Asm() as asm:
        cyl_rod = Cyl(
            lgth=lgth,
            inner_rad=inner_rad,
            outer_rad=outer_rad,
            lgth_div=lgth_div,
            rad_div=rad_div,
            posn=posn,
            rad_div_mode=rad_div_mode,
            name=name,
        )
        Surf(name='base', faces=[cyl_rod.face.base])
        Surf(name='end', faces=[cyl_rod.face.end])
        Surf(name='outer', faces=[cyl_rod.face.outer])
        if inner_rad > 0.:
            Surf(name='inner', faces=[cyl_rod.face.inner])

    rod_part = LPPart(
        asm=asm,
        matl=matl,
        init_temp=init_temp,
        name='rod',
    )
    return rod_part


def rod(
    matl,
    init_temp: float = 0.,
    lgth: float = 1.,
    outer_rad: float = .25,
    lgth_div: int = 1,
    rad_div: int = 1,
    posn: tuple[float, float, float] = (0., 0., 0.),
    rad_div_mode: str = EQUAL_VOL,
    name: str = ''
):
    """Creates a round rod with surfaces 'base', 'end', 'outer'."""
    return pipe(
        matl=matl,
        init_temp=init_temp,
        lgth=lgth,
        inner_rad=0.,
        outer_rad=outer_rad,
        lgth_div=lgth_div,
        rad_div=rad_div,
        posn=posn,
        rad_div_mode=rad_div_mode,
        name=name,
    )





