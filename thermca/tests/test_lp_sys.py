"""Test stationary solutions of lumped parameter systems"""

from math import tau
from numpy import sum as asum, allclose, log

from thermca.lpm.asm import Asm
from thermca.lpm.cube import Cube
from thermca.lpm.cyl import Cyl
from thermca.lpm.asm import Surf, ForceConts
from thermca.lpm.lp_system import LPSystem
from thermca.materials import Solid
from thermca.static_bcs import TempBC, HeatBC, FilmBC, FluxBC


def test_conductance_on_two_horizontal_assembled_blocks():
    """Test stationary solution on two horizontal assembled blocks
    with temperature boundary conditions on left and right surface"""
    with Asm() as asm:
        left_block = Cube(
            'left_block',
            posn=(0., 0., 0.),
            width=2.,
            width_div=3,
            hgt=3.,
            hgt_div=2,
            depth=4.,
            depth_div=3,
        )
        right_block = Cube(
            'right_block',
            posn=(left_block.width, 0., 0.),
            width=2.,
            width_div=3,
            hgt=3.,
            hgt_div=2,
            depth=4.,
            depth_div=3,
        )
        left = Surf(name='left', faces=[left_block.face.left])
        right = Surf(name='right', faces=[right_block.face.right])

    test_matl = Solid(condy=1.5, dens=1., spec_heat=1.)

    part = LPSystem(asm, 0., test_matl)
    result = part.solve([TempBC(left, 0.), TempBC(right, 1.)])
    # Conductance: L = Q / ΔT, ΔT = 1.
    measured_part_cond_left_to_right = -result[left].heat()
    part_cond_left_to_right = (
        left_block.hgt
        * left_block.depth
        * test_matl.condy_interp(0.)
        / (left_block.width + right_block.width)
    )
    assert allclose(
        [part_cond_left_to_right], [measured_part_cond_left_to_right], rtol=0, atol=1e-6
    )


def test_conductance_on_two_blocks_assembled_one_behind_the_other():
    """Test stationary solution on two blocks assembled one behind the other
    with temperature boundary conditions on back and front surface"""
    with Asm() as asm:
        back_block = Cube(
            'back_block',
            posn=(0., 0., 0.),
            width=2.,
            width_div=3,
            hgt=3.,
            hgt_div=2,
            depth=4.,
            depth_div=3,
        )
        front_block = Cube(
            'front_block',
            posn=(0., 0., back_block.depth),
            width=2.,
            width_div=3,
            hgt=3.,
            hgt_div=2,
            depth=4.,
            depth_div=3,
        )
        back = Surf(name='back', faces=[back_block.face.back])
        front = Surf(name='front', faces=[front_block.face.front])

    test_matl = Solid(condy=1.5, dens=1., spec_heat=1.)

    part = LPSystem(asm, 0., test_matl)
    result = part.solve([TempBC(back, 0.), TempBC(front, 1.)])
    # Conductance: L = Q / ΔT, ΔT = 1.
    measured_part_cond_back_to_front = -result[back].heat()
    part_cond_back_to_front = (
        back_block.width
        * back_block.hgt
        * test_matl.condy_interp(0.)
        / (back_block.depth + front_block.depth)
    )
    assert allclose(
        [part_cond_back_to_front], [measured_part_cond_back_to_front], rtol=0, atol=1e-6
    )


def test_conductance_on_two_blocks_assembled_on_top_of_each_other():
    """Test stationary solution on two blocks assembled on top of each other
    with temperature boundary conditions on bottom and top surface"""
    with Asm() as asm:
        btm_block = Cube(
            'btm_block',
            posn=(0., 0., 0.),
            width=2.,
            width_div=3,
            hgt=3.,
            hgt_div=2,
            depth=4.,
            depth_div=3,
        )
        top_block = Cube(
            'top_block',
            posn=(0., btm_block.hgt, 0.),
            width=2.,
            width_div=3,
            hgt=3.,
            hgt_div=2,
            depth=4.,
            depth_div=3,
        )
        btm = Surf(name='btm', faces=[btm_block.face.btm])
        top = Surf(name='top', faces=[top_block.face.top])

    test_matl = Solid(condy=1.5, dens=1., spec_heat=1.)

    part = LPSystem(asm, 0., test_matl)
    result = part.solve([TempBC(btm, 0.), TempBC(top, 1.)])
    # Conductance: L = Q / ΔT, ΔT = 1.
    measured_part_cond_btm_to_top = -result[btm].heat()
    part_cond_btm_to_top = (
        btm_block.width
        * btm_block.depth
        * test_matl.condy_interp(0.)
        / (btm_block.hgt + top_block.hgt)
    )
    assert allclose(
        [part_cond_btm_to_top], [measured_part_cond_btm_to_top], rtol=0, atol=1e-6
    )


def test_conductance_on_two_horizontal_assembled_cylinders():
    """Test stationary solution on two horizontal assembled cylinders
    with temperature boundary conditions on left and right surface"""
    with Asm() as asm:
        left_cyl = Cyl(
            'left_cyl',
            posn=(0., 0., 0.),
            lgth=2.,
            lgth_div=3,
            inner_rad=.5,
            outer_rad=1.,
            rad_div=2,
        )
        right_cyl = Cyl(
            'right_cyl',
            posn=(left_cyl.lgth, 0., 0.),
            lgth=2.,
            lgth_div=3,
            inner_rad=.5,
            outer_rad=1.,
            rad_div=2,
        )
        left = Surf(name='left', faces=[left_cyl.face.base])
        right = Surf(name='right', faces=[right_cyl.face.end])

    test_matl = Solid(condy=1.5, dens=1., spec_heat=1.)

    part = LPSystem(asm, 0., test_matl)
    result = part.solve([TempBC(left, 0.), TempBC(right, 1.)])
    # Conductance: L = Q / ΔT, ΔT = 1.
    measured_part_cond_left_to_right = -result[left].heat()
    part_cond_left_to_right = (
        tau
        / 2.
        * (left_cyl.outer_rad**2. - left_cyl.inner_rad**2)
        * test_matl.condy_interp(0.)
        / (left_cyl.lgth + right_cyl.lgth)
    )
    # print(part_cond_left_to_right, measured_part_cond_left_to_right)
    assert allclose(
        [part_cond_left_to_right], [measured_part_cond_left_to_right], rtol=0, atol=1e-6
    )


def test_conductance_on_two_coaxial_assembled_cylinders():
    """Test stationary solution on two coaxial assembled cylinders
    with temperature boundary conditions on inner and outer surface"""
    with Asm() as asm:
        inner_cyl = Cyl(
            'inner_cyl',
            posn=(0., 0., 0.),
            lgth=2.,
            lgth_div=3,
            inner_rad=.5,
            outer_rad=1.,
            rad_div=2,
        )
        outer_cyl = Cyl(
            'outer_cyl',
            posn=(0., 0., 0.),
            lgth=2.,
            lgth_div=3,
            inner_rad=1.,
            outer_rad=1.5,
            rad_div=2,
        )
        inner = Surf(name='inner', faces=[inner_cyl.face.inner])
        outer = Surf(name='outer', faces=[outer_cyl.face.outer])

    test_matl = Solid(condy=1.5, dens=1., spec_heat=1.)

    part = LPSystem(asm, 0., test_matl)
    result = part.solve([TempBC(inner, 0.), TempBC(outer, 1.)])
    # Conductance: L = Q / ΔT, ΔT = 1.
    measured_part_cond_inner_to_outer = -result[inner].heat()
    part_cond_inner_to_outer = (
        test_matl.condy_interp(0.)
        * tau
        * inner_cyl.lgth
        / log(outer_cyl.outer_rad / inner_cyl.inner_rad)
    )
    # print(part_cond_left_to_right, measured_part_cond_left_to_right)
    assert allclose(
        [part_cond_inner_to_outer],
        [measured_part_cond_inner_to_outer],
        rtol=0,
        atol=1e-6,
    )


def test_asymmetrical_heat_flow_on_two_blocks():
    """Test 2d conductance through blocks in stationary solution.

    Two merged blocks on top of each other with equal height form a square in
    the x-y-plane. Temperature boundary conditions are applied on the
    left and the right of the square. The left is full bound while the
    right is only bound on the lower block. The stationary temperature
    field is used to calculate the thermal conductance of this
    constricted heat flow. It is compared to a reference conductance
    based on a high resolution model.
    """
    spatial_div = 8
    test_matl = Solid(condy=1., dens=1., spec_heat=1.)
    with Asm() as asm:
        lower = Cube(
            width=1.,
            hgt=.5,
            width_div=spatial_div,
            hgt_div=spatial_div // 2,
        )
        upper = Cube(
            posn=(0, .5, 0.),
            width=1.,
            hgt=.5,
            width_div=spatial_div,
            hgt_div=(spatial_div // 2),
        )
        left = Surf(name='left', faces=[lower.face.left, upper.face.left])
        right = Surf(name='right', faces=[lower.face.right])
    part = LPSystem(asm, 0., test_matl)
    result = part.solve([TempBC(left, 0.), TempBC(right, 1.)])
    # Conductance: L = Q / ΔT, ΔT = 1.
    measured_cond = -result[left].heat()
    ref_cond = .81927  # Measured at spatial resolution of 640x640
    # print(f'spatial resolution: {spatial_div}x{spatial_div}')
    # print('calculated conductance:   ', measured_cond)
    # print('reference conductance:   ', ref_cond)
    # at spatial resolution of 32x32 the deviation should be lower than 1%
    # at spatial resolution of 8x8 the deviation should be lower than 4%
    assert allclose(measured_cond, ref_cond, rtol=.04, atol=0.), (
        'The simulated conductance is not close enough to the ' 'expected result!'
    )


def test_asymmetrical_heat_flow_on_two_cylinders():
    """Test heat flow through hollow cylinder assembly in steady state.

    First: heat flow in axial direction starting from the entire base
    surface as 'inlet' to the inner region of the end surface as
    'outlet'

    Second: heat flow in radial direction starting from the entire outer
    surface as 'inlet' to the end region of the inner surface as
    'outlet'

    Cylinder properties: inner radius .5, outer radius 1., length 1.,
    full base surface, end surface from inner radius to .75, full outer
    surface, inner surface from .5 to end length
    """
    spatial_div = 16
    test_matl = Solid(condy=1., dens=1., spec_heat=1.)

    with Asm() as asm:
        inner = Cyl(
            lgth=1.,
            inner_rad=.5,
            outer_rad=.75,
            rad_div=spatial_div // 2,
            lgth_div=spatial_div,
        )
        outer = Cyl(
            lgth=1.,
            inner_rad=.75,
            outer_rad=1.,
            rad_div=spatial_div // 2,
            lgth_div=spatial_div,
        )
        base = Surf(name='base', faces=[inner.face.base, outer.face.base])
        end = Surf(name='end', faces=[inner.face.end])
    part = LPSystem(asm, 0., test_matl)
    result = part.solve([TempBC(base, 0.), TempBC(end, 1.)])
    # result.plot(dpi=250, hide=()).show()
    # Conductance: L = Q / ΔT, ΔT = 1.
    measured_cond = result[end].heat()
    ref_cond = 2.0455  # at 512x512 spatial subdivision
    # print(f'spatial resolution: {spatial_div}x{spatial_div}')
    # print('calculated conductance:   ', measured_cond)
    # print('reference conductance:   ', ref_cond)
    assert allclose(measured_cond, ref_cond, rtol=.025, atol=0.), (
        'The simulated conductance is not close enough to the ' 'expected result!'
    )

    with Asm() as asm:
        left = Cyl(
            lgth=.5,
            inner_rad=.5,
            outer_rad=1.,
            rad_div=spatial_div,
            lgth_div=spatial_div // 2,
        )
        right = Cyl(
            posn=(.5, 0., 0.),
            lgth=.5,
            inner_rad=.5,
            outer_rad=1.,
            rad_div=spatial_div,
            lgth_div=spatial_div // 2,
        )
        outer = Surf(name='outer', faces=[left.face.outer, right.face.outer])
        half_inner = Surf(name='half_inner', faces=[right.face.inner])
    part = LPSystem(asm, 0., test_matl)
    result = part.solve([TempBC(outer, 0.), TempBC(half_inner, 1.)])
    # result.plot(dpi=250, hide=()).show()
    # Conductance: L = Q / ΔT, ΔT = 1.
    measured_cond = result[outer].heat()
    ref_cond = -6.0733  # at 512x512 spatial subdivision
    # print(f'spatial resolution: {spatial_div}x{spatial_div}')
    # print('calculated conductance:   ', measured_cond)
    # print('reference conductance:   ', ref_cond)
    assert allclose(measured_cond, ref_cond, rtol=.026, atol=0.), (
        'The simulated conductance is not close enough to the ' 'expected result!'
    )


def test_conductance_on_two_horizontal_assembled_blocks_with_heat_bc():
    """Test stationary solution on two horizontal assembled blocks
    with heat and temperature boundary conditions on left and right surface"""

    with Asm() as asm:
        left_block = Cube(
            'left_block',
            posn=(0., 0., 0.),
            width=2.,
            width_div=150,
            hgt=3.,
            hgt_div=2,
            depth=4.,
            depth_div=3,
        )
        right_block = Cube(
            'right_block',
            posn=(left_block.width, 0., 0.),
            width=2.,
            width_div=150,
            hgt=3.,
            hgt_div=2,
            depth=4.,
            depth_div=3,
        )
        left = Surf(name='left', faces=[left_block.face.left])
        right = Surf(name='right', faces=[right_block.face.right])

    test_matl = Solid(condy=1.5, dens=1., spec_heat=1.)

    part = LPSystem(asm, 0., test_matl)
    result = part.solve([HeatBC(left, 1.), TempBC(right, 0.)])
    # Mean temp. from margin nodes for conductance: Q = ΔT * L -> L = Q / ΔT
    measured_part_cond_left_to_right = 1. / result[left].temp()
    part_cond_left_to_right = (
        left_block.hgt
        * left_block.depth
        * test_matl.condy_interp(0.)
        / (left_block.width + right_block.width)
    )
    # print(part_cond_left_to_right, measured_part_cond_left_to_right)
    # Relaxed test of accuracy because temperature gets not evaluated at the surface
    # but at the margin nodes instead.
    assert allclose(
        [part_cond_left_to_right], [measured_part_cond_left_to_right], rtol=0, atol=1e-2
    )


def test_steel_plate_with_film_and_heat_bcs():
    """Test static solution on steel plate with 3d heat flow

    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """

    with Asm() as asm:
        btm_block = Cube(
            'btm_block',
            posn=(0., 0., 0.),
            width=.5,
            width_div=40,
            hgt=.5,
            hgt_div=40,
            depth=.05,
            depth_div=4,
        )
        top_block = Cube(
            'top_block',
            posn=(0., btm_block.hgt, 0.),
            width=.5,
            width_div=40,
            hgt=.5,
            hgt_div=40,
            depth=.05,
            depth_div=4,
        )
        btm = Surf(name='btm', faces=[btm_block.face.btm])
        bb = btm_block.face
        bt = top_block.face
        air = Surf(
            name='air',
            faces=[
                bb.left,
                bb.right,
                bb.back,
                bb.front,
                bt.left,
                bt.right,
                bt.back,
                bt.front,
                bt.top,
            ],
        )

    steel = Solid(
        dens=8000.,
        spec_heat=500.,
        condy=50.,
    )

    cuboid = LPSystem(asm, 20., steel)
    bcs = [HeatBC(btm, heat=1000.), FilmBC(air, film=10., env_temp=20.)]
    result = cuboid.solve(bcs)
    mean_surf_temps = [result[btm].temp(), result[air].temp()]
    # print(mean_surf_temps)
    # Relaxed accuracy because temperatures are not measured on surface but on margin nodes
    assert allclose([291.1152, 108.8888], mean_surf_temps, rtol=.02, atol=0.)


def test_conductance_compression_on_surfaces_over_edges():
    """Test stationary solution on block with temperature and film
    boundary conditions on surfaces spanning over edges. In this case
    nodes have two conductances from the margin node to the virtual
    surface node. This is handled as a special case for film and
    temperature boundary conditions."""
    with Asm() as asm:
        block = Cube(
            'block',
            posn=(0., 0., 0.),
            width=3.,
            width_div=1,
            hgt=2.,
            hgt_div=1,
            depth=1.,
            depth_div=1,
        )
        left = Surf(name='left', faces=[block.face.left, block.face.top])
        right = Surf(name='right', faces=[block.face.right, block.face.btm])

    test_matl = Solid(condy=1., dens=1., spec_heat=1.)

    part = LPSystem(asm, 0., test_matl)
    # Heat from margin nodes to surface: Q = ΔT * L
    cond_left_left = block.depth * block.hgt / (block.width / 2.)  # 1.33
    cond_left_up = block.width * block.depth / (block.hgt / 2.)  # 3
    cond_left_surf = cond_left_left + cond_left_up  # 4.33
    cond_right_right = 1. / (
        1. / cond_left_left + 1. / (block.depth * block.hgt * 1.)
    )  # .8
    cond_right_down = 1. / (
        1. / cond_left_up + 1. / (block.width * block.depth * 1.)
    )  # 1.5
    cond_right_surf = cond_right_right + cond_right_down  # 2.3
    part_cond_left_to_right = 1 / (1 / cond_left_surf + 1 / cond_right_surf)  # 1.5
    result = part.solve([TempBC(left, 0.), FilmBC(right, 1., 1.)])
    measured_part_cond_left_to_right0 = -result[left].heat()
    measured_part_cond_left_to_right1 = result[right].heat()
    # print(f"{part_cond_left_to_right=} {measured_part_cond_left_to_right0=} {measured_part_cond_left_to_right1=}")
    assert allclose(
        [measured_part_cond_left_to_right0],
        part_cond_left_to_right,
        rtol=0.,
        atol=1e-6,
    )
    assert allclose(
        [measured_part_cond_left_to_right1],
        part_cond_left_to_right,
        rtol=0.,
        atol=1e-6,
    )
