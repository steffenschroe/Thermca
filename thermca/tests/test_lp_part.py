"""Test transient solutions of lumped parameter systems"""

import math

import pytest
from numpy import (
    array, sin, cos, pi, zeros, exp, set_printoptions, max as amax, abs as aabs, allclose,
    genfromtxt
)

from thermca.lib import lp_parts
from thermca.lpm.asm import Asm
from thermca.lpm.cube import Cube
from thermca.lpm.cyl import Cyl
from thermca.lpm.asm import Surf
from thermca.materials import Solid
from thermca.model import Model
from thermca.network import Network
from thermca.source import HeatSource
from thermca.lpm.lp_part import LPPart
from thermca.pointnodes import BoundNode, StatNode
from thermca.links import FilmLink
import thermca.lib.lp_parts


def test_1d_heat_flow_in_cyl_and_block_linked_by_film_coef():
    """Test one dimensional heat flow on cyl and block bodies

    The test scenario is a rod with heat flow driven by a temperature
    and a heat boundary condition at each end. The rod has adiabatic
    lateral surfaces.The rod is modeled by a cylinder and a block and
    tested one after each other. The temperature and heat boundary
    conditions are modeled with node elements for higher accuracy. The
    body of the rods are connected to the nodes by film link elements.
    """

    def calculate_pde(rod):
        l = rod.lgth
        a = rod.condy / (rod.dens * rod.spec_heat)  # thermal diffusivity
        la = rod.condy
        tx0 = rod.init_temp  # start temp rod
        q0 = rod.heat_flux
        t1 = rod.end_face_temp
        result = zeros((len(rod.result_times), len(rod.result_lctns)))
        for itime, time in enumerate(rod.result_times):
            for ip, p in enumerate(rod.result_lctns):

                def inf_sum(time, x):
                    sum = 0.
                    for n in range(1, 50):
                        kn = (2. * n - 1.) * pi / 2.
                        fn = (
                            2.
                            / kn**2.
                            * (
                                tx0 * kn * sin(kn)
                                - t1 * kn * sin(kn)
                                + q0 / la * (l * cos(kn) - l)
                            )
                        )
                        sum += fn * exp(-a * kn**2 / l**2 * time) * cos(kn * x / l)
                    return sum

                result[itime, ip] = t1 - q0 / la * (p[0] - l) + inf_sum(time, p[0])
        return result

    def simulate(rod):
        with Asm() as rod_asm:
            if rod.body == 'cyl':
                cyl_rod = Cyl(
                    lgth=rod.lgth,
                    lgth_div=rod.node_count,
                    name='cyl',
                )
                Surf(name='begin', faces=[cyl_rod.face.base])
                Surf(name='end', faces=[cyl_rod.face.end])
            elif rod.body == 'block':
                block_rod = Cube(
                    width=rod.lgth,
                    width_div=rod.node_count,
                    name='block',
                )
                Surf(name='begin', faces=[block_rod.face.left])
                Surf(name='end', faces=[block_rod.face.right])

        test_matl = Solid(
            condy=rod.condy,
            dens=rod.dens,
            spec_heat=rod.spec_heat,
        )

        with Model() as model:
            rod_part = LPPart(
                asm=rod_asm,
                matl=test_matl,
                init_temp=rod.init_temp,
                name='rod',
            )
            # Compensate for absent margin conductance:
            # Heat applied to surfaces of LPParts is fed in nodes located
            # inside the part. The margin conductance will not be
            # considered. For comparison with precise PDE solutions, this
            # is compensated for by a stationary node.
            stat_node = StatNode()
            FilmLink(rod_part.surf.begin, stat_node, film=1e32)
            HeatSource(stat_node, heat=rod.heat_flux * rod_part.surf.begin.area())
            # Boundary temperatures on part surfaces are not possible
            # Use a linked temperature bound node instead
            bound_node = BoundNode(temp=rod.end_face_temp)
            FilmLink(rod_part.surf.end, bound_node, film=1e32)
        result = Network(model).sim(
            time_span=(rod.result_times[0], rod.result_times[-1])
        )
        return result[rod_part].temp(rod.result_times, rod.result_lctns)

    def analyse_results(rod, pde_temps, sim_temps):
        set_printoptions(formatter={'float': '{: .3f}'.format})
        for ip in range(len(rod.result_lctns)):
            print()
            print("Position          : ", rod.result_lctns[ip])
            print("Times             : ", rod.result_times)
            print("Temp. simulation  : ", sim_temps.T[ip])
            print("Temp. PDE-solution: ", pde_temps.T[ip])
            print("Absolute deviation: ", sim_temps.T[ip] - pde_temps.T[ip])
            rel_deviation = (sim_temps.T[ip] - pde_temps.T[ip]) / amax(pde_temps.T[ip])
            print("Relative deviation: ", rel_deviation)
            max_rel_deviation = amax(aabs(rel_deviation))
            print("Max. rel. devitat.: ", max_rel_deviation)
            assert max_rel_deviation < .041, (
                "The simulated temperature has a too " "large relative deviation."
            )

    class Rod:
        body = None
        node_count = 5
        lgth = 1.
        condy = 1.
        dens = 1.
        spec_heat = 1.
        init_temp = 0.
        heat_flux = 1.
        end_face_temp = 0.
        result_lctns = array(
            [(.3, 0., 0.), (.7, 0., 0.)]
        )  # On node locations because of missing interpolation
        result_times = array([.01, .02, .05, .1, .2, .5, 1., 2., 5., 10])

    for body in ['cyl', 'block']:
        Rod.body = body
        print(f"\n\n** Body: {body}")
        print("\n* Initial values\n")
        analyse_results(Rod, calculate_pde(Rod), simulate(Rod))
        print("\n* Doubled conductivity\n")
        Rod.condy = 2.
        analyse_results(Rod, calculate_pde(Rod), simulate(Rod))
        print("\n* Doubled capacity\n")
        Rod.condy = 1.
        Rod.dens = 2.
        analyse_results(Rod, calculate_pde(Rod), simulate(Rod))
        print("\n* Doubled lgth\n")
        Rod.dens = 1.
        Rod.lgth = 2.
        Rod.result_lctns = array([(.6, 0., 0.), (1.4, 0., 0.)])
        analyse_results(Rod, calculate_pde(Rod), simulate(Rod))


def test_2d_heat_flow_on_blocks_connected_by_film_coefs_stationary():
    """Test 2d conductance through assembled blocks in steady state

    Two merged blocks on top of each other with equal height form a
    square in the x-y-plane. Temperature boundary conditions are
    applied on the left and the right of the square. The left is full
    bound while the right is only bound on the lower block. The
    stationary temperature field is used to calculate the thermal
    conductance of this constricted heat flow. It is compared to a
    reference conductance based on a high resolution model.
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
        Surf(name='left', faces=[lower.face.left, upper.face.left])
        Surf(name='right', faces=[lower.face.right])

    with Model() as model:
        block = LPPart(
            asm=asm,
            matl=test_matl,
            init_temp=0.,
            name='block',
        )
        stat_node = StatNode()
        FilmLink(block.surf.left, stat_node, film=1e32)
        HeatSource(stat_node, heat=1.)
        # Boundary temperatures on part surfaces are not possible
        # Use a linked temperature bound node instead
        bound_node = BoundNode(temp=0.)
        FilmLink(block.surf.right, bound_node, film=1e32)
    net = Network(model)
    sim_duration = 5.
    result = net.sim([0, sim_duration])
    # conductance for simulation and pde
    # cond = heat/delta_temp
    sim_cond = 1. / abs(result[block.surf.left].temp(sim_duration))
    ref_cond = .81927  # measured at spatial resolution of 640x640
    # print('calculated conductance:   ', sim_cond)
    # print('reference conductance:   ', ref_cond)
    assert allclose(sim_cond, [ref_cond], rtol=.02), (
        'The simulated conductance is not close enough to the ' 'expected result!'
    )


@pytest.mark.parametrize('heat_in_body', [True, False])
def test_3d_heat_flow_on_block_with_heat_and_film_bcs(heat_in_body):
    """Test 3D heat flow on block body with heat and film BCs

    Test on steel plate with transient three-dimensional heat flow.
    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """
    steel = Solid(dens=8000., spec_heat=500., condy=50.)

    with Asm() as asm:
        block = Cube(
            width=.5,
            hgt=1.,
            depth=.05,
            width_div=20,
            hgt_div=80,
            depth_div=2,
        )
        Surf(
            name='upper_faces',
            faces=[
                block.face.left,
                block.face.right,
                block.face.front,
                block.face.back,
                block.face.top,
            ],
        )
        Surf(name='btm', faces=[block.face.btm])

    init_temp = 20.
    with Model() as model:
        cuboid = LPPart(asm=asm, matl=steel, init_temp=init_temp, name='cuboid')
        env = BoundNode(temp=20., posn=(-.3, .5, .025), name='environment')
        FilmLink(cuboid.surf.upper_faces, env, 10.)

        if heat_in_body:
            HeatSource(cuboid.surf.btm, 1000.)
        else:
            stat_node = StatNode()
            FilmLink(cuboid.surf.btm, stat_node, film=1e32)
            HeatSource(stat_node, heat=1000.)

    net = Network(model)

    # Ansys 12800 quadratic and regular hexahedron elements
    # dt_0 = 1e-2, dt_min = 1e-3, dt_max = 20
    ansys_res = genfromtxt(
        'cuboid flat ansys.csv', delimiter=';', names=True, case_sensitive=True
    )
    # Use 10 points in time from Ansys reference time data
    skip = math.ceil(len(ansys_res['Time']) / 10)
    ansys_res = ansys_res[::skip]
    # Skip first times because of big relative deviations
    ansys_res = ansys_res[2:]
    ref_time = ansys_res['Time']
    ref_bottom = ansys_res['Bottom']
    ref_top_vert = ansys_res['Top_vertex']
    ref_btm_vert = ansys_res['Bottom_vertex']

    res = net.sim([0., ref_time[-1]], method='LSODA')
    # Test results on mean surface temperatures and node temperatures
    sim_bottom = res[cuboid.surf.btm].temp(ref_time)
    sim_btm_vert, sim_top_vert = (
        res[cuboid].temp(ref_time, lctn=[(0., 0., .05), (.5, 1., .05)]).T
    )
    # print(rtol(ref_bottom-init_temp, sim_bottom-init_temp))
    # print(rtol(ref_top_vert-init_temp, sim_top_vert-init_temp))
    # print(rtol(ref_btm_vert-init_temp, sim_btm_vert-init_temp))
    if heat_in_body:
        assert allclose(ref_bottom - init_temp, sim_bottom - init_temp, rtol=.03)
        assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.009)
        assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.02)
    else:
        assert allclose(ref_bottom - init_temp, sim_bottom - init_temp, rtol=.03)
        assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.009)
        assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.01)


def test_part_multi_film_on_surf():
    """Test 3D heat flow with 2 film BCs on one surface

    The test scenario is a steel plate with transient 3D heat flow.
    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C. The HTC is split to two
    environments.
    """
    init_temp = 20.

    steel = Solid(dens=8000., spec_heat=500., condy=50.)

    with Asm() as asm:
        block = Cube(
            width=.5,
            hgt=1.,
            depth=.05,
            width_div=20,
            hgt_div=80,
            depth_div=2,
        )
        Surf(
            name='upper_faces',
            faces=[
                block.face.left,
                block.face.right,
                block.face.front,
                block.face.back,
                block.face.top,
            ],
        )
        Surf(name='btm', faces=[block.face.btm])

    with Model() as model:
        cuboid = LPPart(asm=asm, matl=steel, init_temp=init_temp, name='cuboid')
        HeatSource(cuboid.surf.btm, 1000.)
        env0 = BoundNode(temp=20., posn=(-.3, .25, .025), name='environment0')
        FilmLink(cuboid.surf.upper_faces, env0, 2.)
        env1 = BoundNode(temp=20., posn=(-.3, .75, .025), name='environment1')
        FilmLink(cuboid.surf.upper_faces, env1, 8.)

    net = Network(model)
    # Ansys 12800 quadratic and regular hexahedron elements
    # dt_0 = 1e-2, dt_min = 1e-3, dt_max = 20
    ansys_res = genfromtxt(
        'cuboid flat ansys.csv', delimiter=';', names=True, case_sensitive=True
    )
    # Use 10 points in time from Ansys reference time data
    skip = math.ceil(len(ansys_res['Time']) / 10)
    ansys_res = ansys_res[::skip]
    # Skip first times because of big relative deviations
    ansys_res = ansys_res[2:]
    ref_time = ansys_res['Time']
    ref_bottom = ansys_res['Bottom']
    ref_top_vert = ansys_res['Top_vertex']
    ref_btm_vert = ansys_res['Bottom_vertex']

    res = net.sim([0., ref_time[-1]], method='LSODA')
    # Test results on mean surface temperatures and node temperatures
    sim_bottom = res[cuboid.surf.btm].temp(ref_time)
    sim_btm_vert, sim_top_vert = (
        res[cuboid].temp(ref_time, lctn=[(0., 0., .05), (.5, 1., .05)]).T
    )
    # print(rtol(ref_bottom-init_temp, sim_bottom-init_temp))
    # print(rtol(ref_top_vert-init_temp, sim_top_vert-init_temp))
    # print(rtol(ref_btm_vert-init_temp, sim_btm_vert-init_temp))
    assert allclose(ref_bottom - init_temp, sim_bottom - init_temp, rtol=.03)
    assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.008)
    assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.03)


def test_two_linked_lp_parts0():
    """Test 3D heat flow with two linked LP-parts

    The test scenario is a steel plate with transient 3D heat flow.
    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C. The HTC is split to two
    environments.
    """

    steel = Solid(dens=8000., spec_heat=500., condy=50.)

    with Asm() as lower_asm:
        block = Cube(
            width=.5,
            hgt=.5,
            depth=.05,
            width_div=8,
            hgt_div=60,
            depth_div=1,
        )
        Surf(
            name='middel',
            faces=[
                block.face.left,
                block.face.right,
                block.face.front,
                block.face.back,
            ],
        )
        Surf(name='top', faces=[block.face.top])
        Surf(name='btm', faces=[block.face.btm])

    with Asm() as upper_asm:
        block = Cube(
            width=.5,
            hgt=.5,
            depth=.05,
            width_div=2,
            hgt_div=40,
            depth_div=1,
        )
        Surf(
            name='upper_faces',
            faces=[
                block.face.left,
                block.face.right,
                block.face.front,
                block.face.back,
                block.face.top,
            ],
        )
        Surf(name='btm', faces=[block.face.btm])

    with Model() as model:
        lower_cuboid = LPPart(
            asm=lower_asm, matl=steel, init_temp=20., name='lower_cuboid'
        )
        upper_cuboid = LPPart(
            posn=(0., .5, 0.),
            asm=upper_asm,
            matl=steel,
            init_temp=20.,
            name='upper_cuboid',
        )
        env = BoundNode(temp=20., posn=(-.3, .5, .025), name='environment')
        HeatSource(lower_cuboid.surf.btm, 1000.)
        FilmLink.multi(
            [lower_cuboid.surf.middel, upper_cuboid.surf.upper_faces], env, 10.
        )
        FilmLink(lower_cuboid.surf.top, upper_cuboid.surf.btm, 1000.)

    net = Network(model)

    # Ansys 12800 quadratic and regular hexahedron elements
    # dt_0 = 1e-2, dt_min = 1e-3, dt_max = 20
    ansys_res = genfromtxt(
        'cuboid flat ansys.csv', delimiter=';', names=True, case_sensitive=True
    )
    # Use 10 points in time from Ansys reference time data
    skip = math.ceil(len(ansys_res['Time']) / 10)
    ansys_res = ansys_res[::skip]
    ref_time = ansys_res['Time']
    ref_bottom = ansys_res['Bottom']

    res = net.sim(
        [0., ref_time[-1]],
    )
    sim_bottom = res[lower_cuboid.surf.btm].temp(ref_time)
    assert allclose(ref_bottom, sim_bottom, rtol=0, atol=3.)


@pytest.mark.parametrize('film_is_function', [True, False])
def test_two_linked_lp_parts1(film_is_function):
    """Test 3D heat flow with two linked LP-parts

    The test scenario are two horizontally connected blocks.
    Het flow from left to right driven by temperature bound nodes on
    the left and right surface of the blocks. Conductance of the
    connected blocks is measured and compared to a reference value.
    """

    matl = Solid(dens=1., spec_heat=1., condy=1.)

    def film_func(surf_temp, surround_temp, fluid=None):
        return 1e32

    with Model() as model:
        left_block = lp_parts.block(
            matl=matl,
            width=1.,
            width_div=2,
        )
        right_block = lp_parts.block(
            matl=matl,
            posn=(1., 0., 0),
            width=2.,
            width_div=2,
        )
        left_node = BoundNode(temp=1., posn=(-.5, .5, .5), name='left_node')
        # left_node = StatNode(posn=(-.5, .5, .5), name='left_node')
        HeatSource(left_node, 1.)
        right_node = BoundNode(temp=0., posn=(3.5, .5, .5), name='right_node')
        left_link = FilmLink(left_block.surf.left, left_node, 1e32)
        FilmLink(right_block.surf.right, right_node, 1e32)
        film = film_func if film_is_function else 1e32
        block_link = FilmLink(left_block.surf.right, right_block.surf.left, film)

    net = Network(model)
    sim_duration = 25.
    result = net.sim([0, sim_duration])
    # Overall conductance from left to right for simulation and pde
    # cond = heat/delta_temp
    sim_cond = abs(result[left_link].heat(sim_duration))
    # sim_cond = abs(result[left_node].temp(sim_duration))
    ref_cond = .3333
    # print('calculated conductance:   ', sim_cond)
    # print('reference conductance:   ', ref_cond)
    assert allclose(sim_cond, ref_cond, rtol=.02)
    sim_cond = abs(result[block_link].heat(sim_duration))
    assert allclose(sim_cond, ref_cond, rtol=.02)
    sim_film = abs(result[block_link].film(sim_duration))
    assert allclose(sim_film, 1e32, rtol=.02)


def test_film_func():
    """Test film given by function over time

    Test on steel plate with transient three-dimensional heat flow.
    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """

    def film_10(surf_temp, surround_temp, fluid=None):
        """Gets called in every solution step"""
        return 10.

    steel = Solid(dens=8000., spec_heat=500., condy=50.)

    with Asm() as asm:
        block = Cube(
            width=.5,
            hgt=1.,
            depth=.05,
            width_div=20,
            hgt_div=80,
            depth_div=2,
        )
        Surf(
            name='upper_faces',
            faces=[
                block.face.left,
                block.face.right,
                block.face.front,
                block.face.back,
                block.face.top,
            ],
        )
        Surf(name='btm', faces=[block.face.btm])

    init_temp = 20.
    with Model() as model:
        cuboid = LPPart(asm=asm, matl=steel, init_temp=init_temp, name='cuboid')
        env = BoundNode(temp=20., posn=(-.3, .5, .025), name='environment')
        FilmLink(cuboid.surf.upper_faces, env, film_10)
        HeatSource(cuboid.surf.btm, 1000.)

    net = Network(model)

    # Ansys 12800 quadratic and regular hexahedron elements
    # dt_0 = 1e-2, dt_min = 1e-3, dt_max = 20
    ansys_res = genfromtxt(
        'cuboid flat ansys.csv', delimiter=';', names=True, case_sensitive=True
    )
    # Use 10 points in time from Ansys reference time data
    skip = math.ceil(len(ansys_res['Time']) / 10)
    ansys_res = ansys_res[::skip]
    # Skip first times because of big relative deviations
    ansys_res = ansys_res[2:]
    ref_time = ansys_res['Time']
    ref_bottom = ansys_res['Bottom']
    ref_top_vert = ansys_res['Top_vertex']
    ref_btm_vert = ansys_res['Bottom_vertex']

    res = net.sim([0., ref_time[-1]], method='LSODA')
    # Test results on mean surface temperatures and node temperatures
    sim_bottom = res[cuboid.surf.btm].temp(ref_time)
    sim_btm_vert, sim_top_vert = (
        res[cuboid].temp(ref_time, lctn=[(0., 0., .05), (.5, 1., .05)]).T
    )
    # print(rtol(ref_bottom-init_temp, sim_bottom-init_temp))
    # print(rtol(ref_top_vert-init_temp, sim_top_vert-init_temp))
    # print(rtol(ref_btm_vert-init_temp, sim_btm_vert-init_temp))
    assert allclose(ref_bottom - init_temp, sim_bottom - init_temp, rtol=.03)
    assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.009)
    assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.02)


def test_temp_dep_condy():
    """Test rod with temperature dependent conductivity

    The test scenario is a rod with transient axial heat flow. The
    heat flow is caused by one temperature bound base and one heat
    loaded base. The lateral surfaces are adiabatic.
    """

    class Rod:
        lgth = 1.
        condy = array(
            [[0., 5.],
             [.1, .5]]
        )
        dens = 1.
        spec_heat = 2.
        initial_temp = 1.
        heat_flux = 1.  # On left end x = 0.
        border_temp = 0.  # On right end x = 1.
        result_lctns = [(.3, 0., 0.)]

    # Reference temperature at x = .3, calculated with lumped parameter model
    ref_times = array([.01, 1.244, 2.252, 3.177, 4.055, 4.905, 5.734, 6.548, 7.354, 8.152, 8.945, 9.736])
    ref_temps = array([1., 1.773, 2.199, 2.47, 2.657, 2.787, 2.88, 2.946, 2.994, 3.028, 3.052, 3.07])

    with Asm() as rod_asm:
        # 1x.05x.05
        cyl_rod = Cyl(
            lgth=1.,
            lgth_div=350,  # 5 * x
            name='cyl',
        )
        Surf(name='begin', faces=[cyl_rod.face.base])
        Surf(name='end', faces=[cyl_rod.face.end])

    test_matl = Solid(
        condy=Rod.condy,
        dens=Rod.dens,
        spec_heat=Rod.spec_heat,
    )

    with Model() as model:
        rod_part = LPPart(
            asm=rod_asm,
            matl=test_matl,
            init_temp=Rod.initial_temp,
            temp_dependent=True,
            name='rod',
        )
        # Compensate for absent margin conductance:
        # Heat applied to surfaces of LPParts is fed in nodes located
        # inside the part. The margin conductance will not be
        # considered. For comparison with precise PDE solutions, this
        # is compensated for by a stationary node.
        stat_node = StatNode()
        FilmLink(rod_part.surf.begin, stat_node, film=1e32)
        HeatSource(stat_node, heat=Rod.heat_flux * rod_part.surf.begin.area())
        # Boundary temperatures on part surfaces are not possible
        # Use a linked temperature bound node instead
        bound_node = BoundNode(temp=Rod.border_temp)
        FilmLink(rod_part.surf.end, bound_node, film=1e32)

    result = Network(model).sim(time_span=(ref_times[0], ref_times[-1]), method='LSODA')
    temps = result[rod_part].temp(ref_times, Rod.result_lctns).ravel()
    # print(f"{ref_temps=}")
    # print(f"{temps=}")
    # print(f"{temps-ref_temps=}")
    assert allclose(ref_temps, temps, rtol=0, atol=.005)


def test_temp_dep_dens():
    """Test rod with temperature dependent density

    The test scenario is a rod with transient axial heat flow. The
    heat flow is caused by one temperature bound base and one heat
    loaded base. The lateral surfaces are adiabatic.
    """

    class Rod:
        lgth = 1.
        condy = .2
        dens = array(
            [[0., 5.],
             [1., .5]]
        )
        spec_heat = 2.
        initial_temp = 1.
        heat_flux = 1.  # On left end x = 0.
        border_temp = 0.  # On right end x = 1.
        result_lctns = [(.3, 0., 0.)]

    # Reference temperature at x = .3, calculated with lumped parameter model
    ref_times = array([.01, 1.212, 2.304, 3.325, 4.297, 5.236, 6.148, 7.043, 7.925, 8.798, 9.665])
    ref_temps = array([1., 1.769, 2.28, 2.631, 2.88, 3.057, 3.184, 3.275, 3.341, 3.388, 3.422])

    with Asm() as rod_asm:
        # 1x.05x.05
        cyl_rod = Cyl(
            lgth=1.,
            lgth_div=350,  # 5 * x
            name='cyl',
        )
        Surf(name='begin', faces=[cyl_rod.face.base])
        Surf(name='end', faces=[cyl_rod.face.end])

    test_matl = Solid(
        condy=Rod.condy,
        dens=Rod.dens,
        spec_heat=Rod.spec_heat,
    )

    with Model() as model:
        rod_part = LPPart(
            asm=rod_asm,
            matl=test_matl,
            init_temp=Rod.initial_temp,
            temp_dependent=True,
            name='rod',
        )
        # Compensate for absent margin conductance:
        # Heat applied to surfaces of LPParts is fed in nodes located
        # inside the part. The margin conductance will not be
        # considered. For comparison with precise PDE solutions, this
        # is compensated for by a stationary node.
        stat_node = StatNode()
        FilmLink(rod_part.surf.begin, stat_node, film=1e32)
        HeatSource(stat_node, heat=Rod.heat_flux * rod_part.surf.begin.area())
        # Boundary temperatures on part surfaces are not possible
        # Use a linked temperature bound node instead
        bound_node = BoundNode(temp=Rod.border_temp)
        FilmLink(rod_part.surf.end, bound_node, film=1e32)

    result = Network(model).sim(time_span=(ref_times[0], ref_times[-1]), method='LSODA')
    temps = result[rod_part].temp(ref_times, Rod.result_lctns).ravel()
    # print(f"{ref_temps=}")
    # print(f"{temps=}")
    # print(f"{temps-ref_temps=}")
    assert allclose(ref_temps, temps, rtol=0, atol=.006)


def rtol(sim, ref):
    return abs(ref - sim) / abs(ref)
