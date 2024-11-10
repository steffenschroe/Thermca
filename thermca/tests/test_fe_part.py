"""Test part implementation"""
import math

import pytest
from numpy import array, genfromtxt, allclose, zeros, sin, cos, exp, pi

from thermca import *
from thermca.fem.fe_system import fenics_installed

test_steel = Solid(dens=8000., spec_heat=500., condy=50.)


@pytest.mark.parametrize('mor_dof', [None, 20])
@pytest.mark.parametrize('fe_assembly', ['skfem', 'fenics'])
def test_part_flux_and_film(mor_dof, fe_assembly):
    """Test one dimensional heat flow

    The test scenario is a rod with heat flow driven by a temperature
    and a heat boundary condition at each end. The rod has adiabatic
    lateral surfaces. The rod is modeled with a long cuboid measuring
    1 x .05 x .05 m. The temperature boundary condition is implemented
    with a node element connected by a film link element because
    temperature boundary conditions are not possible directly on
    surfaces.
    """

    if fe_assembly == 'fenics' and not fenics_installed:
        pytest.skip()

    class Rod:
        lgth = 1.
        condy = .2
        dens = 1.
        spec_heat = 2.
        init_temp = 1.
        heat_flux = 1.  # On left end x = 0.
        bound_temp = 0.  # On right end x = 1.
        result_lctns = [(.3, 0., 0.)]
        result_times = array([.01, .02, .05, .1, .2, .5, 1., 2., 5., 10])

    def pde(rod):
        """Analytical solution as reference"""
        l = rod.lgth
        a = rod.condy / (rod.dens * rod.spec_heat)  # thermal diffusivity
        la = rod.condy
        tx0 = rod.init_temp
        q0 = rod.heat_flux
        t1 = rod.bound_temp
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

    def fem_sim(rod):
        cuboid_mesh = Mesh.read('rod_1x.05x.05.med')
        test_matl = Solid(condy=Rod.condy, dens=Rod.dens, spec_heat=Rod.spec_heat)
        with Model() as model:
            cuboid = FEPart(
                mesh=cuboid_mesh,
                matl=test_matl,
                init_temp=rod.init_temp,
                mor_dof=mor_dof,
                name='rod',
            )
            env = BoundNode(temp=rod.bound_temp)
            FluxSource(cuboid.surf.left, flux=rod.heat_flux)
            FilmLink(cuboid.surf.right, env, 1.e3)
        result = Network(model).sim(
            time_span=(rod.result_times[0], rod.result_times[-1]), method='LSODA'
        )
        sim_temps = zeros((len(rod.result_times), len(rod.result_lctns)))
        for itime, time in enumerate(rod.result_times):
            sim_temps[itime, :] = result[cuboid].temp(time, rod.result_lctns)
        return sim_temps

    ref_temps = pde(Rod).ravel()
    sim_temps = fem_sim(Rod).ravel()
    # print(f"{ref_temps=}")
    # print(f"{sim_temps=}")
    # print(f"{sim_temps-ref_temps=}")
    assert allclose(ref_temps, sim_temps, rtol=0, atol=.02)


@pytest.mark.parametrize('mor_dof', [None, 30])
@pytest.mark.parametrize('fe_assembly', ['skfem', 'fenics'])
def test_part_heat_and_film(mor_dof, fe_assembly):
    """Test 3D heat flow on steel plate with heat and film BCs

    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """
    if fe_assembly == 'fenics' and not fenics_installed:
        pytest.skip()

    init_temp = 20.

    mesh = Mesh.read('flat_cuboid_fine.med')  # surfs: 'bottom', 'upper_faces'
    # mesh.plot().show()

    with Model() as model:
        cuboid = FEPart(
            mesh,
            test_steel,
            init_temp=init_temp,
            fe_assembly=fe_assembly,
            mor_dof=mor_dof,
            name='cuboid',
        )
        env = BoundNode(temp=20., posn=(-.3, .5, .025), name='environment')
        HeatSource(cuboid.surf.bottom, 1000.)
        FilmLink(cuboid.surf.upper_faces, env, 10.)

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
    sim_bottom = res[cuboid.surf.bottom].temp(ref_time)
    sim_btm_vert, sim_top_vert = res[cuboid].temp(
        ref_time, lctn=[(0., 0., .05,), (.5, 1., .05,)]).T
    # print(f"{mor_dof=}, {fe_assembly=}")
    # print(rtol(ref_bottom - init_temp, sim_bottom - init_temp))
    # print(rtol(ref_top_vert - init_temp, sim_top_vert - init_temp))
    # print(rtol(ref_btm_vert - init_temp, sim_btm_vert - init_temp))
    if mor_dof is None:
        assert allclose(ref_bottom - init_temp, sim_bottom - init_temp, rtol=.0007)
        assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.002)
        assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.0006)
    else:
        assert allclose(ref_bottom - init_temp, sim_bottom - init_temp, rtol=.0013)
        assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.061)
        assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.0071)


@pytest.mark.parametrize('mor_dof', [None, 30])
def test_part_multi_film_on_surf(mor_dof):
    """Test 3D heat flow with 2 film BCs on one surface

    The test scenario is a steel plate with transient 3D heat flow.
    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """

    init_temp = 20.

    mesh = Mesh.read('flat_cuboid_fine.med')  # surfs: 'bottom', 'upper_faces'

    with Model() as model:
        cuboid = FEPart(
            mesh, test_steel, init_temp=init_temp, mor_dof=mor_dof, name='cuboid'
        )
        env0 = BoundNode(temp=20., posn=(-.3, .25, .025), name='environment0')
        env1 = BoundNode(temp=20., posn=(-.3, .75, .025), name='environment1')
        HeatSource(cuboid.surf.bottom, 1000.)
        FilmLink(cuboid.surf.upper_faces, env0, 4.)
        FilmLink(cuboid.surf.upper_faces, env1, 6.)

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
    sim_bottom = res[cuboid.surf.bottom].temp(ref_time)
    sim_btm_vert, sim_top_vert = res[cuboid].temp(
        ref_time, lctn=[(0., 0., .05,), (.5, 1., .05,)]).T
    # print(f"{mor_dof=}")
    # print(rtol(ref_bottom - init_temp, sim_bottom - init_temp))
    # print(rtol(ref_top_vert - init_temp, sim_top_vert - init_temp))
    # print(rtol(ref_btm_vert - init_temp, sim_btm_vert - init_temp))
    if mor_dof is None:
        assert allclose(ref_bottom - init_temp, sim_bottom - init_temp, rtol=.0007)
        assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.002)
        assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.0006)
    else:
        assert allclose(ref_bottom - init_temp, sim_bottom - init_temp, rtol=.0009)
        assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.2)
        assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.007)


@pytest.mark.parametrize('mor_dof', [None, 30])
@pytest.mark.parametrize('fe_assembly', ['skfem', 'fenics'])
def test_part_heat_and_film_on_multi_surf(mor_dof, fe_assembly):
    """Steel plate with transient three-dimensional heat flow

    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """
    if fe_assembly == 'fenics' and not fenics_installed:
        pytest.skip()

    init_temp = 20.

    mesh = Mesh.read('plate.msh')

    with Model() as model:
        plate = FEPart(
            mesh,
            test_steel,
            init_temp=init_temp,
            fe_assembly=fe_assembly,
            mor_dof=mor_dof,
            name='cuboid',
        )
        env = BoundNode(temp=20., posn=(-.3, .5, .025), name='environment')
        HeatSource(plate.surf.bottom, 1000.)
        FilmLink.multi(
            [
                plate.surf.left,
                plate.surf.right,
                plate.surf.front,
                plate.surf.back,
                plate.surf.top,
            ],
            env,
            10.,
        )

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
    sim_bottom = res[plate.surf.bottom].temp(ref_time)
    sim_btm_vert, sim_top_vert = res[plate].temp(
        ref_time, lctn=[(0., 0., .05,), (.5, 1., .05,)]).T
    # print(f"{mor_dof=}, {fe_assembly=}")
    # print(rtol(ref_bottom - init_temp, sim_bottom - init_temp))
    # print(rtol(ref_top_vert - init_temp, sim_top_vert - init_temp))
    # print(rtol(ref_btm_vert - init_temp, sim_btm_vert - init_temp))
    if mor_dof is None:
        assert allclose(ref_bottom - init_temp, sim_bottom - init_temp, rtol=.003)
        assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.02)
        assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.006)
    else:
        assert allclose(ref_bottom - init_temp, sim_bottom - init_temp, rtol=.003)
        assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.06)
        assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.003)


@pytest.mark.parametrize('mor_dof', [None, 30])
def test_part_with_film_func(mor_dof):
    """Steel plate with transient three-dimensional heat flow

    Test film coefficient given by a function.

    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """

    def film_10(surf_temp, surround_temp, fluid=None):
        """Gets called in every solution step"""
        return 10.

    mesh = Mesh.read('flat_cuboid_fine.med')  # surfs: 'bottom', 'upper_faces'

    with Model() as model:
        cuboid = FEPart(
            mesh, test_steel, init_temp=20., mor_dof=mor_dof, name='cuboid'
        )
        env = BoundNode(temp=20., posn=(-.3, .5, .025), name='environment')
        HeatSource(cuboid.surf.bottom, 1000.)
        FilmLink(cuboid.surf.upper_faces, env, film_10)

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

    res = net.sim([0., ref_time[-1]], method='LSODA')

    sim_bottom = res[cuboid.surf.bottom].temp(ref_time)
    # print(f"{mor_dof=}")
    # print(f"{ref_time}")
    # print(f"{ref_bottom}")
    # print(f"{abs(ref_bottom-sim_bottom)=}")
    assert allclose(ref_bottom, sim_bottom, rtol=0, atol=.7)


@pytest.mark.parametrize('mor_dof', [None, 30])
def test_two_connected_parts(mor_dof):
    """Steel plate with transient three-dimensional heat flow

    Test steel plate modelled as two connected parts.

    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """

    # half_flat_cuboid: .5x.5x.05; body: 'cuboid', surfs: 'bottom', 'middel', 'top'
    half_cuboid = Mesh.read('half_flat_cuboid.med')

    with Model() as model:
        lower_cuboid = FEPart(
            half_cuboid,
            test_steel,
            init_temp=20.,
            mor_dof=mor_dof,
            name='lower_cuboid',
        )
        upper_cuboid = FEPart(
            half_cuboid,
            test_steel,
            posn=(0., .5, 0.),
            mor_dof=mor_dof,
            init_temp=20.,
            name='upper_cuboid',
        )
        env = BoundNode(temp=20., name='environment')
        HeatSource(lower_cuboid.surf.bottom, 1000.)
        FilmLink(lower_cuboid.surf.middel, env, 10.)
        FilmLink(lower_cuboid.surf.top, upper_cuboid.surf.bottom, 1000.)
        FilmLink(upper_cuboid.surf.middel, env, 10.)
        FilmLink(upper_cuboid.surf.top, env, 10.)

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

    res = net.sim([0., ref_time[-1]], method='LSODA')
    sim_bottom = res[lower_cuboid.surf.bottom].temp(ref_time)
    # Mean dock temperature
    # print(f"{mor_dof=}")
    # print(f"{abs(ref_bottom-sim_bottom)=}")
    assert allclose(ref_bottom, sim_bottom, rtol=0, atol=2.)


@pytest.mark.parametrize('mor_dof', [None, 30])
def test_connected_parts_with_film_between_unequal_surfaces(mor_dof):
    """Transient radial heat flow over two connected parts

    The test aims at the thermal connection of unequal surface areas
    by film coefficients. The resulting conductance should be film
    coefficient multiplied by the mean surface area of the connected
    surfaces. Furthermore, one part is connected to by two surfaces.

    The parts are short concentrically arranged steel tubes. The inner
    tube has an inner radius of .01 and an outer radius of .05. The
    outer tube has an inner radius of .06 and an outer radius of 1..
    The end surfaces of the tubes are adiabatic. The connection between
    the outer and inner tube is modelled with a film coefficient of
    1000. The heat flow is driven by a heat source of 250. on the inner
    surface of the inner ring. The outer surface of the outer ring is
    connected to temperature bound node with a film coefficient of 100.
    The initial temperature is 20.
    """

    # Reference by lumped parameter simulation
    inner_source_ref_res = genfromtxt(
        'rings_with_inner_source.csv', delimiter=',', names=True
    )
    ref_times = inner_source_ref_res['time']
    ref_outer_tube_outer = inner_source_ref_res['outer_tube_outer']
    inner_mesh = Mesh.read('inner_tube.med')
    outer_mesh = Mesh.read('outer_tube.med')
    init_temp = 20.
    outer_film = 100.
    gap_film = 1000.

    with Model() as fem_model:
        inner_tube = FEPart(
            mesh=inner_mesh,
            matl=test_steel,
            init_temp=init_temp,
            mor_dof=mor_dof,
            name='inner_tube',
        )
        outer_tube = FEPart(
            mesh=outer_mesh,
            matl=test_steel,
            init_temp=init_temp,
            mor_dof=mor_dof,
            name='outer_tube',
        )
        env = BoundNode(temp=init_temp, posn=(0., .13, .025), name='environment')
        FilmLink(inner_tube.surf.outer, outer_tube.surf.inner, gap_film)
        FilmLink(outer_tube.surf.outer, env, outer_film)
        HeatSource(inner_tube.surf.inner, 250.)
    # fem_model.plot().show()
    net = Network(fem_model)
    # net.plot().show()
    res = net.sim((0., 1.5 * 60 * 60), method='RK45')
    outer_tube_outer_temps = res[outer_tube.surf.outer].temp(time=ref_times)
    # print(f"{mor_dof=}")
    print("dtemp_max: ", max(abs(ref_outer_tube_outer - outer_tube_outer_temps)))
    assert allclose(ref_outer_tube_outer, outer_tube_outer_temps, rtol=0., atol=3.5)
    num_temps = len(ref_outer_tube_outer)
    assert allclose(
        (ref_outer_tube_outer - init_temp)[num_temps // 5 :],
        (outer_tube_outer_temps - init_temp)[num_temps // 5 :],
        rtol=.1,
        atol=.2,
    )
    # On two outer locations at middle of outer surface
    outer_tube_outer_temps = (
        res[outer_tube].temp(time=ref_times, lctn=[(0., .1, .025)]).T[0]
    )
    print("dtemp_max: ", max(abs(ref_outer_tube_outer - outer_tube_outer_temps)))
    assert allclose(ref_outer_tube_outer, outer_tube_outer_temps, rtol=0., atol=.6)
    outer_tube_outer_temps = res[outer_tube].temp(time=ref_times, lctn=[(.1, 0., .025)]).T[0]
    print("dtemp_max: ", max(abs(ref_outer_tube_outer - outer_tube_outer_temps)))
    assert allclose(ref_outer_tube_outer, outer_tube_outer_temps, rtol=0., atol=.3)


@pytest.mark.parametrize('mor_dof', [None, 10])
@pytest.mark.parametrize('fe_assembly', ['skfem', 'fenics'])
def test_connected_parts_with_unequal_heat_sources_on_connected_surfaces(
    mor_dof, fe_assembly
):
    """Transient radial heat flow over two connected parts

    The test aims at the thermal connection of unequal surface areas
    by film coefficients and additionally with unequal heat sources on
    the connected surfaces.

    The parts are short concentrically arranged steel tubes. The inner
    tube has an inner radius of .01 and an outer radius of .05. The
    outer tube has an inner radius of .06 and an outer radius of 1..
    The end surfaces of the tubes are adiabatic. The connection between
    the outer and inner tube is modelled with a film coefficient of
    500. The heat flow is driven by unequal heat source on the gap
    surfaces (150. on inner tube outer surface, 250. on outer tube
    inner surface). The outer surface of the outer ring is connected
    to temperature bound node with a film coefficient of 50. The
    initial temperature is 20.
    """
    if fe_assembly == 'fenics' and not fenics_installed:
        pytest.skip()

    inner_source_ref_res = genfromtxt(
        'rings_with_gap_sources.csv', delimiter=',', names=True
    )
    ref_times = inner_source_ref_res['time']
    ref_outer_tube_outer = inner_source_ref_res['outer_tube_outer']
    inner_mesh = Mesh.read('inner_tube.med')
    outer_mesh = Mesh.read('outer_tube.med')
    init_temp = 20.
    gap_film = 500.
    outer_film = 50.
    steel = Solid(condy=25., dens=8000., spec_heat=500.)
    with Model() as fem_model:
        inner_tube = FEPart(
            mesh=inner_mesh,
            matl=steel,
            init_temp=init_temp,
            fe_assembly=fe_assembly,
            mor_dof=mor_dof,
            name='inner_tube',
        )
        outer_tube = FEPart(
            mesh=outer_mesh,
            matl=steel,
            init_temp=init_temp,
            fe_assembly=fe_assembly,
            mor_dof=mor_dof,
            name='outer_tube',
        )
        env = BoundNode(
            temp=init_temp,
            posn=(0., .13, .025),
            name='environment',
        )
        FilmLink(inner_tube.surf.outer, outer_tube.surf.inner, gap_film)
        FilmLink(outer_tube.surf.outer, env, outer_film)
        FilmLink(inner_tube.surf.inner, env, outer_film)
        HeatSource(inner_tube.surf.outer, 150.)
        HeatSource(outer_tube.surf.inner, 250.)
    # fem_model.plot().show()
    net = Network(fem_model)
    # net.plot().show()
    res = net.sim((0., 3. * 60 * 60))
    outer_tube_outer_temps = res[outer_tube.surf.outer].temp(time=ref_times)
    # print(f"{mor_dof=}")
    # print(f"{ref_outer_tube_outer=}, \n{outer_tube_outer_temps=}")
    # print("dtemp_max: ", max(abs(ref_outer_tube_outer-outer_tube_outer_temps)))
    num_temps = len(ref_outer_tube_outer)
    assert allclose(ref_outer_tube_outer, outer_tube_outer_temps, rtol=0., atol=.4)
    assert allclose(
        (ref_outer_tube_outer - init_temp)[num_temps // 5 :],
        (outer_tube_outer_temps - init_temp)[num_temps // 5 :],
        rtol=.01,
        atol=0.,
    )


def test_part_with_temp_dep_condy():
    """Rod with transient axial heat flow and temperature dependent conductivity

    The heat flow is caused by one temperature bound base and one heat
    loaded base. The lateral surfaces are adiabatic.
    """

    class Rod:
        lgth = 1.
        condy = array([[0., 5.], [.1, .5]])
        dens = 1.
        spec_heat = 2.
        initial_temp = 1.
        heat_flux = 1.  # On left end x = 0.
        border_temp = 0.  # On right end x = 1.
        result_lctns = [(.3, 0., 0.)]

    # Reference temperature at x = .3, calculated with lumped parameter model
    ref_times = array([.01, 1.244, 2.252, 3.177, 4.055, 4.905, 5.734, 6.548, 7.354, 8.152, 8.945, 9.736])
    ref_temps = array([1., 1.773, 2.199, 2.47, 2.657, 2.787, 2.88, 2.946, 2.994, 3.028, 3.052, 3.07])

    cuboid_mesh = Mesh.read('rod_1x.05x.05.med')  # With vertex at x = .3
    test_matl = Solid(
        condy=Rod.condy,
        dens=Rod.dens,
        spec_heat=Rod.spec_heat,
    )
    with Model() as model:
        cuboid = FEPart(
            mesh=cuboid_mesh,
            matl=test_matl,
            init_temp=Rod.initial_temp,
            temp_dependent=True,
            name='rod',
        )
        env = BoundNode(temp=Rod.border_temp)
        FluxSource(cuboid.surf.left, flux=Rod.heat_flux)
        FilmLink(cuboid.surf.right, env, 1.e3)
    result = Network(model).sim(time_span=(ref_times[0], ref_times[-1]), method='LSODA')
    temps = result[cuboid].temp(ref_times, Rod.result_lctns).ravel()
    # print(f"{ref_temps=}")
    # print(f"{temps=}")
    # print(f"{temps-ref_temps=}")
    assert allclose(ref_temps, temps, rtol=0, atol=.005)


def test_part_with_temp_dep_dens():
    """Rod with transient axial heat flow and temperature dependent density

    The heat flow is caused by one temperature bound base and one heat
    loaded base. The lateral surfaces are adiabatic.
    """

    class Rod:
        lgth = 1.
        condy = .2
        dens = array([[0., 5.], [1., .5]])
        spec_heat = 2.
        initial_temp = 1.
        heat_flux = 1.  # On left end x = 0.
        border_temp = 0.  # On right end x = 1.
        result_lctns = [(.3, 0., 0.)]

    # Reference temperature at x = .3, calculated with lumped parameter model
    ref_times = array([.01, 1.212, 2.304, 3.325, 4.297, 5.236, 6.148, 7.043, 7.925, 8.798, 9.665])
    ref_temps = array([1., 1.769, 2.28, 2.631, 2.88, 3.057, 3.184, 3.275, 3.341, 3.388, 3.422])

    cuboid_mesh = Mesh.read('rod_1x.05x.05.med')  # With vertex at x = .3
    test_matl = Solid(
        condy=Rod.condy,
        dens=Rod.dens,
        spec_heat=Rod.spec_heat,
    )
    with Model() as model:
        cuboid = FEPart(
            mesh=cuboid_mesh,
            matl=test_matl,
            init_temp=Rod.initial_temp,
            temp_dependent=True,
            name='rod',
        )
        env = BoundNode(temp=Rod.border_temp)
        FluxSource(cuboid.surf.left, flux=Rod.heat_flux)
        FilmLink(cuboid.surf.right, env, 1.e3)
    result = Network(model).sim(time_span=(ref_times[0], ref_times[-1]), method='LSODA')
    temps = result[cuboid].temp(ref_times, Rod.result_lctns).ravel()
    # print(f"{ref_temps=}")
    # print(f"{temps=}")
    # print(f"{temps-ref_temps=}")
    assert allclose(ref_temps, temps, rtol=0, atol=.02)


def rtol(sim, ref):
    return abs(ref - sim) / abs(ref)
