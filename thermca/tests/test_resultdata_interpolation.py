from numpy import array, exp, allclose, set_printoptions, column_stack
from thermca import *

set_printoptions(formatter={'float': '{: 0.4f}'.format}) #, linewidth=300)


def test_result_conductance_interpolation():

    print('\nTest result conductance interpolation:')

    times = array([0, .5, 1, 1.5, 2, 2.5, 3.])
    conds = array([0, .5, 1, 1.5, 2, 2.5, 3.])
    with Model() as model:
        cond_inp = Input(column_stack([times, conds]))
        bound_temp = BoundNode(
            temp=1.,
            posn=(0., 0., 0.),
            name='environment_temperature'
        )
        point_capy = Node(posn=(1., 0., 0.), capy=1., name='point_body')

        def cond_func(temp0, temp1, matl):
            return cond_inp.value
        cond_link = CondLink(
            point_capy,
            bound_temp,
            cond=cond_func)

    # simulation
    # simtime = 3.  # duration of the simulation
    net = Network(model)
    # create a Result instance
    result = net.sim((0., 3.), rel_tol=1e-6, abs_tol=1e-6, method='RK23')
    # interpolate node element temperature results over time
    result_times = times[:-1]
    point_capy_temps = result[point_capy].temp(result_times)
    reference_conds = conds[:-1]
    sim_conds = result[cond_link].cond(result_times + .1)  # test just slightly after step
    print('times: ', result._data.times)
    # print('temps: ', result[point_capy].data.temps[:, 0].T)
    # print('conds: ', result[point_capy, bound_temp].cond(result.data.times))
    # print('Times:            ', times)
    print('Temps.:           ', point_capy_temps)
    print('reference conds.: ', reference_conds)
    print('Sim. res. conds.: ', sim_conds)

    assert (allclose(sim_conds, reference_conds, rtol=0, atol=2e-03)), (
        "The simulated conductances are not close enough to the "
        "expected result!")


def test_result_heat_flow_interpolation():
    print('\nTest result heat flow interpolation:')
    # first build the model
    cond = 1.
    with Model() as model:
        body = Node(capy=1., posn=(1., 0., 0.), name='body')
        bnode = BoundNode(1., posn=(0., 0., 0.), name='bound_temperature')
        cond_link = CondLink(body, bnode, cond=cond)
    # simulation
    result_times = array([.01, .02, .05, .1, .2, .5, 1.])
    net = Network(model)
    result = net.sim((0., 1.), rel_tol=1e-6, abs_tol=1e-6, method='RK23')
    # verification of the temperature with the pde-solution
    def pde_solution_heat(time):
        delta_temp = (1. - exp(-time)) - 1.
        return cond * delta_temp
    reference_heats = pde_solution_heat(result_times)
    sim_heats = result[cond_link].heat(result_times)
    print('Times:           ', result_times)
    print('Temperatures:    ', result[body].temp(result_times))
    print('Reference flows: ', reference_heats)
    print('Sim. res. flows: ', sim_heats)
    assert (allclose(sim_heats, reference_heats, rtol=0, atol=2e-03)), (
        "The simulated heat flows are not close enough to the "
        "expected result!")


def test_result_film_coefficient_and_heat_flux_interpolation():
    print('\nTest result film coefficient and heat flux interpolation:')
    height = 2.

    test_matl = Solid(
        condy=1.,
        dens=1.,
        spec_heat=2.)

    times = array([0, .5, 1, 1.5, 2, 2.5, 3])
    films = array([0, .5, 1, 1.5, 2, 2.5, 3])

    with Asm() as cuboid:
        block = Cube(hgt=height)
        Surf(name='surface', faces=block.face)

    with Model() as model:
        film_inp = Input(column_stack([times, films]))
        bound_temp = BoundNode(
            temp=1.,
            name='environment_temperature'
        )
        block = LPPart(asm=cuboid, matl=test_matl, name='Block_body')
        film_link = FilmLink(
            block.surf.surface,
            bound_temp,
            film=lambda temp0, temp1, fluid: film_inp.value
        )

    # simulation
    net = Network(model)
    # create a Result instance
    result = net.sim(time_span=[0., 3.], rel_tol=1e-6, abs_tol=1e-6, method='RK23')
    # result.data.plot2d(1.)  # .condensed()  , hide=('label', 'node num')
    res_times = times[:-1]
    reference_films = films[:-1]
    sim_films = result[film_link].film(res_times + .1)  # test just slightly after step
    print('Times:            ', times)
    print('Temps.:           ', result[block].temp(times))
    print('Reference films.: ', reference_films)
    print('Sim. res. films.: ', sim_films)
    assert (allclose(sim_films, reference_films, rtol=0, atol=2e-03)), (
        "The simulated film coefficients are not close enough to the "
        "expected result!")
    reference_fluxs = result[film_link].heat(times) / block.surf.surface.area()
    sim_fluxs = result[film_link].flux(times)
    print('Reference fluxs.: ', reference_fluxs)
    print('Sim. res. fluxs.: ', sim_fluxs)
    assert (allclose(sim_fluxs, reference_fluxs, rtol=0, atol=2e-03)), (
        "The simulated heat fluxes are not close enough to the "
        "expected result!")
