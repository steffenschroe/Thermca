from numpy import array, zeros, allclose, exp, linspace

from thermca.pointnodes import Node, BoundNode
from thermca.model import Model
from thermca.network import Network
from thermca.links import CondLink
from thermca.source import HeatSource
from thermca.input import Input
from numpy import set_printoptions


set_printoptions(formatter={'float': '{: 0.3f}'.format})


# first build the test model
with Model() as model:
    body = Node(capy=1., posn=(1., 0., 0.), name='body')
    bnode = BoundNode(temp=1., posn=(0., 0., 0.), name='bound_temperature')
    clink = CondLink(bnode, body, cond=1.)
    time_step = Input([[0., 0.],
                       [.2, 1.]])  # used later
    hsource = HeatSource(body, heat=0.)  # used later


def test_instant_step_in_environment_temp():
    """The test case is a simple PT1 behaviour. It is a step response of an
    even heated body which gets heated from the environment. It is build
    with one node with the capacity of 1. and the start temperature of 0.
    One boundary node with the constant temperature of 1. is connected with
    a simple link of the conduction 1. The simulation is done over time of
    1. The body should at this time be heated to a temperature of 0.632.
    """
    print()
    print('Response of a node with a step in the temperature of the environment:')
    # simulation
    result_times = array([.01, .02, .05, .1, .2, .5, 1.])
    net = Network(model)
    result = net.sim(time_span=(0., result_times[-1]), rel_tol=1e-6, abs_tol=1e-6)
    # evaluation of the 'Body' node temperature over time
    sim_temps = result[body].temp(result_times)
    print('times                : ', result_times)
    print('simulated body temps.: ', sim_temps)
    # verification of the temperature with the pde-solution
    pde_solution = lambda time: 1-exp(-time)
    reference_temps = pde_solution(result_times)
    print('reference body temps.: ', reference_temps)
    assert(allclose(sim_temps, reference_temps, rtol=0, atol=2e-02)), (
        'The simulated temperatures are not close enough to the '
        'expected result!')
    print()


def test_delayed_step_in_environment_temp():
    print('Response of a node with a with a environment temperature step after .2 '
          'seconds:')
    # first, change the model
    with model:
        bnode.temp = time_step.get_value
    node_net = Network(model)
    # simulation
    result_times = array([.01, .02, .05, .1, .2, .5, 1., 2.])
    result = node_net.sim(time_span=(0., result_times[-1]), rel_tol=1e-6, abs_tol=1e-6)
    # evaluation of the 'Body' node temperature over time
    sim_temps = result[body].temp(result_times)
    print('times                : ', result_times)
    print('environment temps.   : ', result[bnode].temp(result_times))
    print('simulated body temps.: ', sim_temps)
    # verification of the temperature with pde-solution as reference
    pde_solution = lambda time: 0. if time < .2 else 1-exp(-(time-.2))
    reference_temps = zeros(result_times.shape)
    for i, time in enumerate(result_times):
        reference_temps[i] = pde_solution(time)
    print('reference body temps.: ', reference_temps)
    assert(allclose(sim_temps, reference_temps, rtol=0, atol=1e-02)), (
        'The simulated temperatures are not close enough to the '
        'expected result!')
    print()


def test_delayed_step_in_conduction():
    print('Response of a node with a with a delayed conduction step after .2 '
          'second:')
    # first change the model
    bnode.temp = 1.
    clink.cond = lambda _0, _1, _2: time_step.value
    node_net = Network(model)
    # simulation
    result_times = array([.01, .02, .05, .1, .2, .5, 1., 2.])
    result = node_net.sim(time_span=(0., result_times[-1]), rel_tol=1e-6, abs_tol=1e-6)
    # evaluation of the 'Body' node temperature over time
    sim_temps = result[body].temp(result_times)
    print('times                  : ', result_times)
    print('body-environment conds.: ', result[clink].cond(result_times))
    print('simulated body temps.  : ', sim_temps)
    # verification of the temperature with the pde-solution
    pde_solution = lambda time: 0. if time < .2 else 1-exp(-(time-.2))
    reference_temps = zeros(result_times.shape)
    for i, time in enumerate(result_times):
        reference_temps[i] = pde_solution(time)
    print('reference body temps.  : ', reference_temps)
    assert(allclose(sim_temps, reference_temps, rtol=0, atol=1e-02)), (
        'The simulated temperatures are not close enough to the '
        'expected result!')
    print()


def test_instant_step_in_heat_source():
    """The test case is a simple PT1 behaviour. It is a step response of an even
    heated body which gets heated from a heat source. It is build with one
    node with the capacity of 1. and the start temperature of 0. One
    boundary node with the constant temperature of 0. is connected with a
    simple link of conduction 1. The simulation is done over time of 1. The
    body should at this time be heated to a temperature of 0.632.
    """
    print('Response of a node with a instant heat source step:')
    # first change the model
    bnode.temp = 0.
    clink.cond = 1.
    hsource.heat = 1.
    node_net = Network(model)
    result_times = array([.01, .02, .05, .1, .2, .5, 1., 2.])
    result = node_net.sim(time_span=(0., 1), rel_tol=1e-6, abs_tol=1e-6)
    sim_temp = result[body].temp(result_times)
    print('times            : ', result_times)
    source_heat = result[body].heat_src(result_times)
    print('source heat      : ', source_heat)
    print('simulated temps. : ', sim_temp)
    sim_temp = result[body].temp(1.)
    setpoint = 1-exp(-1)
    print('temperature at time 1, expected:, ', setpoint, 'calculated:', sim_temp)
    assert(allclose(sim_temp, setpoint, rtol=0, atol=1e-03)), (
        'The simulated temperature {0:.4f} not close enough to the '
        'expected result {1:.4f}!'.format(sim_temp, setpoint))
    print()


def test_delayed_step_in_heat_source():
    print('Response of a node with a delayed heat source step beginning after .2 '
          'seconds:')
    bnode.temp = 0.
    clink.cond = 1.
    hsource.heat = lambda temp: time_step.value
    node_net = Network(model)
    result_times = array([.01, .02, .05, .1, .2, .5, 1., 2.])
    result = node_net.sim(time_span=(0., result_times[-1]), rel_tol=1e-6, abs_tol=1e-6)
    source_heat = result[body].heat_src(result_times)
    sim_temps = result[body].temp(result_times)
    print('times            : ', result_times)
    print('source heat      : ', source_heat)
    print('simulated temps. : ', sim_temps)

    pde_solution = lambda time: 0. if time < .2 else 1-exp(-(time-.2))
    reference_temps = zeros(result_times.shape)
    for i, time in enumerate(result_times):
        reference_temps[i] = pde_solution(time)
    print('reference temps. : ', reference_temps)
    assert(allclose(sim_temps, reference_temps, rtol=0, atol=1e-02)), (
        'The simulated temperatures are not close enough to the '
        'expected result!')
    print()



