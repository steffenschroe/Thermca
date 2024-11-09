"""Test transient solutions of linked LP- and FE-systems"""

import math

from numpy import allclose, genfromtxt

from thermca.lpm.asm import Asm
from thermca.lpm.cube import Cube
from thermca.lpm.cyl import Cyl, EQUAL_RAD
from thermca.lpm.asm import Surf, ForceConts
from thermca.lpm.lp_system import LPSystem
from thermca.materials import Solid
from thermca.model import Model
from thermca.network import Network
from thermca.source import HeatSource, FluxSource
from thermca.lpm.lp_part import LPPart
from thermca.pointnodes import BoundNode
from thermca.links import FilmLink
from thermca.mesh import Mesh
from thermca.fem.fe_part import FEPart


def test_linked_lp_and_fe_part():
    """Test linked LP- and FE-part with transient 3d heat flow

    Test steel plate modelled as two connected cuboids of different
    type. One modelled as finite element based part and one as lumped
    parameter based block.

    The plate has the dimensions of 0.5 x 1.0 x 0.05 in m. The initial
    temperature is 20.0 °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """

    steel = Solid(
        dens=8000.,
        spec_heat=500.,
        condy=50.)

    # half_flat_cuboid: .5x.5x.05; body: 'cuboid', surfs: 'bottom', 'middel', 'top'
    half_cuboid = Mesh.read('half_flat_cuboid.med')

    with Asm() as asm:
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
            ]
        )
        Surf(name='btm', faces=[block.face.btm])

    with Model() as model:
        lower_cuboid = FEPart(
            half_cuboid, steel,
            init_temp=20.,
            mor_dof=10,
            name='lower_cuboid')
        upper_cuboid = LPPart(
            posn=(0., .5, 0.),
            asm=asm,
            matl=steel,
            init_temp=20.,
            name='upper_cuboid'
        )
        env = BoundNode(
            temp=20.,
            posn=(-.3, .5, .025),
            name='environment')
        HeatSource(lower_cuboid.surf.bottom, 1000.)
        FilmLink.multi(
            [lower_cuboid.surf.middel, upper_cuboid.surf.upper_faces],
            env,
            10.)
        FilmLink(lower_cuboid.surf.top, upper_cuboid.surf.btm, 1000.)

    net = Network(model)

    # Ansys 12800 quadratic and regular hexahedron elements
    # dt_0 = 1e-2, dt_min = 1e-3, dt_max = 20
    ansys_res = genfromtxt('cuboid flat ansys.csv', delimiter=';', names=True, case_sensitive=True)
    # Use 10 points in time from Ansys reference time data
    skip = math.ceil(len(ansys_res['Time']) / 10)
    ansys_res = ansys_res[::skip]
    ref_time = ansys_res['Time']
    ref_bottom = ansys_res['Bottom']

    res = net.sim(
        [0., ref_time[-1]],
    )
    sim_bottom = res[lower_cuboid.surf.bottom].temp(ref_time)
    assert allclose(ref_bottom, sim_bottom, rtol=0, atol=2.)


def test_parts_with_unequal_heat_sources_on_connected_surfaces():
    """Transient radial heat flow over connected LP- and FE-parts

    The test aims at the thermal connection of unequal surface areas
    by film coefficients and additionally with unequal heat sources on
    the connected surfaces of different element types.

    The parts are short concentrically arranged steel tubes. The inner
    tube has an inner radius of 0.01 and an outer radius of 0.05. The
    outer tube has an inner radius of 0.06 and an outer radius of 1.0.
    The end surfaces of the tubes are adiabatic. The connection between
    the outer and inner tube is modelled with a film coefficient of
    500. The heat flow is driven by unequal heat sources on the gap
    surfaces (150. on inner tube outer surface, 250. on outer tube
    inner surface). The outer surface of the outer ring is connected
    to temperature bound node with a film coefficient of 50. The
    initial temperature is 20.
    """
    inner_source_ref_res = genfromtxt('rings_with_gap_sources.csv',  delimiter=',', names=True)
    ref_times = inner_source_ref_res['time']
    ref_outer_tube_outer = inner_source_ref_res['outer_tube_outer']
    outer_mesh = Mesh.read('outer_tube.med')
    init_temp = 20.
    gap_film = 500.
    outer_film = 50.

    steel = Solid(
        condy=25.,
        dens=8000.,
        spec_heat=500.)

    with Asm() as asm:
        inner_cyl = Cyl(
            inner_rad=.01,
            rad_div=7,
            outer_rad=.05,
            lgth=.05,
            name='inner_cyl'
        )
        Surf(
            name='inner',
            faces=[inner_cyl.face.inner],
        )
        Surf(
            name='outer',
            faces=[inner_cyl.face.outer],
        )

    with Model() as fem_model:
        inner_tube = LPPart(
            asm=asm,
            matl=steel,
            init_temp=init_temp,
            name='inner_tube',
        )
        outer_tube = FEPart(
            mesh=outer_mesh,
            matl=steel,
            init_temp=init_temp,
            name='outer_tube',
        )
        env = BoundNode(
            temp=init_temp,
            posn=(0., .13, .025),
            name='environment')
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
    # print(f"{ref_outer_tube_outer=}, \n{outer_tube_outer_temps=}")
    # print("dtemp_max: ", max(abs(ref_outer_tube_outer-outer_tube_outer_temps)))
    num_temps = len(ref_outer_tube_outer)
    assert allclose(ref_outer_tube_outer, outer_tube_outer_temps, rtol=0., atol=5.)
    assert allclose(
        (ref_outer_tube_outer - init_temp)[num_temps // 5:],
        (outer_tube_outer_temps - init_temp)[num_temps // 5:],
        rtol=.05, atol=0.
    )

