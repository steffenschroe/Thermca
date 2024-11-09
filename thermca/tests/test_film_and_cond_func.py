"""Test transient solutions with film and conductance functions"""

import math

import pytest
from numpy import allclose, genfromtxt

from thermca.lpm.asm import Asm
from thermca.lpm.cube import Cube
from thermca.lpm.asm import Surf
from thermca.materials import Solid, Fluid
from thermca.model import Model
from thermca.network import Network
from thermca.source import HeatSource
from thermca.lpm.lp_part import LPPart
from thermca.pointnodes import BoundNode
from thermca.links import FilmLink, CondLink


def test_film_two_identcal_funcs():
    """Test film given by two identical functions

    Test on steel plate with transient three-dimensional heat flow.
    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """

    test_air = Fluid(dens=1., spec_heat=1.5, condy=1.5, visc=2.)

    def film_5(surf_temp, surround_temp, fluid):
        """Gets called in every solution step"""
        films = (
            fluid.vol_capy_interp(surf_temp)
            + fluid.condy_interp(surround_temp)
            + fluid.visc_interp(surf_temp)
        )  # Shoud create a vector filled with 5.
        return films

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
        env = BoundNode(
            temp=20.,
            posn=(-.3, .5, .025),
            name='environment'
        )
        FilmLink(cuboid.surf.upper_faces, env, film_5, matl=test_air)
        env2 = BoundNode(
            temp=20.,
            posn=(-.3, .5, .025),
            name='environment'
        )
        FilmLink(cuboid.surf.upper_faces, env2, film_5, matl=test_air)
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
    assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.008)
    assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.03)


def test_film_two_different_funcs():
    """Test film given by two different functions

    Test on steel plate with transient three-dimensional heat flow.
    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """

    test_air = Fluid(dens=1., spec_heat=1.5, condy=1.5, visc=2.)

    def film_5_0(surf_temp, surround_temp, fluid):
        """Gets called in every solution step"""
        films = (
            fluid.vol_capy_interp(surf_temp)
            + fluid.condy_interp(surround_temp)
            + fluid.visc_interp(surf_temp)
        )  # Shoud create a vector filled with 5.
        return films

    def film_5_1(surf_temp, surround_temp, fluid):
        """Gets called in every solution step"""
        films = (
            fluid.vol_capy_interp(surf_temp)
            + fluid.condy_interp(surround_temp)
            + fluid.visc_interp(surf_temp)
        )  # Shoud create a vector filled with 5.
        return films

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
        env = BoundNode(
            temp=20.,
            posn=(-.3, .5, .025),
            name='environment'
        )
        FilmLink(cuboid.surf.upper_faces, env, film_5_0, matl=test_air)
        env2 = BoundNode(
            temp=20.,
            posn=(-.3, .5, .025),
            name='environment'
        )
        FilmLink(cuboid.surf.upper_faces, env2, film_5_1, matl=test_air)
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
    assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.008)
    assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.03)


def test_cond_two_identcal_funcs():
    """Test conductance given by two identical functions

    Test on steel plate with transient three-dimensional heat flow.
    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """

    test_air = Fluid(dens=1., spec_heat=1.5, condy=1.5, visc=2.)

    def cond_5(surf_temp, surround_temp, fluid):
        """Gets called in every solution step"""
        films = (
            fluid.vol_capy_interp(surf_temp)
            + fluid.condy_interp(surround_temp)
            + fluid.visc_interp(surf_temp)
        )  # Shoud create a vector filled with 5.
        return films * 1.125  # film * area

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
        env = BoundNode(
            temp=20.,
            posn=(-.3, .5, .025),
            name='environment'
        )
        CondLink(cuboid.surf.upper_faces, env, cond_5, matl=test_air)
        env2 = BoundNode(
            temp=20.,
            posn=(-.3, .5, .025),
            name='environment'
        )
        CondLink(cuboid.surf.upper_faces, env2, cond_5, matl=test_air)
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
    assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.008)
    assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.03)


def test_cond_two_different_funcs():
    """Test conductance given by two different functions

    Test on steel plate with transient three-dimensional heat flow.
    The plate has the dimensions of .5 x 1. x .05 in m. The initial
    temperature is 20. °C. The heat flow is caused by 1 kW heat input
    at the bottom. The other surfaces are coupled with a HTC of
    10 W/(m²K) to environment at 20 °C.
    """

    test_air = Fluid(dens=1., spec_heat=1.5, condy=1.5, visc=2.)

    def cond_0(surf_temp, surround_temp, fluid):
        """Gets called in every solution step"""
        films = (
                fluid.vol_capy_interp(surf_temp)
                + fluid.condy_interp(surround_temp)
                + fluid.visc_interp(surf_temp)
        )  # Shoud create a vector filled with 5.
        return films * 1.125  # film * area

    def cond_1(surf_temp, surround_temp, fluid):
        """Gets called in every solution step"""
        films = (
                fluid.vol_capy_interp(surf_temp)
                + fluid.condy_interp(surround_temp)
                + fluid.visc_interp(surf_temp)
        )  # Shoud create a vector filled with 5.
        return films * 1.125  # film * area

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
        env = BoundNode(
            temp=20.,
            posn=(-.3, .5, .025),
            name='environment'
        )
        CondLink(cuboid.surf.upper_faces, env, cond_0, matl=test_air)
        env2 = BoundNode(
            temp=20.,
            posn=(-.3, .5, .025),
            name='environment'
        )
        CondLink(cuboid.surf.upper_faces, env2, cond_1, matl=test_air)
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
    assert allclose(ref_top_vert - init_temp, sim_top_vert - init_temp, rtol=.008)
    assert allclose(ref_btm_vert - init_temp, sim_btm_vert - init_temp, rtol=.03)


def rtol(sim, ref):
    return abs(ref - sim) / abs(ref)
