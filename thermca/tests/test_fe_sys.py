"""Test fe_part implementation"""

from dataclasses import dataclass
from numpy import allclose
from thermca import *
from thermca.fem.fe_system import FESystem
from thermca.static_bcs import HeatBC, FilmBC, TempBC

steel = Solid(
    dens=8000.,
    spec_heat=500.,
    condy=50.,
)


@dataclass
class Surf:
    name: str


def test_stationary_solution_with_heat_and_film_bcs():
    cuboid_mesh = Mesh.read("flat_cuboid_fine.med")
    cuboid = FESystem(
        body_mesh=cuboid_mesh.extract_type('tetra'),
        surf_meshes=cuboid_mesh.extract_type('triangle'),
        init_temp=20.,
        matl=steel,
        lump_cond_meth=None,
    )
    bcs = [
        HeatBC(Surf('bottom'), 1000.),
        FilmBC(Surf('upper_faces'), film=10., env_temp=20.)
    ]
    temps = cuboid.stationary_solution(bcs)
    mean_surf_temps = cuboid.C_surf.T @ temps
    assert allclose([291.1152, 108.8888], mean_surf_temps, rtol=0, atol=0.01)


def test_stationary_solution_with_heat_and_temp_bcs():
    # Conductance (L) measuring on cuboid from left to right surface
    # Heat flow (Q) of 1. on left surface and fixed temperature (T)
    # of 0. on right surface; L = Q/|T_left - T_right)
    # Reference: L = λ*A/l, conductivity λ, Surface, cross-sectional
    # area A, length in heat flow direction l
    # cuboid lengt in x = 1. m, height in y = .5 m, depth in z = .25 m
    # surfaces: 'left', 'right', 'bottom', 'top', 'front', 'back'
    lgth, hgt, depth = 1., .5, .25
    cuboid_mesh = Mesh.read("cuboid.med")
    matl = Solid(condy=2., spec_heat=1., dens=2., )
    cuboid = FESystem(
        body_mesh=cuboid_mesh.extract_type('tetra'),
        surf_meshes=cuboid_mesh.extract_type('triangle'),
        init_temp=0.,
        matl=matl,
        lump_cond_meth=None,
    )
    bcs = [
        HeatBC(Surf('left'), 1.),
        TempBC(Surf('right'), temp=0.)
    ]
    temps = cuboid.stationary_solution(bcs)
    mean_surf_temps = cuboid.C_surf.T @ temps
    # print(mean_surf_temps)
    left_temp = mean_surf_temps[cuboid.surf_names.index('left')]
    cond = 1. / left_temp  # conductivity
    cond_left_to_right_ref = matl.condy_interp(0.) * hgt*depth/lgth
    # print([cond_left_to_right_ref], [cond])
    assert allclose([cond_left_to_right_ref], [cond], rtol=0, atol=.01)


def test_stationary_solution_of_asymmetrical_heat_flow_in_hollow_cylinder():
    """Test heat flow through hollow cylinder in steady state.

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
    cyl_mesh = Mesh.read('constr_cyl.msh')
    # cyl_mesh.points /= 1000
    # cyl_mesh.write('cylinder_coarse.med')
    # cyl_mesh.plot(dpi=250).show()
    test_matl = Solid(
        condy=1.,
        dens=1.,
        spec_heat=1.,
    )
    cyl = FESystem(
        body_mesh=cyl_mesh.extract_type('tetra'),
        surf_meshes=cyl_mesh.extract_type('triangle'),
        init_temp=0.,
        matl=test_matl,
        lump_cond_meth=None,
    )

    # First: heat flow in axial direction
    bcs = [
        HeatBC(Surf('base'), 1.),
        TempBC(Surf('half_end'), temp=0.)
    ]
    temps = cyl.stationary_solution(bcs)
    mean_surf_temps = cyl.C_surf.T @ temps
    base_temp = mean_surf_temps[cyl.surf_names.index('base')]
    measured_cond = 1. / base_temp  # conductivity
    ref_cond = 2.0455  # LP model at 512x512 spatial subdivision
    # print('calculated conductance:   ', measured_cond)
    # print('reference conductance:   ', ref_cond)
    assert allclose(measured_cond, ref_cond, rtol=.07, atol=0.), (
        'The simulated conductance is not close enough to the ' 'expected result!'
    )

    # Second: heat flow in radial direction
    bcs = [
        HeatBC(Surf('outer'), 1.),
        TempBC(Surf('half_inner'), temp=0.)
    ]
    temps = cyl.stationary_solution(bcs)
    mean_surf_temps = cyl.C_surf.T @ temps
    outer_mean_temp = mean_surf_temps[cyl.surf_names.index('outer')]
    measured_cond = 1. / outer_mean_temp  # conductivity
    ref_cond = 6.0733  # LP model at 512x512 spatial subdivision
    # print('calculated conductance:   ', measured_cond)
    # print('reference conductance:   ', ref_cond)
    assert allclose(measured_cond, ref_cond, rtol=.07, atol=0.), (
        'The simulated conductance is not close enough to the ' 'expected result!'
    )
