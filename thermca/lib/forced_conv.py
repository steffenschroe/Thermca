"""Library of functions to calculate forced convection film
coefficients on surfaces of bodies

- The functions return `FilmFunc` functions, which can be used as
  arguments in :class:`~thermca.links.FilmLink` elements.
- Units after SI standard and temperature in °C if not otherwise stated
"""

from math import tau

from numpy import where, sin, sqrt
from numpy.core.multiarray import ndarray  # for type hints

from thermca._utils.func_tools import curry
from thermca.materials import Fluid
from thermca.lib.free_conv import _bound_layer_props


@curry
def pipe(
        surf_temp: ndarray,
        fluid_temp: ndarray,
        fluid: Fluid,
        vel=.5,
        lgth=.5,
        hydr_rad=.1,
        factor=1.,
) -> ndarray:
    """Forced convection in pipes [Buc78]_

    Args:
        surf_temp: Surface temperature
        fluid_temp: Bulk fluid temperature
        fluid: Fluid
        vel: Fluid velocity
        lgth: Pipe length as characteristic length
        hydr_rad: Hydraulic radius
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    hydr_diam = 4. * hydr_rad
    condy, vol_capy, visc = _bound_layer_props(surf_temp, fluid_temp, fluid)
    prandtl = visc*vol_capy/condy
    reynolds = vel*hydr_diam/visc
    h = where(reynolds <= 2300,
              # laminar flow
              1.86*(hydr_diam/lgth*reynolds*prandtl)**.33*condy/hydr_diam
              *(visc*fluid.dens_interp((surf_temp + fluid_temp)/2.)
                 /(fluid.visc_interp(surf_temp)*fluid.dens_interp(surf_temp))
                 )**.14,
              # turbulent flow
              .0235*reynolds**.8*prandtl**.48*condy/hydr_diam)
    return h*factor
    # nusselt = alpha * surf_height / condy


@curry
def rot_cyl_in_air(
        surf_temp: ndarray = None,
        air_temp: ndarray = None,
        fluid: Fluid = None,  # for convenient use
        rot_freq=1.,
        rad=1.,
        factor=1.,
) -> float:
    """Forced convection of a rotating cylindrical surface in air
    [Jun10]_

    Args:
        surf_temp: Not used
        air_temp: Not used
        fluid: Not used
        rot_freq: Rotation frequency
        rad: Cylinder radius
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    return 3.75*rot_freq**.7*rad**.4*factor


@curry
def rot_disc_in_air(
        surf_temp: ndarray = None,
        air_temp: ndarray = None,
        fluid: Fluid = None,  # for convenient use
        rot_freq=1.,
        rad=1.,
        factor=1.,

) -> float:
    """Forced convection of a rotating disc surface in air [Jun10]_

    Args:
        surf_temp: Not used
        air_temp: Not used
        fluid: Not used
        rot_freq: Rotation frequency
        rad: Outer radius
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    return 2.73*rot_freq**.8*rad**.6*factor


@curry
def rot_cone_in_air(
        surf_temp: ndarray = None,
        air_temp: ndarray = None,
        fluid: Fluid = None,  # for convenient use
        rot_freq=1.,
        rad=1.,
        cone_angle=0,
        factor=1.,
) -> float:
    """Forced convection of a rotating cone surface in air [Jun10]_

    Args:
        surf_temp: Not used
        air_temp: Not used
        fluid: Not used
        rot_freq: Rotation frequency
        rad: Outer radius
        cone_angle: Angle from axle to cone surface in radians
            e.g. pi/2 for a disc
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    return 2.73*(rot_freq/sin(cone_angle))**.8*rad**.6*factor


@curry
def rot_cyl_air_gap(
        surf0_temp: ndarray = None,
        surf1_temp: ndarray = None,
        fluid: Fluid = None,  # for convenient use
        rot_freq=1.,
        rad=1.,
        gap=1.,
        factor=1.,
) -> float:
    """Forced convection in a rotating cylindrical air gap [Jun10]_

    Convection film coefficient that connects a pair of smooth
    opposing cylindrical surfaces with an air gap in between.
    The cylinder surface of the rotor rotates.

    Args:
        surf0_temp: Not used
        surf1_temp: Not used
        fluid: Not used
        rot_freq: Rotor rotation frequency
        rad: Gap radius
        gap: Gap width
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    return .765*(rot_freq*rad)**.666/gap**.333*factor


@curry
def plate(
        surf_temp: ndarray,
        fluid_temp: ndarray,
        fluid: Fluid,
        vel=.5,
        length=.5,
        factor=1.,
) -> ndarray:
    """Forced convection for longitudinal flow on a plane plate [Jun10]_.

    Args:
        surf_temp: Surface temperature
        fluid_temp: Bulk fluid temperature
        fluid: Fluid
        vel: Fluid velocity
        length: Surface length as characteristic length
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    condy, vol_capy, visc = _bound_layer_props(surf_temp, fluid_temp, fluid)
    prandtl = visc*vol_capy/condy
    reynolds = vel*length/visc
    # print(f'laminar {reynolds <= 3e5}')
    h = where(reynolds <= 3e5,
              .664*sqrt(reynolds)*prandtl**.33*condy/length,  # laminar flow
              .037*reynolds**.8*prandtl**.48*condy/length)  # turbulent flow
    return h*factor
    # nusselt = h * surf_height / condy

@curry
def cyl_cross_flow(
        surf_temp: ndarray,
        fluid_temp: ndarray,
        fluid: Fluid,
        vel=.5,
        rad=.5,
        factor=1.,
) -> ndarray:
    """Forced convection for transverse flow on a cylinder [Jun10]_.

    Args:
        surf_temp: Surface temperature
        fluid_temp: Bulk fluid temperature
        fluid: Fluid
        vel: Fluid velocity
        rad: Radius
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    condy, vol_capy, visc = _bound_layer_props(surf_temp, fluid_temp, fluid)
    prandtl = visc*vol_capy/condy
    length = .5*tau*rad
    reynolds = vel*length/visc
    h = where(reynolds <= 3e5,
              (.3 + .664*sqrt(reynolds)*prandtl**.33)*condy/length,  # laminar flow
              (.3 + .037*reynolds**.8*prandtl**.48)*condy/length)  # turbulent flow
    return h*factor
