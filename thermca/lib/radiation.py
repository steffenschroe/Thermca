"""Functions to calculate radiation film coefficients on surfaces

- The functions return `FilmFunc` functions, which can be used as
  arguments in :class:`~thermca.links.FilmLink` elements.
- Units after SI standard and temperature in °C if not otherwise stated
"""
from __future__ import annotations
from typing import TypeVar

from numpy.core.multiarray import ndarray  # for type hints

from thermca._utils.func_tools import curry
from thermca.materials import Fluid

ScalarOrArray = TypeVar('ScalarOrArray', float, ndarray)


def res_emis(
    emis0: ScalarOrArray,
    emis1: ScalarOrArray,
    view0: ScalarOrArray,
    view1: ScalarOrArray,
) -> ScalarOrArray:

    """Resulting emissivity of two surfaces in thermal radiation
    exchange

    Hint:
        `emis0*emis1` gives a rough approximation of the lower limit of
         the resulting emissivity. In case of high emissivities
         (typically on electrically non-conducting surfaces) the error
         is low [Jun10]_.

    Args:
        emis0: Emissivity of surface 0
        emis1: Emissivity of surface 1
        view0: View factor of surface 0 viewing to surface 1
        view1: View factor of surface 1 viewing to surface 0

    Returns:
        Resulting emissivity of two surface emissivities in
        thermal radiation exchange
    """
    return emis0 * emis1 / (1. - (1. - emis0) * (1. - emis1) * view0 * view1)


@curry
def therm_radn(
    surf0_temp: ndarray,
    surf1_temp: ndarray,
    fluid: Fluid = None,
    view=1.,
    res_emis=.95,
    factor=1.,
) -> ndarray:
    """Thermal radiation between two surfaces

    Args:
        surf0_temp: Temperature of first surface
        surf1_temp: Temperature of second surface
        fluid: Not used
        view: View factor of surface 0 viewing to surface 1
        res_emis: Resulting emissivity of two surface emissivities in
            thermal radiation exchange
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    abs_surf0_temp, abs_surf1_temp = surf0_temp + 273.15, surf1_temp + 273.15
    # Prevent zero division in the temperature function
    # (x**4 - y**4)/(x - y) -> (x**2 + y**2)*(x + y)
    return (
        5.6703e-8
        * view
        * res_emis
        * (abs_surf0_temp**2 + abs_surf1_temp**2)
        * (abs_surf0_temp + abs_surf1_temp)
    ) * factor


@curry
def therm_radn_room(
    surf_temp: ndarray,
    surroundings_temp: ndarray,
    fluid: Fluid = None,
    view=1.,
    emis_coef=.95,
    factor=1.,
) -> ndarray:
    """Thermal radiation between surfaces and room environment

    It contains a temperature dependent approximation that is very good
    with both temperatures in the range of 0°C to 40°C and has small
    deviations from 40°C to 100°C.
    The resulting emissivity of the surface and the
    surrounding surfaces is assumed as the factor of the two
    emissivities, with surroundings at .8. This approximation is very
    good with small and high emissivities. This fits the real world,
    because typically either small or high emissivities occur.

    Args:
        surf_temp: Temperature of first surface
        surroundings_temp: Temperature of surrounding air and
            surrounding surfaces in radiation exchange
        fluid: Not used
        view: View factor of surface 0 viewing to surface 1
        emis_coef: Emission coefficient of the surface
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    # Cpython constant folding: expression parts with side by side standing constant
    # numbers are preevaluated by the byte code compiler
    return (
        view
        * emis_coef
        * (.8 * 5.6703e-8 * 4.)
        * ((surf_temp + surroundings_temp) / 2. + 273.15) ** 3.
    ) * factor
