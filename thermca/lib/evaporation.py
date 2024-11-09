"""Library of functions to calculate evaporation film coefficients

    - The functions return `FilmFunc` functions, which can be used as
      arguments in :class:`~thermca.links.FilmLink` elements.
    - Units after SI standard and temperature in °C if not otherwise stated
"""

from numpy import exp
from numpy.core.multiarray import ndarray  # for type hints

from thermca._utils.func_tools import curry
from thermca.materials import Fluid


@curry
def free_forc_water_to_air(
    water_temp: ndarray,
    air_temp: ndarray,
    fluid: Fluid = None,
    rel_hum=0.6,
    vel=1.0,
    factor=1.0,
) -> ndarray:
    """Free and forced evaporation from heated water bodies in to air
    [Ada90]_

    Args:
        water_temp: Bulk water temperature
        air_temp: Air temperature
        fluid: Not used
        vel: Air velocity
        rel_hum: Relative humidity in air
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """

    def sat_press(temp):
        """Saturation vapor pressure in hPa (Magnus formula)"""
        return 6.112 * exp(17.62 * temp / (243.12 + temp))

    delta_temp = water_temp - air_temp
    free_conv = 2.7 * delta_temp**0.33
    # Proposed approach seems not to work well with very small areas typical
    # in e.g. technical domains
    # forced_conv = 5.1e-4*area**-.05*vel
    # Lake Hefner approach is used for now, future investigation needed regarding
    # small areas typical in the technical field
    forced_conv = 3.8 * vel
    flux = (free_conv**2 + forced_conv**2) ** 0.5 * (
        sat_press(water_temp) - rel_hum * sat_press(air_temp)
    )
    # Resulting heat flow is driven by vapor pressure not by temperature difference
    return flux / delta_temp * factor
