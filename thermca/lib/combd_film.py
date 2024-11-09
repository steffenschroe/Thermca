"""Library of functions to calculate combined film coefficients
of different heat transfer mechanisms on surfaces of bodies

    - The functions return `FilmFunc` functions, which can be used as
      arguments in :class:`~thermca.links.FilmLink` elements.
    - Units after SI standard and temperature in °C if not otherwise stated
"""

from numpy import abs, add as aadd, subtract as asub

from thermca._utils.func_tools import curry

# Flow directions of mixed convection streams
ASSISTING = 'assisting'
OPPOSING = 'opposing'
TRANSVERSE = 'transverse'


def mix_conv(film0, film1, flow=ASSISTING):
    """Mixed free and forced convection

    This is a common approach to describe combined acting of forced
    and natural convection (mixed convection). It distinguishes
    between assisting, opposing and transverse convection flows.

    The heat transfer coefficient is combined as follows:
    h = (h_free**n ± h_forced**n)**(1/n)
    The free and forced convection part is added (+) in case of
    assisting and transverse flows and subtracted (-) in case of
    opposing flows.

    Args:
        film0: Film function or scalar
        film1: Film function or scalar
        flow: Flow directions may be ASSISTING, OPPOSING
            or TRANSVERSE

    Returns:
        Film function or scalar
    """

    if flow == ASSISTING or flow == OPPOSING:
        exponent = 3
        if flow == ASSISTING:
            operator_func = aadd
        else:
            operator_func = asub
    elif flow == TRANSVERSE:
        exponent = 4
        operator_func = aadd
    else:
        raise ValueError(
            "Value of flow argument must be ASSISTING, " "OPPOSING or TRANSVERSE"
        )

    def mix_film_func_func(surf_temp, fluid_temp, fluid):
        return abs(operator_func(
            film0(surf_temp, fluid_temp, fluid) ** exponent,
            film1(surf_temp, fluid_temp, fluid) ** exponent,
        )) ** (1 / exponent)

    def mix_film_func_scalar(surf_temp, fluid_temp, fluid):
        return abs(operator_func(
            film0(surf_temp, fluid_temp, fluid) ** exponent, film1**exponent
        )) ** (1 / exponent)

    def mix_film_scalar_func(surf_temp, fluid_temp, fluid):
        return abs(operator_func(
            film0**exponent, film1(surf_temp, fluid_temp, fluid) ** exponent
        )) ** (1 / exponent)

    if callable(film0):
        if callable(film1):
            return mix_film_func_func
        else:
            return mix_film_func_scalar
    else:
        if callable(film1):
            return mix_film_scalar_func
        else:
            return float(operator_func(film0**exponent, film1**exponent)) ** (
                1 / exponent
            )


@curry
def conv_radn_room_at_room_temps(
    surf_temp,
    surroundings_temp,
    fluid=None,  # for convenient use
    emis_coef=.95,
    factor=1.,
):
    """Heat exchange between a surface and surrounding room at room
    temperatures.

    Heat exchange between a surface and an enclosed air filled room
    as surrounding.
    The heat exchange includes temperature dependent free convection
    derived from vertical surface models. The model for turbulent flow
    is used. This is a rough estimation and gives good results for
    surface heights above 1 m and shows higher deviations at small
    heights. A slight part of forced convection is assumed because
    of moving people, fan heaters and other things moving the air
    [Jun10]_. It is modeled as an 30% increase of free convection.
    Heat radiation is assumed as constant. This is a good
    approximation with both temperatures in the range of 15°C to 25°C
    and only a very rough approximation with both temperatures in the
    range of 0°C to 40°C. The resulting emissivity of the surface and
    the surrounding surfaces is assumed as the factor of the two
    emissivities, with surroundings at .8. This approximation is very
    good with small and high emissivities. This fits the real world,
    because typically either small or high emissivities occur.

    Args:
        surf_temp: Surface temperature
        surroundings_temp: Temperature of surrounding air and
            surrounding surfaces in radiation exchange
        fluid: Not used
        emis_coef: Emission coefficient of the surface
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    return (
        emis_coef * (.8 * 1e8 * 5.6703e-8)  # radiation
        + (1.7 * 1.3) * abs(surf_temp - surroundings_temp) ** .33  # convection
    ) * factor


@curry
def conv_radn_room(
    surf_temp,
    surroundings_temp,
    fluid=None,  # for convenient use
    emis_coef=.95,
    factor=1.,
):
    """Heat exchange between a surface and surrounding room.

    Heat exchange between a surface and an enclosed air filled room
    as surrounding.
    The heat exchange includes temperature dependent free convection
    derived from vertical surface models. The model for turbulent flow
    is used. This is a rough estimation and gives good results for
    surface heights above 1 m and shows higher deviations at small
    heights. A slight part of forced convection is assumed because
    of moving people, fan heaters and other things moving the air
    [Jun10]_. It is modeled as an 30% increase of free convection.
    Heat radiation is modeled with a temperature dependent
    approximation. It has a very good validity with both
    temperatures in the range of 0°C to 40°C and small deviations from
    40°C to 100°C. The resulting emissivity of the surface and the
    surrounding surfaces is assumed as the factor of the two
    emissivities, with surroundings at .8. This approximation is very
    good with small and high emissivities. This fits the real world,
    because typically either small or high emissivities occur.

    Args:
        surf_temp: Surface temperature
        surroundings_temp: Temperature of surrounding air and
            surrounding surfaces in radiation exchange
        fluid: Not used
        emis_coef: Emission coefficient of the surface
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    # CPython constant folding: expression parts with side by side standing constant
    # numbers are preevaluated by the byte code compiler
    return (
        emis_coef * (.8 * 5.6703e-8 * 4.) * ((surf_temp + surroundings_temp) / 2 + 273.15) ** 3  # Radiation
        + (1.7 * 1.3) * abs(surf_temp - surroundings_temp) ** .33  # Free and forced convection
    ) * factor
