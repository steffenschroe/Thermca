"""Library of functions to calculate free convection film coefficients
on surfaces of bodies

- The functions return `FilmFunc` functions, which can be used as
  arguments in :class:`~thermca.links.FilmLink` elements.
- Units after SI standard and temperature in °C if not otherwise stated
"""
from typing import Tuple

from numpy import abs, where, sqrt, log, empty_like
from numpy import ndarray  # for type hints

from thermca._utils.func_tools import curry
from thermca.materials import Fluid
from thermca.lib.fluids import air


def _bound_layer_props(
        surf_temp: ndarray,
        fluid_temp: ndarray,
        fluid: Fluid
) -> Tuple[ndarray, ndarray, ndarray]:
    """Helper function to calculate the temperature dependent fluid
     properties of the convection boundary layer.

     Args:
         surf_temp: Surface temperature
         fluid_temp: Bulk fluid temperature
         fluid: Fluid
    """
    bound_layer_temp = (surf_temp + fluid_temp)/2.
    condy = fluid.condy_interp(bound_layer_temp)
    vol_capy = fluid.vol_capy_interp(bound_layer_temp)
    visc = fluid.visc_interp(bound_layer_temp)
    return condy, vol_capy, visc


def _gap_bound_layer_props(
        bound_layer_temp: ndarray,
        fluid: Fluid
) -> Tuple[ndarray, ndarray, ndarray]:
    """Helper function to calculate the temperature dependent fluid
     properties of the convection boundary layer in gaps.

     Args:
         surf_temp: Surface temperature
         fluid_temp: Bulk fluid temperature
         fluid: Fluid
    """
    condy = fluid.condy_interp(bound_layer_temp)
    vol_capy = fluid.vol_capy_interp(bound_layer_temp)
    visc = fluid.visc_interp(bound_layer_temp)
    return condy, vol_capy, visc


@curry
def vert_surf(
        surf_temp: ndarray,
        fluid_temp: ndarray,
        fluid: Fluid,
        surf_hgt=.5,
        factor=1.,
) -> ndarray:
    """Free convection on a vertical surface [Jun00]_

    Args:
        surf_temp: Surface temperature
        fluid_temp: Bulk fluid temperature
        fluid: Fluid
        surf_hgt: Surface height
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    condy, vol_capy, visc = _bound_layer_props(surf_temp, fluid_temp, fluid)
    prandtl = visc*vol_capy/condy
    temp_diff = surf_temp - fluid_temp
    grashof = 9.81 * abs(temp_diff) * surf_hgt ** 3 / ((273.15 + fluid_temp) * visc ** 2)
    rayleigh = prandtl*grashof
    # TODO: may be handle each link separate because now sublinks of one link can have different film coefficients
    h = where(rayleigh <= 5.e8,
              .55 * rayleigh ** .25 * condy / surf_hgt,
              .14 * rayleigh ** .33 * condy / surf_hgt)
    # nusselt = alpha * surf_height / condy
    return h*factor


@curry
def horiz_top_surf(
        surf_temp: ndarray,
        fluid_temp: ndarray,
        fluid: Fluid,
        surf_width=.5,
        factor=1.,
) -> ndarray:
    """Free convection on a horizontal top surface [Jun10]_

    Args:
        surf_temp: Surface temperature
        fluid_temp: Bulk fluid temperature
        fluid: Fluid
        surf_width: Surface width
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    half_surf_width = surf_width / 2  # Characteristic length
    condy, vol_capy, visc = _bound_layer_props(surf_temp, fluid_temp, fluid)
    prandtl = visc*vol_capy/condy
    temp_diff = surf_temp - fluid_temp
    grashof = 9.81 * abs(temp_diff) * half_surf_width ** 3 / ((273.15 + fluid_temp) * visc ** 2)
    rayleigh = prandtl*grashof

    h = where(
        rayleigh > 1.6e5,
        .14 * rayleigh ** .33 * condy / half_surf_width,
        where(
            rayleigh < 3e2,
            1.02 * rayleigh ** .125 * condy / half_surf_width,
            .5 * rayleigh ** .25 * condy / half_surf_width
        )
    )
    # nusselt = alpha * surf_height / condy
    return h*factor


@curry
def vert_gap(
        surf0_temp: ndarray,
        surf1_temp: ndarray,
        fluid=None,
        gap_fluid: Fluid = air,
        width=.1,
        hgt=1.,
        factor=1.,
):
    """Free convection in a vertical gap [VDI10]_

    Natural convection heat transfer in a closed vertical gap.

    Args:
        surf0_temp: Surface temperature
        surf1_temp: Surface temperature
        fluid: Not used
        gap_fluid: Fluid in the vertical enclosure
        width: Gap width
        hgt: Gap height
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    # TODO: test
    bound_layer_temp = (surf0_temp + surf1_temp) / 2.
    condy, vol_capy, visc = _gap_bound_layer_props(bound_layer_temp, gap_fluid)
    prandtl = visc*vol_capy/condy
    temp_diff = surf0_temp - surf1_temp
    grashof = 9.81*abs(temp_diff)*width**3/((273.15 + bound_layer_temp)*visc**2)
    rayleigh = prandtl*grashof
    nusselt = where(
        rayleigh < 1e7,
        .42*prandtl**.012*rayleigh**.25*(hgt/width)**-.25,
        .049*rayleigh**.33
    )
    h = where(
        nusselt >= 1.,
        nusselt*condy/width,
        condy/width  # heat conduction
    )
    return h*factor


@curry
def horiz_gap(
        surf0_temp: ndarray,
        surf1_temp: ndarray,
        fluid=None,  # for convenient use
        gap_fluid: Fluid = air,
        hgt=1.,
        factor=1.,
):
    """Free convection in a horizontal gap [VDI10]_

    Natural convection heat transfer in a closed horizontal gap.

    Args:
        surf0_temp: Surface temperature
        surf1_temp: Surface temperature
        fluid: Not used
        gap_fluid: Fluid in the horizontal enclosure
        hgt: Gap height
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    # TODO: test
    bound_layer_temp = (surf0_temp + surf1_temp) / 2.
    condy, vol_capy, visc = _gap_bound_layer_props(bound_layer_temp, gap_fluid)
    prandtl = visc*vol_capy/condy
    temp_diff = surf0_temp - surf1_temp
    grashof = 9.81*abs(temp_diff)*hgt**3/((273.15 + bound_layer_temp)*visc**2)
    rayleigh = prandtl*grashof
    nusselt = where(
        rayleigh < 2.2e4,
        .208*rayleigh**.25,
        .092*rayleigh**.33
    )
    h = where(
        rayleigh > 1708.,
        nusselt*condy/hgt,
        condy/hgt  # heat conduction
    )
    return h*factor


@curry
def horiz_cyl_gap(
        surf0_temp: ndarray,
        surf1_temp: ndarray,
        fluid=None,  # for convenient use
        gap_fluid: Fluid = air,
        inr_rad=.1,
        out_rad=.2,
        factor=1.,
):
    """Free convection in a horizontal cylindrical gap [Ito70]_

    Natural convection heat transfer in horizontal cylindrical annuli.
    Existing correlations, are made for heat transfer from inner to
    outer cylinder surfaces [VDI10]_. For the other direction there are
    no correlations found.
    Here, the correlation from [Ito70]_ is used and the simplified
    assumption is made, that the correlation is valid for heat transfer
    in both directions.

    Args:
        surf0_temp: Surface temperature
        surf1_temp: Surface temperature
        fluid: Not used
        gap_fluid: Fluid in the cylindrical enclosure
        inr_rad: Inner gap radius
        out_rad: Outer gap radius
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    bound_layer_temp = (surf0_temp + surf1_temp) / 2.
    condy, vol_capy, visc = _gap_bound_layer_props(bound_layer_temp, gap_fluid)
    prandtl = visc*vol_capy/condy
    temp_diff = surf0_temp - surf1_temp
    char_width = sqrt(out_rad*inr_rad)*log(out_rad/inr_rad)
    grashof = 9.81*abs(temp_diff)*char_width**3/((273.15 + bound_layer_temp)*visc**2)
    rayleigh = prandtl*grashof
    nusselt = .2*rayleigh**.25*sqrt(out_rad/inr_rad)

    h = where(
        nusselt > 1.,
        nusselt*condy/char_width,
        condy/char_width,  # heat conduction
    )
    return h * factor


@curry
def vert_cyl_gap(
        surf0_temp: ndarray,
        surf1_temp: ndarray,
        fluid=None,  # for convenient use
        gap_fluid: Fluid = air,
        inr_rad=.1,
        out_rad=.2,
        hgt=1.,
        factor=1.,
):
    """Free convection in a vertical cylindrical gap [VDI10]_

    Natural convection heat transfer in a vertical cylindrical annuli.
    Existing correlations are made for heat transfer from inner to
    outer cylinder surfaces [VDI10]_. For the other direction there are
    no correlations found.
    Here, the correlation from [VDI10]_ is used and the simplified
    assumption is made, that the correlation is valid for heat transfer
    in both directions.

    Args:
        surf0_temp: Surface temperature
        surf1_temp: Surface temperature
        fluid: Not used
        gap_fluid: Fluid in the cylindrical enclosure
        inr_rad: Inner gap radius
        out_rad: Outer gap radius
        hgt: Gap height
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    # TODO: test
    bound_layer_temp = (surf0_temp + surf1_temp)/2.
    condy, vol_capy, visc = _gap_bound_layer_props(bound_layer_temp, gap_fluid)
    prandtl = visc*vol_capy/condy
    temp_diff = surf0_temp - surf1_temp
    char_width = sqrt(out_rad * inr_rad)*log(out_rad/inr_rad)
    grashof = 9.81*abs(temp_diff)*char_width**3/((273.15 + bound_layer_temp)*visc**2)
    rayleigh = prandtl*grashof

    n = (rayleigh*(hgt/char_width)**3)**-.25*(hgt/inr_rad)
    c1 = empty_like(n)
    c2 = empty_like(n)
    n1 = empty_like(n)
    n2 = empty_like(n)
    # Good speed only with numba
    for i in range(len(n)):
        if n[i] < .2:
            c1[i], c2[i], n1[i], n2[i] = .48, 854., .75, 0.
        elif n[i] < 1.48:
            c1[i], c2[i], n1[i], n2[i] = .93, 1646., .84, .36
        else:
            c1[i], c2[i], n1[i], n2[i] = .49, 862., .95, .8

    nusselt = ((c1*rayleigh*(hgt/char_width)**2)
               /(c2*(hgt/out_rad)**4*(inr_rad/hgt)
                  + (rayleigh*(hgt/char_width)**3)**n1*(inr_rad/hgt)**n2)
               )
    h = where(
        nusselt > 1.,
        nusselt*condy/char_width,
        condy/char_width,  # Heat conduction
    )
    return h*factor


@curry
def horiz_cyl(
        surf_temp,
        air_temp,
        fluid=None,  # For convenient use
        rad=.5,
        factor=1.,
):
    """Free convection on a horizontal cylinder in air [Jun10]_

    Args:
        surf_temp: Surface temperature
        air_temp: Bulk fluid (air) temperature
        fluid: Not used
        rad: Cylinder radius
        factor: Factor by which the result gets multiplied

    Returns:
        Film coefficients
    """
    temp_diff = abs(surf_temp - air_temp)
    return (1.34*temp_diff**.33 + .164*(temp_diff/(rad*2)**7)**.1) * factor




