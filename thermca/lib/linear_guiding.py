"""Library for heat loss and conductance functions of linear guidance
machine elements

- Library contend is located in the `linear_guiding` module.
- Units after SI standard and temperature in °C if not otherwise stated
"""

from math import tau

from numpy import exp, log, hstack, geomspace, abs as aabs
from scipy import interpolate

from thermca.lib.fluids import oil_vg68


def cont_heat_loss(
    rlr_type='ball',
    rlr_rad=0.003,
    rlr_len=0.01,
    rlr_count=3,
    load=2700.0,
    vel=(1000.0 / 60 * tau),
    lube=oil_vg68,
    sealed=True,
    factor=1.0,
):
    """Heat loss of roller guiding [Jun10]_

    Returns a `HeatFunc` function for use as argument in `HeatSource`
    elements. It takes the temperature dependent lube viscosity into
    account, which may change over simulation time. Additionally, the
    arguments `rot_freq` and `factor` may be time-dependent.

    Args:
        rlr_type: Roller type,
            may be one of the following:
            'ball 2-point contact' for ball in 2-point contact,
            'ball 4-point contact' for ball in 2-point contact on
            gothic profile,
            and 'cyl' for cylinder
            In case of ball elements, systems with 2-point contact are
            mainly used. 4-point contact can be found in compact
            guiding elements like miniature systems. Due to the four
            contact surfaces, guiding shoes with only two running
            grooves can be realised.
        rlr_rad: Diameter of Roller
        rlr_len: Length of Roller, only needed in case
            of cylinder
        rlr_count: Number of the Rollers
        load: Summed normal contact force
        vel: Velocity of the linear guidance
        lube: Lube base fluid
        sealed: If the linear guidance is sealed
        factor: Factor by which the result gets multiplied

    Returns:
        Heat loss function
    """

    # Returns a compact function of the form:
    # heat_loss = k0*rot_freq*(rot_freq*exp(k1/(temp + 95)))**k2.
    # The function parameters are calculated by a friction model of the
    # rolling contact: equation 9.49 from [Jun10]_.

    red_mod = 2.31e11  # E' reduced elastic modulus
    elem_load = load / rlr_count
    lube_visc = lube.visc_interp(40)
    # Coefficients of Vogel-Cameron viscosity model [Jun10]_ equation 4.23
    k1 = log(lube.visc_interp(20) / lube_visc) / (1 / (20 + 95) - 1 / 135)  # B
    a = lube_visc / exp(k1 / 135)  # A
    visc_press_coeff = 0.000000129 * (lube_visc**0.204)  # α'
    matl_coeff = visc_press_coeff * red_mod  # G
    # The calculated friction torque of the rolling contact gets multiplied
    # by the seal friction factor to take seal friction in to account.
    seal_factor = 2.0 if sealed else 1.0
    if rlr_type == 'ball 2-point contact':
        k2 = 0.538
        cont_load_coeff = elem_load / (red_mod * rlr_rad**2)  # W
        k0 = (
            rlr_count
            * 4.31
            * (lube.dens_interp(40) * a / (2 * red_mod * rlr_rad)) ** k2
            * matl_coeff**-0.2
            * cont_load_coeff**-0.569
            * elem_load
            * seal_factor
        )
    elif rlr_type == 'ball 4-point contact':
        k2 = 0.513
        cont_load_coeff = elem_load / (red_mod * rlr_rad**2)  # W
        k0 = (
            rlr_count
            * 20.0
            * (lube.dens_interp(40) * a / (2 * red_mod * rlr_rad)) ** k2
            * matl_coeff**-0.2
            * cont_load_coeff**-0.466
            * elem_load
            * seal_factor
        )
    elif rlr_type == 'cyl':
        k2 = 0.598
        cont_load_coeff = elem_load / (red_mod * rlr_rad * rlr_len)  # W
        k0 = (
            rlr_count
            * 85.7
            * (lube.dens_interp(40) * a / (2 * red_mod * rlr_rad)) ** k2
            * matl_coeff**-0.352
            * cont_load_coeff**-0.575
            * elem_load
            * seal_factor
        )
    else:
        raise ValueError('The given ' + rlr_type + ' is not a supported Roller type.')

    def loss_func(
        temp,
        vel=vel,
        k0=k0,
        k1=k1,
        k2=k2,
        factor=factor,
    ):
        return (k0 * vel * (vel * exp(k1 / (temp + 95))) ** k2) * factor

    return loss_func


def cont_cond(
    cont_temp=40.0,
    bear_type='angular contact ball',
    rlr_rad=0.0045,
    rlr_lgth=0.01,
    rlr_count=40,
    load=2000.0,
    vel=(2.0 / 60.0),
    lube=oil_vg68,
    factor=1.0,
):
    """Conductance of linear guides by a contact model [Jun10]_
    page 175

    Returns a `CondFunc` function, which can be used as argument in
    `CondLink` elements.
    The argument `rot_freq` may be time-dependent.

    Args:
        cont_temp: Temperature of the rolling contact
        bear_type: Roller type, may be one of the following:
            'angular contact ball' and 'spherical roller'
        rlr_rad: Roller diameter
        rlr_lgth: Roller length, not needed in case
            of 'spherical roller'
        rlr_count: Number of Rollers
        load: Load as axial outer force on the bearing
        vel: Guide velocity, may be time-dependent input
        lube: Lube base fluid
        factor: Factor by which the result gets multiplied

    Returns:
        Conductance function
    """
    max_vel = 50.0  # 3000 m/min, 180 km/h
    vel_pts = hstack([[1e-32], geomspace(0.002, max_vel, 12)])
    red_mod = 2.31e11  # E' Reduced elastic modulus
    lube_visc = lube.visc_interp(cont_temp)
    lube_dyn_visc = lube_visc * lube.dens_interp(cont_temp)  # η0
    lube_condy = lube.condy_interp(cont_temp)
    steel_condy = 32.8  # solids.bearing_steel.condy_interp(30.)
    steel_diff = steel_condy / (7850.0 * 480.0)
    speed_coeff = lube_dyn_visc * vel_pts / (2.0 * red_mod * rlr_rad)  # U
    if bear_type == 'angular contact ball':
        cont_force = load / rlr_count  # F_k
        cont_load_coeff = cont_force / (red_mod * rlr_rad**2.0)  # W
        cont_cond = (
            1.8
            * steel_condy
            * rlr_rad
            * cont_load_coeff**0.3333
            * (
                1.0
                + cont_load_coeff**0.1667 * (vel_pts * rlr_rad / steel_diff) ** 0.5
            )
        )
        film_cond = (
            0.22 * lube_condy * rlr_rad * cont_load_coeff**0.83 * speed_coeff**-0.67
        )
        filling_oil_cond = 0.05 * steel_condy * rlr_rad
    elif bear_type == 'spherical roller':
        cont_force = load / rlr_count  # F_k
        cont_load_coeff = cont_force / (red_mod * rlr_rad * rlr_lgth)  # W
        cont_cond = (
            0.32
            * steel_condy
            * rlr_lgth
            * cont_load_coeff**0.08615
            * (
                1.0
                + 1.13
                * cont_load_coeff**0.25
                * (vel_pts * rlr_rad / steel_diff) ** 0.5
            )
        )
        film_cond = (
            0.041
            * lube_condy
            * rlr_lgth
            * cont_load_coeff**0.72
            * speed_coeff**-0.7
        )
        filling_oil_cond = 0.12 * steel_condy * rlr_lgth
    else:
        raise ValueError(f"The given {bear_type} is not a supported bearing type.")
    vel_pts[0] = 0.0
    rlr_cond = cont_cond * film_cond / (2 * cont_cond + film_cond) + filling_oil_cond
    cond = rlr_count * rlr_cond * factor

    # No math. approximation function found because resulting curves differ strongly
    interp_func = interpolate.interp1d(vel_pts, cond)

    def rolling_bearing_cont_cond_func(
        temp0=None,
        temp1=None,
        matl=None,
        vel=vel,
    ):
        return interp_func(aabs(vel))

    return rolling_bearing_cont_cond_func
