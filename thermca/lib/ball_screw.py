"""Library for heat loss and conductance functions of ball screw drives

    - Library contend is located in the `ball_screw` module.
    - Units after SI standard and temperature in °C if not otherwise stated
"""

from math import tau, pi

from numpy import hstack, cos, exp, arctan, geomspace, log, sin, abs as aabs
from scipy import interpolate

from thermca.materials import Fluid
from thermca.lib.fluids import oil_vg68

SNGL_NUT = 'single nut'
DBLE_NUT = 'double nut'


def cont_loss(
    ball_screw_type=SNGL_NUT,
    nom_rad=.08 / 2.,
    pitch=.02,
    ball_rad=.003 / 2,
    cir_count=3.,
    preload=2700,
    rot_freq=1000. / 60 * tau,
    cont_ang=tau / 8,
    lube: Fluid = oil_vg68,
    sealed=True,
    factor=1.,
):
    """Heat loss of ball screw drive by a contact model [Jun10]_

    Returns a `HeatFunc` function for use as argument in `HeatSource`
    elements. It takes the temperature dependent lube viscosity into
    account, which may change over simulation time. Additionally, the
    arguments `rot_freq`, `addend` and `factor` may be time-dependent.

    Args:
        ball_screw_type: Ball screw type, may be one of the following:
            SNGL_NUT for a single-nut with balls in 4-point-contact,
            DBLE_NUT for a double-nut with balls in 2-point-contact
        nom_rad: Nominal radius of the ball screw
        pitch: Pitch of the thread
        ball_rad: Ball radius
        cir_count: Number of ball circulations
        preload: Preload as axial force
        rot_freq: Rotation frequency
        cont_ang: Contact angle
        lube: Lube base fluid
        sealed: If the ball screw is sealed

        factor: Factor by which the result gets multiplied

    Returns:
        Heat loss function
    """

    # Returns a simple function of the form:
    # heat_loss = k0*rot_freq*(rot_freq*exp(k1/(temp + 95)))**k2.
    # The function parameters are calculated by a friction model of the
    # rolling contact: equation 9.49 from [Jun10]_.

    red_mod = 2.31e11  # Reduced elastic modulus
    pitch_angle = arctan(pitch / (tau * nom_rad))
    ball_count = cir_count * nom_rad * pi / (cos(pitch_angle) * ball_rad)
    cont_force = preload / (ball_count * sin(cont_ang) * cos(pitch_angle))
    cont_load_coeff = cont_force / (red_mod * ball_rad**2)  # W
    pitch_circle_ratio = nom_rad / ball_rad
    lube_visc = lube.visc_interp(40)
    # Coefficients of Vogel-Cameron viscosity model [Jun10]_ equation 4.23
    k1 = log(lube.visc_interp(20) / lube_visc) / (1 / (20 + 95) - 1 / 135)  # B
    a = lube_visc / exp(k1 / 135)  # A
    visc_press_coeff = .000000129 * (lube_visc**.204)  # α'
    matl_coeff = visc_press_coeff * red_mod  # G
    # The calculated friction torque of the rolling contact gets multiplied
    # by the seal friction factor to take seal friction in to account.
    seal_factor = 2. if sealed else 1.
    if ball_screw_type == SNGL_NUT:
        k2 = .4936
        k0 = (
            ball_count
            * 9.16
            * matl_coeff**-.2
            * cont_load_coeff**-.5
            * (
                lube.dens_interp(40)
                * a
                * pitch_circle_ratio
                / (2 * red_mod * cos(pitch_angle))
            )
            ** k2
            * cont_force
            * nom_rad
            / cos(pitch_angle)
            * seal_factor
        )
    elif ball_screw_type == DBLE_NUT:
        k2 = .512
        k0 = (
            ball_count
            * 9.16
            * matl_coeff**-.2
            * cont_load_coeff**-.516
            * (
                lube.dens_interp(40)
                * a
                * pitch_circle_ratio
                / (2. * red_mod * cos(pitch_angle))
            )
            ** k2
            * cont_force
            * nom_rad
            / cos(pitch_angle)
            * seal_factor
        )
    else:
        raise ValueError(
            f"The given {ball_screw_type} is not a supported ball screw type."
        )

    def ball_screw_cont_loss_func(
        temp,  # ball screw temperature
        rot_freq=rot_freq,
        k0=k0,
        k1=k1,
        k2=k2,
        factor=factor,
    ):
        rot_freq = aabs(rot_freq)
        return (k0 * rot_freq * (rot_freq * exp(k1 / (temp + 95.))) ** k2) * factor

    return ball_screw_cont_loss_func


def cont_cond(
    cont_temp=40.,
    ball_screw_type=SNGL_NUT,
    nom_rad=(.08 / 2.),
    pitch=.02,
    ball_rad=(.003 / 2.),
    cir_count=3.,
    preload=2700,
    cont_ang=(tau / 8.),
    rot_freq=(100. / 60. * tau),
    lube: Fluid = oil_vg68,
    factor=1.,
):
    """Conductance of ball screw drive by a contact model [Jun10]_
    page 175

    Returns a `CondFunc` function, which can be used as argument in
    `CondLink` elements.
    The argument `rot_freq` may be time-dependent.

    Args:
        cont_temp: Temperature in the rolling contact
        ball_screw_type: Ball screw type, may be one of the following:
            SNGL_NUT for a single-nut with balls in 4-point-contact,
            DBLE_NUT for a double-nut with balls in 2-point-contact
        nom_rad: Nominal radius of the ball screw
        pitch: Pitch of the thread
        ball_rad: Ball radius
        cir_count: Number of ball circulations
        preload: Preload as axial force
        cont_ang: Contact angle
        rot_freq: Rotation frequency, may be time-dependent input
        lube: Lube base fluid
        factor: Factor by which the result gets multiplied

    Returns:
        Conductance function
    """
    max_rot_freq = 200.  # 12000 1/min
    rot_freq_pts = hstack([[1e-32], geomspace(.5, max_rot_freq, 12)])
    red_mod = 2.31e11  # E' Reduced elastic modulus
    pitch_angle = arctan(pitch / (tau * nom_rad))
    ball_count = cir_count * nom_rad * pi / (cos(pitch_angle) * ball_rad)
    cont_force = preload / (ball_count * sin(cont_ang) * cos(pitch_angle))
    lube_visc = lube.visc_interp(cont_temp)
    lube_dyn_visc = lube_visc * lube.dens_interp(cont_temp)  # η0
    lube_condy = lube.condy_interp(cont_temp)
    steel_condy = 32.8  # solids.bearing_steel.condy_interp(30.)
    steel_diff = steel_condy / (7850. * 480.)
    speed_coeff = (
        lube_dyn_visc
        * rot_freq_pts
        * nom_rad
        / (2. * red_mod * pitch_angle * ball_rad)
    )  # U
    cont_load_coeff = cont_force / (red_mod * ball_rad**2.)  # W
    cont_cond = (
        1.8
        * steel_condy
        * ball_rad
        * cont_load_coeff**.3333
        * (
            1.
            + cont_load_coeff**.1667
            * (rot_freq_pts * ball_rad * nom_rad / steel_diff) ** .5
        )
    )
    film_cond = (
        .22 * lube_condy * ball_rad * cont_load_coeff**.83 * speed_coeff**-.67
    )
    filling_oil_cond = .05 * steel_condy * ball_rad
    rot_freq_pts[0] = 0.
    if ball_screw_type == SNGL_NUT:
        rlr_cond = (
            2. * cont_cond * film_cond / (2 * cont_cond + film_cond) + filling_oil_cond
        )
    elif ball_screw_type == DBLE_NUT:
        rlr_cond = (
            cont_cond * film_cond / (2 * cont_cond + film_cond) + filling_oil_cond
        )
    else:
        raise ValueError(
            f"The given {ball_screw_type} is not a supported ball screw type."
        )
    cond = ball_count * rlr_cond * factor

    # No math. approximation function found because resulting curves differ strongly
    interp_func = interpolate.interp1d(rot_freq_pts, cond)

    def ball_screw_cont_cond_func(
        temp0=None,
        temp1=None,
        matl=None,
        rot_freq=rot_freq,
    ):
        return interp_func(aabs(rot_freq))

    return ball_screw_cont_cond_func
