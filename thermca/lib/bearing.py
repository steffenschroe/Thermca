"""Library for heat loss and conductance functions of bearings

- Library contend is located in the `bearing` module.
- Units after SI standard and temperature in °C if not otherwise stated
"""

from math import tau

from numpy import exp, log, sqrt, sin, hstack, geomspace, abs as aabs
from scipy import interpolate

from thermca.lib.fluids import oil_vg68


def heat_loss(
        bear_type='grooved ball',
        mean_rad=.03,
        rot_freq=(1000./60*tau),
        load=1000.,
        lube=oil_vg68,
        lube_type='grease',
        factor=1.
):
    """Heat loss of roller bearings by Palmgren model [Jun10]_

    Returns a `HeatFunc` function for use as argument in `HeatSource`
    elements. It takes the temperature dependent lube viscosity into
    account, which may change over simulation time. Additionally the
    arguments `rot_freq`, `load` and `factor` may be time
    dependent.

    Load dependent on axial load ax_l, radial load rad_l and
    axial load factor ax_f (see manufacturer data, symbol Y) [Sch13]_:

    +------------------+-----------------------------+-------------------------+
    | Bearing type     | Single bearing              | Double bearing          |
    +==================+=============================+=========================+
    | Grooved ball     | 3.3*ax_l - .1*rad_l         |                         |
    +------------------+-----------------------------+-------------------------+
    | Angular contact  | ax_l - .1*rad_l             | 1.4*ax_l - .1*rad_l     |
    | ball single row  |                             |                         |
    +------------------+-----------------------------+-------------------------+
    | Angular contact  | 1.4*ax_l - .1*rad_l         |                         |
    | ball double row  |                             |                         |
    +------------------+-----------------------------+-------------------------+
    | Four point       | 1.5*ax_l - 3.6*rad_l        |                         |
    +------------------+-----------------------------+-------------------------+
    | Tapered roller   | highest value of            | highest value of        |
    |                  | 2*ax_f + 3.6*rad_l or rad_l | 1.21*ax_f*ax_l or rad_l |
    +------------------+-----------------------------+-------------------------+
    | Spherical roller | 1.6*ax_l/e if ax_l/rad_l > e                          |
    |                  | rad_l*(1 + .6*(ax_l/(e*rad_l)**3)) if ax_l/rad_l <= e |
    +------------------+-------------------------------------------------------+
    | Cylindr. roller  | if ax_l > 0 add additional axial dependent torque     |
    +------------------+-------------------------------------------------------+

    .. warning::

        load = rad_l if load < rad_l

    Args:
        bear_type: Bearing type, may be one of the following:
            'grooved ball',
            'angular contact ball single row',
            'angular contact ball double row',
            'four point',
            'spherical roller',
            'tapered roller',
            'cylindrical roller cage',
            'full complement cylindrical roller',
            'needle roller',
            'grooved ball thrust',
            'axial cylindrical roller',
            'axial needle roller',
        mean_rad: Mean bearing radius
        rot_freq: Rotation frequency
        load: Bearing load force
        lube: Lube base fluid
        lube_type: Lube type, one of the following:
            'grease', 'oil bath', 'oil bath vertical shaft',
            'oil injection', 'oil air'
        factor: Factor by which the result gets multiplied

    Returns:
        Heat loss function
    """

    bearing_types = (
        'grooved ball',
        'angular contact ball single row',
        'angular contact ball double row',
        'four point',
        'spherical roller',
        'tapered roller',
        'cylindrical roller cage',
        'full complement cylindrical roller',
        'needle roller',
        'grooved ball thrust',
        'axial cylindrical roller',
        'axial needle roller',
    )
    idx = bearing_types.index(bear_type)
    if lube_type == 'grease':
        f0_grease_values = (1, 2, 4, 6, 6, 6, .8, 5, 12, 5.5, 9, 14)
        f0 = f0_grease_values[idx]
    else:
        f0_oil_bath_values = (2, 3.3, 6.5, 6, 6, 6, 3, 5, 12, 1.5, 3.5, 5)
        if lube_type == 'oil bath':
            f0 = f0_oil_bath_values[idx]
        elif lube_type == 'oil bath vertical shaft' or lube_type == 'oil injection':
            f0 = 2*f0_oil_bath_values[idx]
        elif lube_type == 'oil air':
            f0 = .5*f0_oil_bath_values[idx]
        else:
            raise ValueError('The given ' + lube_type + ' is not a supported lube type.')
    # f1 for 10 % dynamic load
    f1_values = (.00022, .00046, .00046, .00046, .00060, .00040, .00030, .00055, .00200, .00037, .00150, .00150,)
    f1 = f1_values[idx]

    def rlr_bearing_loss_func(
            temp,
            rot_freq=rot_freq,
            load=load,
            k0=(128*mean_rad**3),
            k1=(36008*f0*mean_rad**3),
            k2=(.5*f1*mean_rad),
            visc=lube.visc_interp,
            factor=factor,
    ):
        v = visc(temp)*aabs(rot_freq)
        if v < 2e-4:
            m0 = k0
        else:
            m0 = k1*v**.666
        return ((m0 + k2*load)*rot_freq)*factor

    return rlr_bearing_loss_func


def cont_heat_loss(
        bear_type='angular contact ball',
        mean_rad=(.0425/2.),
        rlr_rad=(.0045/2.),
        rlr_lgth=.01,
        rlr_count=40,
        cont_ang=.436,
        load=2700,
        rot_freq=(1000./60*tau),
        lube=oil_vg68,
        sealed=True,
        factor=1.,
):
    """Heat loss of roller bearings by a contact model [Jun10]_

    Returns a `HeatFunc` function for use as argument in `HeatSource`
    elements. It takes the temperature dependent lube viscosity into
    account, which may change over simulation time. Additionally, the
    arguments `rot_freq` and `factor` may be time
    dependent.

    Args:
        bear_type: Bearing type, may be one of the following:
            'angular contact ball' and 'spherical roller'
        mean_rad: Mean bearing radius
        load: Load as axial outer force on the bearing
        rot_freq: Rotation frequency
        rlr_rad: Roller radius
        rlr_lgth: Roller length
        rlr_count: Roller number
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
    lube_visc = lube.visc_interp(40)
    # Coefficients of Vogel-Cameron viscosity model [Jun10]_ equation 4.23
    k1 = log(lube.visc_interp(20)/lube_visc)/(1/(20 + 95) - 1/135)  # B
    a = lube_visc/exp(k1/135)  # A
    visc_press_coeff = .000000129*(lube_visc**.204)    # α'
    matl_coeff = visc_press_coeff*red_mod  # G
    # The calculated friction torque of the rolling contact gets multiplied
    # by the seal friction factor to take seal friction in to account.
    seal_factor = 2. if sealed else 1.
    if bear_type == 'angular contact ball':
        cont_force = load/(rlr_count*sin(cont_ang))  # F_k
        cont_load_coeff = cont_force/(red_mod*rlr_rad**2)  # W
        pitch_circle_ratio = mean_rad/rlr_rad  # t_m
        k2 = .5386
        k0 = (rlr_count*4.31*matl_coeff**-.2*cont_load_coeff**-.569
              *(lube.dens_interp(40)*a*pitch_circle_ratio
                /(2*red_mod))**k2
              *cont_force*mean_rad
              *seal_factor)
    elif bear_type == 'spherical roller':
        cont_force = load/rlr_count  # F_k
        cont_load_coeff = cont_force/(red_mod*rlr_rad*rlr_lgth)  # W
        pitch_circle_ratio = mean_rad/rlr_rad  # t_m
        k2 = .598
        k0 = (rlr_count*85.7*matl_coeff**-.352*cont_load_coeff**-.575
              *(lube.dens_interp(40)*a*pitch_circle_ratio
                /(2*red_mod))**k2
              *cont_force*mean_rad
              *seal_factor)
    else:
        raise ValueError(
            f"The given {bear_type} is not a supported bearing type."
        )

    def rlr_bearing_cont_loss_func(
            temp,
            rot_freq=rot_freq,
            k0=k0,
            k1=k1,
            k2=k2,
            factor=factor,
    ):
        return (k0*rot_freq*(aabs(rot_freq)*exp(k1/(temp + 95)))**k2)*factor
    return rlr_bearing_cont_loss_func


def inner_race_heat_loss_share(
        mean_rad=.035,
        rlr_rad=.0095,
        cont_ang=.628
):
    """Inner race share of bearing heat loss [Jun10]_

    The bearing heat loss is created on the inner and the outer race.
    This function calculates the normalised share of the inner race.
    Valid for: mean_rad/rol_elem_rad > 3

    Args:
        mean_rad: Mean bearing radius
        rlr_rad: Roller radius
        cont_ang: Contact angle

    Returns:
        Normalised heat loss share
    """
    return .5 + 1.08 * rlr_rad / mean_rad * (1. - sin(cont_ang)) ** .37


def seal_heat_loss(
        seal_rad=.05,
        outer_rad=.06,
        bear_type='grooved ball',
        rot_freq=tau*1000./60,
        factor=1.,
):
    """Total seal heat loss of roller bearing [SKF06]_

    Returns a `HeatFunc` function, which can be used as argument in
    `HeatSource` elements.
    The arguments `rot_freq` and `factor` may be time dependent.

    Args:
        bear_type: Bearing type, may be one of the following:
            'grooved ball',
            'angular contact ball',
            'self-aligning ball',
            'spherical roller',
            'cylindrical roller',
        seal_rad: Radius of seal contact
        outer_rad: Outer bearing radius
        rot_freq: Rotation frequency
        factor: Factor by which the result gets multiplied

    Returns:
        Heat loss function
    """
    if bear_type == 'grooved ball':
        b = 2.25
        if outer_rad <= .031:
            ks1, ks2 = .023, 2
        elif .031 < outer_rad <= .04:
            ks1, ks2 = .018, 20
        elif .04 < outer_rad <= .05:
            ks1, ks2 = .018, 15
        else:
            ks1, ks2 = .018, 0
    elif (bear_type == 'angular contact ball'
          or bear_type == 'self-aligning ball'):
        b = 2
        ks1, ks2 = .014, 10
    elif bear_type == 'cylindrical roller':
        b = 2
        ks1, ks2 = .032, 50
    elif bear_type == 'spherical roller':
        b = 2
        ks1, ks2 = .057, 50
    else:
        raise ValueError(
            f"The given {bear_type} is not a supported bearing type."
        )

    def seal_heat_loss_func(
            temp=None,
            rot_freq=rot_freq,
            torque=(ks1*(seal_rad*2000)**b + ks2)/1000,
            factor=factor,):
        return aabs(rot_freq)*torque*factor

    return seal_heat_loss_func


def cond(
        inner_rad=.035,
        rlr_rad=.0095,
        rlr_count=14,
        rot_freq=tau*1000./60,
        factor=1.,
):
    """Radial heat conductance of rolling bearing [Sch93]_

    Returns a `CondFunc` function, which can be used as argument in
    `CondLink` elements.
    The argument `rot_freq` may be time-dependent.

    Args:
        inner_rad: Inner bearing radius
        rlr_rad: Roller radius
        rlr_count: Roller number
        rot_freq: Bearing rotation frequency
        factor: Factor by which the result gets multiplied

    Returns:
        Conductance function
    """

    def rolling_bearing_cond_func(
            temp0=None,
            temp1=None,
            matl=None,
            rot_freq=rot_freq,
            k0=rlr_count * 590. * (2.*rlr_rad)**2. * factor,
            k1=(2*inner_rad + 2*rlr_rad) / 2000.,
            k2=-log(2.*rlr_rad) + 7,
    ):
        # prevent too small conductances on small speeds
        rot_freq = aabs(rot_freq)
        rot_freq = 1. if rot_freq < 1. else rot_freq
        return k0*sqrt(log(rot_freq*k1) + k2)

    return rolling_bearing_cond_func


def rlr_count(inner_rad, bear_type='angular contact ball'):
    """Estimated number of bearing rollers

    'angular contact ball': valid from 0.01 to 0.12, created with 72 series data from Coredemar;
    'grooved ball': valid from 0.004 to 0.13, created with 60 series data from HCH
    """
    if bear_type == 'angular contact ball':
        return 5. + 14. * (1. - exp(-.046 * inner_rad))
    elif bear_type == 'grooved ball':
        return 6.4 + 8.4 * (1. - exp(-.046 * inner_rad))
    else:
        raise ValueError(f"The given {bear_type} is not a supported bearing type.")


def rlr_rad(inner_rad, bear_type='angular contact ball'):
    """Estimated radius of bearing rollers

    'angular contact ball': valid from 0.01 to 0.12, created with 72 series data from Coredemar;
    'grooved ball': valid from 0.004 to 0.13, created with 60 series data from HCH
    """
    if bear_type == 'angular contact ball':
        return .001849 + inner_rad * .1819
    elif bear_type == 'grooved ball':
        return (-3.755 + inner_rad * 2000.)**.4485 / 1000.
    else:
        raise ValueError(f"The given {bear_type} is not a supported bearing type.")


def cont_cond(
        cont_temp=40.,
        bear_type='angular contact ball',
        mean_rad=.0425,
        rlr_rad=.0045,
        rlr_lgth=.01,
        rlr_count=40,
        cont_ang=.436,
        load=2700,
        rot_freq=(.75 * 1000. / 60 * tau),
        lube=oil_vg68,
        factor=1.
):
    """Conductance of roller bearings by a contact model [Jun10]_
    page 175

    Returns a `CondFunc` function, which can be used as argument in
    `CondLink` elements.
    The argument `rot_freq` may be time-dependent.

    Args:
        cont_temp: Temperature in the rolling contact
        bear_type: Bearing type, may be one of the following:
            'angular contact ball' and 'spherical roller'
        mean_rad: Mean bearing radius
        rlr_rad: Roller radius
        rlr_lgth: Roller length, not needed in case
            of 'spherical roller'
        rlr_count: Number of Rollers
        cont_ang: Contact angle
        load: Load as axial outer force on the bearing
        rot_freq: Rotation frequency, may be time-dependent input
        lube: Lube base fluid
        factor: Factor by which the result gets multiplied

    Returns:
        Conductance function
    """
    max_rot_freq = 1000.  # 60000 1/min
    rot_freq_pts = hstack([[1e-32], geomspace(.5, max_rot_freq, 12)])
    red_mod = 2.31e11  # E' Reduced elastic modulus
    lube_visc = lube.visc_interp(cont_temp)
    lube_dyn_visc = lube_visc * lube.dens_interp(cont_temp)  # η0
    lube_condy = lube.condy_interp(cont_temp)
    steel_condy = 32.8  # solids.bearing_steel.condy_interp(30.)
    steel_diff = steel_condy / (7850. * 480.)
    speed_coeff = lube_dyn_visc * rot_freq_pts * mean_rad / (2. * red_mod * rlr_rad)  # U
    if bear_type == 'angular contact ball':
        cont_force = load / (rlr_count * sin(cont_ang))  # F_k
        cont_load_coeff = cont_force / (red_mod * rlr_rad ** 2.)  # W
        cont_cond = (
                1.8 * steel_condy * rlr_rad * cont_load_coeff ** .3333
                * (1. + cont_load_coeff ** .1667 * (rot_freq_pts * rlr_rad * mean_rad / steel_diff) ** .5)
        )
        film_cond = .22 * lube_condy * rlr_rad * cont_load_coeff ** .83 * speed_coeff ** -.67
        filling_oil_cond = .05 * steel_condy * rlr_rad
    elif bear_type == 'spherical roller':
        cont_force = load / rlr_count  # F_k
        cont_load_coeff = cont_force / (red_mod * rlr_rad * rlr_lgth)  # W
        cont_cond = (
                .32 * steel_condy * rlr_lgth * cont_load_coeff ** .08615
                * (1. + 1.13 * cont_load_coeff ** .25 * (rot_freq_pts * rlr_rad * mean_rad / steel_diff) ** .5)
        )
        film_cond = .041 * lube_condy * rlr_lgth * cont_load_coeff ** .72 * speed_coeff ** -.7
        filling_oil_cond = .12 * steel_condy * rlr_lgth
    else:
        raise ValueError(
            f"The given {bear_type} is not a supported bearing type."
        )
    rot_freq_pts[0] = 0.
    rlr_cond = (cont_cond * film_cond / (2 * cont_cond + film_cond) + filling_oil_cond)
    cond = rlr_count * rlr_cond * factor

    # No math. approximation function found because resulting curves differ strongly
    interp_func = interpolate.interp1d(rot_freq_pts, cond)

    def rolling_bearing_cont_cond_func(
            temp0=None,
            temp1=None,
            matl=None,
            rot_freq=rot_freq,
    ):
        return interp_func(aabs(rot_freq))

    return rolling_bearing_cont_cond_func
