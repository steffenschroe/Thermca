"""Thermal contact film coefficients

Example:
    >>> from thermca import *
    >>> c0 = .13  # Conductivity oil
    >>> c1 = 50.  # Conductivity steel
    >>> rz1 = 1e-6  # Roughness after fine grinding
    >>> ten_str = 7.5e8  # Tensile strength hardened steel
    >>> hardn = 250e7  # Hardened steel 250 HV,  1 HV = 1e7 N/m**2
    >>> pres = .0001  # Contact pressure
    >>> film0 = c0 / (rz1 + rz1)  # Rough approximation
    >>> film1 = contact.metals(c0, c1, rz1, pres, ten_str)
    >>> film2 = contact.solids(c0, c1, rz1, pres, hardn)
    >>> print(f"{film0=:.1f} {film1=:.1f} {film2=:.1f}")
    film0=65000.0 film1=435500.0 film2=65000.0
"""


def metals(gap_condy, condy, rough, pres, ten_str, condy2=None, rough2=None):
    """Thermal contact of metal surfaces

    Source: [Jun10]_

    Attainable roughness Rz:
        milling 6.3e-6, fine milling 1.6e-6, turning 25e-6, grinding 0.6e-6,
        fine grinding 0.1e-6, cylindrical grinding 0.25e-6, drilling 25.e-6,
        scraping 2.5e-6, honing 0.4e-6, shaving 2.5e-6, polish 0.04e-6

    Conversion of roughness degrees:
        Rz ≈ (3) ... 4 ... (7) * Ra

    Args:
        gap_condy: Conductivity of gap filling material
        condy: Conductivity of the metal
        rough: Roughness Rz of the metal surface as defined in
            ISO 4287
        pres: Contact pressure
        ten_str: Tensile strength of the softer surface
        condy2: If given, it's the conductivity of the second contact
            partner, otherwise the conductivity is the same
        rough2: If given, it's the roughness of the second contact
            partner, otherwise the roughness is the same

    Returns:
        Film coefficient of the contact
    """
    condy2 = condy if condy2 is None else condy2
    rough2 = rough if rough2 is None else rough2
    sum_rough = rough + rough2
    if sum_rough >= 30e-6:
        k = 1.0
    elif sum_rough >= 15e-6:
        k = (30e-6 / sum_rough) ** 0.33
    else:
        k = 15e-6 / sum_rough
    film0 = gap_condy / sum_rough
    condy_row = condy * condy2 / (condy + condy2)
    cond_pres = (pres / ten_str * k) ** 0.86
    film1 = 928.0 * condy_row * cond_pres
    return 6.7 * (film0 + film1)


def solids(gap_condy, condy, rough, pres, hardn, condy2=None, rough2=None):
    """Thermal contact of solid surfaces

    Jungnickel [Jun10]_ approximation considers the ratio of direct
    contact area to nominal area corresponding to the ratio of surface
    pressure to the hardness of the softer partner.

    Conversion of hardness degrees, HV, HB, HK in 1e7 N/m**2:
        HV ≈ 1.05 * HB
        HV ≈ 194. + 0.08 * HRC**2.14
        HV ≈ 76. + 0.0047 * HRB**2.30
        HV ≈ 1.59 * HK
        HV ≈ 85. + 0.75 * HSD**1.53
        HV ≈ 86. + 3.2e−6 * HSA**4.
        HV ≈ 3.35 * HM**3

    Args:
        gap_condy: Conductivity of gap filling material
        condy: Conductivity of the metal
        rough: Roughness Rz of the metal surface as defined in
            ISO 4287
        pres: Contact pressure
        hardn: Hardness HV of the softer surface
        condy2: If given, it's the conductivity of the second contact
            partner, otherwise the conductivity is the same
        rough2: If given, it's the roughness of the second contact
            partner, otherwise the roughness is the same

    Returns:
        Film coefficient of the contact
    """
    condy2 = condy if condy2 is None else condy2
    rough2 = rough if rough2 is None else rough2
    return (pres / hardn) / (rough / condy + rough2 / condy2) + (1 - pres / hardn) / (
        (rough + rough2) / gap_condy
    )
