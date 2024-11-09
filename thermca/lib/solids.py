"""Database for thermal properties of solid state materials

The library contend is located in the `solids` module.

Example:
    >>> # Create a brick consisting of glass.
    >>> from thermca import *
    >>>
    >>> with Model() as model:
    ...     glass_brick = lp_parts.block(
    ...         posn=(1., 0., 0.),
    ...         matl=solids.glass,
    ...         name='glass_brick'
    ...     )
"""

from thermca.materials import Solid, func_to_table, plot
from numpy import linspace

#: Steel with number 1.0315, DIN St37
steel = steel_10315 = st37 = Solid(
    condy=func_to_table(
        linspace(0, 600, 11),
        lambda temp: (
            5.774e-3 * temp - 1.301e-4 * temp**2 + 1.088e-7 * temp**3 + 57.177
        ),
    ),
    dens=7850.0,
    spec_heat=func_to_table(
        linspace(0, 600, 11),
        lambda temp: (
            414.337 + 0.983 + temp - 2.232e-3 * temp**2 + 2.691e-6 * temp**3
        ),
    ),
    name='Steel 1.0315',
)

#: Bearing stainless steel with number 1.3505, DIN 100Cr6
bearing_steel = steel_13505 = Solid(
    condy=func_to_table(
        linspace(0, 200, 11), lambda temp: 33.0 * (1.0 - 0.0005 * (temp - 20.0))
    ),
    dens=7840.0,
    spec_heat=func_to_table(
        linspace(0, 200, 11), lambda temp: 461.0 * (1.0 + 0.001 * (temp - 20.0))
    ),
    name='Steel 1.3505',
)

#: Spring steel with number 1.8159, DIN 51CrV4
spring_steel = steel_18159 = Solid(
    condy=42.0,
    dens=7850.0,
    spec_heat=460.0,
)

#: lamellar graphite cast iron with number .603, EN-GJL-300; [Jun10]_
cast_iron_gjl_300 = cast_iron_06030 = Solid(
    condy=47.5,
    dens=7250.0,
    spec_heat=460.0,
)

#: spheroidal graphite cast iron with number .706, EN-GJS-600; [Jun10]_
cast_iron_gjs_600 = cast_iron_07060 = Solid(
    condy=32.0,
    dens=7200.0,
    spec_heat=500.0,
)

#: unalloyed medium carbon steel capable of through hardening, number 1.0503, DIN C45; [Jun10]_
carbon_steel = c45 = carbon_steel_10503 = Solid(
    condy=48.0,
    dens=7860.0,
    spec_heat=740.0,
)

#: Aluminium alloy with number 3.3211, DIN EN AW6061, DIN AlMg1SiCu
al_alloy = aw6061 = al_alloy_33211 = Solid(
    condy=func_to_table(
        linspace(0, 400, 11), lambda temp: 164.257 + 0.082 * temp - 1.071e-4
    ),
    dens=2700.0,
    spec_heat=func_to_table(
        linspace(0, 400, 11), lambda temp: 1100.0 * (1.0 + 0.0005 * (temp - 20.0))
    ),
    name='Aluminium alloy 3.3211',
)

#: Aluminium
al = Solid(
    # Fundamentals of Heat and Mass Transfer 6th edn.
    condy=[
        [-173.15, -73.15, 126.85, 326.85, 526.85],
        [302.0, 237.0, 240.0, 231.0, 218.0],
    ],
    # for performance the temperature points of dens and spec_heat should be the same
    # Kurochkin, A.R., Popel’, P.S., Yagodin, D.A. et al. High Temp (2013) 51: 197. https://doi.org/10.1134/S0018151X13020120
    dens=func_to_table(
        [-173.15, -73.15, 126.85, 326.85, 526.85], lambda temp: 2714 - temp * 0.185
    ),
    # Fundamentals of Heat and Mass Transfer 6th edn.
    spec_heat=[
        [-173.15, -73.15, 126.85, 326.85, 526.85],
        [482.0, 798.0, 949.0, 1033.0, 1146.0],
    ],
    name='Aluminium',
)

#: Copper
copper = Solid(
    # Fundamentals of Heat and Mass Transfer 6th edn.
    condy=[
        [-173.15, -73.15, 126.85, 326.85, 526.85, 726.85, 926.85],
        [482.0, 413.0, 393.0, 379.0, 366.0, 352.0, 339.0],
    ],
    dens=8933.0,
    spec_heat=[
        [-173.15, -73.15, 126.85, 326.85, 526.85, 726.85, 926.85],
        [252.0, 356.0, 397.0, 417.0, 433.0, 451.0, 480.0],
    ],
    name='Copper',
)

#: Porcelain
porcelain = Solid(condy=1.03, dens=2400.0, spec_heat=1080.0, name='Porcelain')

#: Aluminium_oxide, purity of the ceramic 95%
alumina = aluminium_oxide = al2o3 = al_oxide = Solid(
    condy=[[20, 50, 100, 150, 200, 230], [20.65, 19.1, 17.25, 15.6, 13.95, 13.1]],
    dens=3500.0,
    spec_heat=920.0,
    name='Aluminium oxide',
)

#: Glass, average material values
glass = Solid(condy=1.0, dens=2500.0, spec_heat=800.0, name='Glass')

#: Cardboard
cardboard = Solid(condy=0.26, dens=1000.0, spec_heat=1340.0, name='Cardboard')

#: Epoxy resin
epoxy_resin = Solid(condy=0.25, dens=1180.0, spec_heat=1170.0, name='Epoxy resin')

#: Dense styrofoam
dense_styrofoam = Solid(
    condy=0.045, dens=100.0, spec_heat=1380.0, name='Dense styrofoam'
)

#: Soft styrofoam
styrofoam = soft_styrofoam = Solid(
    condy=0.029, dens=15.0, spec_heat=1250.0, name='Soft styrofoam'
)

#: Calcium silicate insulating
calcium_silicate_insulating = Solid(
    # Calcium Silicate Insulating Boards, Celsius Heat Management België NV
    # mean values of different materials, 0 and 1000°C extrapolated
    condy=[
        [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0],
        [0.05, 0.07, 0.1, 0.14, 0.17, 0.2],
    ],
    dens=257.0,
    spec_heat=1040.0,
    name='Calcium silicate insulating',
)

#: Salted butter
salted_butter = Solid(
    # Composition: 80% butterfat, 16% moisture, 2% NaCl
    # E.L. Watson: THERMAL PROPERTIES OF BUTTER. CANADIAN AGRICULTURAL ENGINEERING,
    # VOL. 17 NO. 2, DECEMBER 1975
    # condy 30°C is extrapolated; dens: valid 0-90°C from Ginzburg (1985)
    condy=[
        [-40.0, -30.0, -20.0, -10.0, -7.0, -3.0, 0.0, 10.0, 025.0, 30.0],
        [0.244, 0.239, 0.2285, 0.218, 0.2165, 0.218, 0.2215, 0.2435, 0.304, 0.326],
    ],
    dens=func_to_table(
        [-40.0, -20.9, -17.6, -10.6, -8.6, -1.6, 4.5, 22.6, 32.6, 40.0],
        lambda temp: 949.8 - 0.7 * temp,
    ),
    spec_heat=[
        [-40.0, -20.9, -17.6, -10.6, -8.6, -1.6, 4.5, 22.6, 32.6, 40.0],
        [1650.0, 1930.0, 5060.0, 3360.0, 4690.0, 2780.0, 2850.0, 4500.0, 4100.0, 2.6e3],
    ],
    name='Salted butter',
)

#: Thermal compound
#: From arctic mx-4 thermal compound for CPU and VGA cards, from 2017 spec sheet
thermal_compound = thermal_grease = Solid(
    condy=8.5, dens=2500.0, spec_heat=1500.0, name='Thermal compound'  # guessed
)

#: PVC-P used as cable insulator, floor covering, roof sealing
soft_pvc = pvc_p = Solid(condy=0.14, dens=1270.0, spec_heat=1350.0, name='Soft PVC')

#: Silicone elastomer is a rubber like material used for electrical insulators,
#: automotive applications, cooking, baking, and food storage products
silicone_rubber = silicone_elastomer = Solid(
    condy=0.22, dens=1300.0, spec_heat=1400.0, name='Silicone elastomer'
)

#: Cotton laminated fabric (e.g. PF CC 201) is made of phenolic-formaldehyde resin (PF)
#: and cotton fabrics. It is used for electrical and thermal insulators.
cotton_laminated_fabric = Solid(
    condy=0.35, dens=1350.0, spec_heat=1500.0, name='Cotton laminated fabric'
)

if __name__ == '__main__':
    # plot material parameters of all solids
    globvals = list(globals().values())
    solids = set((v for v in globvals if isinstance(v, Solid)))
    for solid in solids:
        print(solid)
        # plot(solid)
