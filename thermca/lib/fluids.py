"""Database for thermal properties of fluid materials

The library contend is located in the `fluids` module.
"""

from thermca.materials import Fluid, plot
from numpy import linspace, exp


#: Dry air fluid at 1 bar pressure [VDI13]_
air = Fluid(
    condy=[[ -150., -100.,   -50.,     0.,    25.,   50.,     75.,   100., 150.,   200.,   300.,   400.,   600., 1000.],
           [.01168, .0162, .02042, .02425, .02625, .02808, .02987, .03162, .035, .03825, .04442, .05024, .06114, .0811]],
    # For performance: the temperature points of dens and spec_heat should be the same
    dens=[[-150., -100.,  -50.,    0.,   25.,   50.,   75.,  100.,  150.,  200.,  300.,  400.,  600., 1000.],
          [2.860, 2.019, 1.563, 1.276, 1.169, 1.078, 1.000, .9333, .8229, .7359, .6075, .5172, .3988, .2735]],
    spec_heat=[[-150., -100.,  -50.,    0.,   25.,   50.,   75.,  100.,  150.,  200.,  300.,  400.,  600., 1000.],
               [1021., 1009., 1006., 1006., 1007., 1008., 1009., 1011., 1017., 1025., 1045., 1069., 1116., 1185.]],
    visc=[[   -150.,    -100.,     -50.,     0.,      25.,      50.,      75.,     100.,   150.,     200.,     300.,     400.,   600.,     1000.],
          [30.29e-7, 58.34e-7, 93.49e-7, 135e-7, 157.9e-7, 182.2e-7, 207.8e-7, 234.6e-7, 292e-7, 353.9e-7, 490.7e-7, 643.5e-7, 993e-7, 1851.4e-7]],
    name='Air')

#: Water Fluid at 1 bar pressure
water = Fluid(
    condy=[[-20., -10., 0., 10., 20., 30., 40., 50., 75., 100.],
           [.513, .539, .562, .582, .6, .615, .629, .641, .6635, .678]],
    dens=[[-20., -10., 0., 10., 20., 30., 40., 50., 75., 100.],
          [993.6, 998.1, 999.8, 999.7, 998.2, 995.7, 992.2, 988.1, 974.9,
           958.6]],
    spec_heat=[[-20., -10., 0., 10., 20., 30., 40., 50., 75., 100.],
               [4401., 4272., 4219., 4195., 4185., 4180., 4179., 4180., 4192.,
                4216.]],
    visc=[[-20., -10., 0., 10., 20., 30., 40., 50., 75., 100.],
          [4390e-9, 2649e-9, 1792e-9, 1306e-9, 1003e-9, 801e-9, 658e-9, 553e-9,
           387e-9, 295e-9]],
    name='Water')


def _visc(a, b, temp):
    return a*exp(b/(temp + 95))


_temp = linspace(-40, 200, 13)

# vgs = ('vg10', 'vg15', 'vg22', 'vg32', 'vg46', 'vg68', 'vg100', 'vg150', 'vg220', 'vg320', 'vg460')
# a = array([ .1, .0932, .087, .0813, .0762, .071, .0662, .0616, .0575, .0537, .0503])*1e-6
# b = array([623,   687,  748,   808,   866,  929,   990,  1055,  1116,  1176,  1234])

#: Mineral oil with temperature dependent viscosity based on
#: ISO 3448 viscosity grade VG10
oil_vg10 = Fluid(condy=.13, dens=870, spec_heat=1900,
                 visc=(_temp, _visc(.1e-6, 623, _temp)),
                 name='Oil ISO VG10')
#: Mineral oil with temperature dependent viscosity based on
#: ISO 3448 viscosity grade VG15
oil_vg15 = Fluid(condy=.13, dens=870, spec_heat=1900,
                 visc=(_temp, _visc(.0932e-6, 687, _temp)),
                 name='Oil ISO VG15')
#: Mineral oil with temperature dependent viscosity based on
#: ISO 3448 viscosity grade VG22
oil_vg22 = Fluid(condy=.13, dens=870, spec_heat=1900,
                 visc=(_temp, _visc(.087e-6, 748, _temp)),
                 name='Oil ISO VG22')
#: Mineral oil with temperature dependent viscosity based on
#: ISO 3448 viscosity grade VG32
oil_vg32 = Fluid(condy=.13, dens=870, spec_heat=1900,
                 visc=(_temp, _visc(.0813e-6, 808, _temp)),
                 name='Oil ISO VG32')
#: Mineral oil with temperature dependent viscosity based on
#: ISO 3448 viscosity grade VG46
oil_vg46 = Fluid(condy=.13, dens=870, spec_heat=1900,
                 visc=(_temp, _visc(.0762e-6, 866, _temp)),
                 name='Oil ISO VG46')
#: Mineral oil with temperature dependent viscosity based on
#: ISO 3448 viscosity grade VG68
oil_vg68 = Fluid(condy=.13, dens=870, spec_heat=1900,
                 visc=(_temp, _visc(.071e-6, 929, _temp)),
                 name='Oil ISO VG68')
#: Mineral oil with temperature dependent viscosity based on
#: ISO 3448 viscosity grade VG100
oil_vg100 = Fluid(condy=.13, dens=870, spec_heat=1900,
                  visc=(_temp, _visc(.0662e-6, 990, _temp)),
                  name='Oil ISO VG100')
#: Oil with temperature dependent viscosity based on
#: ISO 3448 viscosity grade VG150
oil_vg150 = Fluid(condy=.13, dens=870, spec_heat=1900,
                  visc=(_temp, _visc(.0616e-6, 1055, _temp)),
                  name='Oil ISO VG150')
#: Oil with temperature dependent viscosity based on
#: ISO 3448 viscosity grade VG220
oil_vg220 = Fluid(condy=.13, dens=870, spec_heat=1885,
                  visc=(_temp, _visc(.0575e-6, 1116, _temp)),
                  name='Oil ISO VG220')
#: Oil with temperature dependent viscosity based on
#: ISO 3448 viscosity grade VG320
oil_vg320 = Fluid(condy=.13, dens=870, spec_heat=1885,
                  visc=(_temp, _visc(.0537e-6, 1176, _temp)),
                  name='Oil ISO VG320')
#: Mineral oil with temperature dependent viscosity based on
#: ISO 3448 viscosity grade VG460
oil_vg460 = Fluid(condy=.13, dens=870, spec_heat=1900,
                  visc=(_temp, _visc(.0503e-6, 1234, _temp)),
                  name='Oil ISO VG460')

#: Ethylene glycol at 1 bar pressure
ethylene_glycol = Fluid(
    condy=[[0.,  7., 17., 27., 37., 47., 57., 67.],
           [.242, .244, .248, .252, .255, .258, .260, .261]],
    visc=[[0.,  7., 17., 27., 37., 47., 57., 67.],
          [57.6e-6, 37.3e-6, 22.1e-6, 14.1e-6, 9.65e-6, 6.91e-6, 5.15e-6, 3.98e-6]],
    dens=[[0.,  7., 17., 27., 37., 47., 57., 67.],
          [1131., 1126., 1119., 1114., 1104., 1096., 1089., 1084.]],
    spec_heat=[[0.,  7., 17., 27., 37., 47., 57., 67.],
               [2294., 2323., 2368., 2415., 2460., 2505., 2549., 2592.]],
    name='Ethylene glycol',
)


if __name__ == '__main__':
    # plot material parameters of all fluids
    globvals = list(globals().values())
    fluids = set((v for v in globvals if isinstance(v, Fluid)))
    for fluid in fluids:
        print(repr(fluid))
        plot(fluid)

    print(water.spec_heat_interp(20))
