from numpy import array, allclose

from thermca import Solid, solids, fluids

"""
nan = float('nan')
molding_compound2 = Solid(
    name='Copper 85 %, Molding compound 15 %',
    condy=array(
        [[-173, -123, -73, -23, 26, 76, 126, 226, 326, 526, 726, 926],
         [410, 365, 351, 345, 341, 337, 334, 328, 322, 311, 299, 288]]),
    dens=array([[nan],
                [7.99e+03]]),
    spec_heat=array([[-73.2, -23.1, 25, 76.8, 127, 227, 327],
                     [502, 518, 527, 533, 538, 546, 554]]),
    limits=(-73.2, 327),
)
print(molding_compound2)
sensor_matl = Solid.mix(
    matls=(copper, molding_compound),
    shares=(.85, .15),
    #name='Sensor material',
)
print(sensor_matl)
print(copper)
"""


def test_material_mix():
    matl0 = Solid(
        condy=[[0, 1, 2],
               [1, 2, 1]],
        dens=[[0, 1, 2],
              [1, 2, 1]],
        spec_heat=[[0, 1, 2],
                   [1, 2, 1]],
        name='matl0'
    )

    matl1 = Solid(
        condy=2,
        dens=2,
        spec_heat=2,
        name='matl1'
    )

    matl_mix = Solid.mix(
        matls=(matl0, matl1),
        shares=(.5, .5)
    )
    assert (
        allclose(matl_mix._condy, array([[0, 1, 2], [1.5, 2, 1.5]]),
            rtol=0, atol=3e-03)), (
        'The mix conductivity is not close enough to the expected result!')

    assert (
        allclose(matl_mix._dens, array([[0, 1, 2], [1.5, 2, 1.5]]),
             rtol=0, atol=3e-03)), (
        'The mix density is not close enough to the expected result!')

    assert (
        allclose(matl_mix._spec_heat, array([[0, 1, 2], [1.6666, 2, 1.6666]]),
                 rtol=0, atol=3e-03)), (
        'The mix specific heat is not close enough to the expected result!')

   # dens*sheat must be [ 2.5  4. 2.5]