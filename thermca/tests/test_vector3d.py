from numpy import array, isclose, allclose, eye, dot as adot, cross as across

from scipy.linalg import expm, norm as lnorm

from thermca.vector3d import add, sub, mul, mul_scal, div, div_scal, copy, cross, dot, norm, normalize, rot_mat


def test_vector3d():
    """Call all functions to compile them and test against numpy functions"""
    vec0, vec1, angle = array([0., 1., 2.]), array([1., 2., 3.]), 1.5
    assert allclose(add(vec0, vec1), vec0 + vec1)
    assert allclose(sub(vec0, vec1), vec0 - vec1)
    assert allclose(mul(vec0, vec1), vec0*vec1)
    assert allclose(mul_scal(vec0, 4.), vec0*4.)
    assert allclose(div(vec0, vec1), vec0/vec1)
    assert allclose(div_scal(vec0, 4.), vec0/4.)
    assert allclose(copy(vec0), vec0)
    assert allclose(cross(vec0, vec1), across(vec0, vec1))
    assert allclose(dot(vec0, vec1), adot(vec0, vec1))
    assert allclose(norm(vec0), lnorm(vec0))
    assert allclose(normalize(vec0), vec0/lnorm(vec0))
    assert allclose(rot_mat(angle, vec0),
                    expm(across(eye(3), vec0/lnorm(vec0)*angle)))