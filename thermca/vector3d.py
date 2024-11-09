"""Numba accelerated 3d vector functions

To enhance the readability, the functions always return a new array.
The computational speed is sacrificed for readability.
"""

import math

from numpy import empty
from numba import jit, double


@jit(nopython=True)
def vec(scal0, scal1, scal2):
    """Create 3d vector"""
    vecr = empty(3)
    vecr[0] = scal0
    vecr[1] = scal1
    vecr[2] = scal2
    return vecr


@jit(nopython=True)
def add(vec0, vec1):
    """Add two 3d vectors"""
    vecr = empty(3)
    vecr[0] = vec0[0] + vec1[0]
    vecr[1] = vec0[1] + vec1[1]
    vecr[2] = vec0[2] + vec1[2]
    return vecr


@jit(nopython=True)
def sub(vec0, vec1):
    """Subtract 3d vectors"""
    vecr = empty(3)
    vecr[0] = vec0[0] - vec1[0]
    vecr[1] = vec0[1] - vec1[1]
    vecr[2] = vec0[2] - vec1[2]
    return vecr


@jit(nopython=True)
def mul(vec0, vec1):
    """Multiply 3d vectors"""
    vecr = empty(3)
    vecr[0] = vec0[0]*vec1[0]
    vecr[1] = vec0[1]*vec1[1]
    vecr[2] = vec0[2]*vec1[2]
    return vecr


@jit(nopython=True)
def mul_scal(vec, scal):
    """Multiply 3d vector with scalar"""
    vecr = empty(3)
    vecr[0] = vec[0]*scal
    vecr[1] = vec[1]*scal
    vecr[2] = vec[2]*scal
    return vecr


@jit(nopython=True)
def div(vec0, vec1):
    """Divide 3d vectors"""
    vecr = empty(3)
    vecr[0] = vec0[0]/vec1[0]
    vecr[1] = vec0[1]/vec1[1]
    vecr[2] = vec0[2]/vec1[2]
    return vecr


@jit(nopython=True)
def div_scal(vec, scal):
    """Divide 3d vector by scalar"""
    res = empty(3)
    res[0] = vec[0] / scal
    res[1] = vec[1] / scal
    res[2] = vec[2] / scal
    return res


@jit(nopython=True)
def copy(vec):
    """Divide 3d vector by scalar"""
    vecr = empty(3)
    vecr[0] = vec[0]
    vecr[1] = vec[1]
    vecr[2] = vec[2]
    return vecr


@jit(nopython=True)
def cross(vec0, vec1):
    """Cross product of two 3d vectors"""
    vecr = empty(3)
    a0, a1, a2 = double(vec0[0]), double(vec0[1]), double(vec0[2])
    b0, b1, b2 = double(vec1[0]), double(vec1[1]), double(vec1[2])
    vecr[0] = a1*b2 - a2*b1
    vecr[1] = a2*b0 - a0*b2
    vecr[2] = a0*b1 - a1*b0
    return vecr


@jit(nopython=True)
def dot(vec0, vec1):
    """Dot product of two 3d vectors"""
    return vec0[0]*vec1[0] + vec0[1]*vec1[1] + vec0[2]*vec1[2]


@jit(nopython=True)
def norm(vec):
    """Norm of a 3d vector"""
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])


@jit(nopython=True)
def normalize(vec):
    """Normalized vector"""
    vecr = empty(3)
    norm_ = norm(vec)
    if norm_ < 1e-6:
        vecr[0] = 0.
        vecr[1] = 0.
        vecr[2] = 0.
    else:
        vecr[0] = vec[0]/norm_
        vecr[1] = vec[1]/norm_
        vecr[2] = vec[2]/norm_
    return vecr


@jit(nopython=True)
def rot_mat(angle, axis):
    """Gives a rotation matrix for a rotation of angle radians around
    the vector axis
                
    Args:
        angle: angle of rotation
        axis: 3d vector that represents the axis

    Returns:
        rotation matrix
    """
    # vpython, https://github.com/BruceSherwood/vpython-wx/blob/master/src/core/util/tmatrix.cpp
    # see also https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    # or https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glRotate.xml
    ax = normalize(axis)
    matr = empty((3, 3))
    c = math.cos(angle)
    s = math.sin(angle)
    ic = 1. - c
    icxx = ic*ax[0]*ax[0]
    icxy = ic*ax[0]*ax[1]
    icxz = ic*ax[0]*ax[2]
    icyy = ic*ax[1]*ax[1]
    icyz = ic*ax[1]*ax[2]
    iczz = ic*ax[2]*ax[2]
    matr[0, 0], matr[1, 0], matr[2, 0] = icxx + c, icxy + ax[2]*s, icxz - ax[1]*s
    matr[0, 1], matr[1, 1], matr[2, 1] = icxy - ax[2]*s, icyy + c, icyz + ax[0]*s
    matr[0, 2], matr[1, 2], matr[2, 2] = icxz + ax[1]*s, icyz - ax[0]*s, iczz + c
    return matr


@jit(nopython=True)
def mat_vec_mul(mat, vec):
    vecr = empty(3)
    vecr[0] = mat[0, 0]*vec[0] + mat[1, 0]*vec[1] + mat[2, 0]*vec[2]
    vecr[1] = mat[0, 1]*vec[0] + mat[1, 1]*vec[1] + mat[2, 1]*vec[2]
    vecr[2] = mat[0, 2]*vec[0] + mat[1, 2]*vec[1] + mat[2, 2]*vec[2]
    return vecr


@jit(nopython=True)
def mat_mul(mat0, mat1):
    matr = empty((3, 3))
    for col in range(3):
        matr[col, 0] = (mat0[0, 0]*mat1[col, 0] + mat0[1, 0]*mat1[col, 1]
                        + mat0[2, 0]*mat1[col, 2])
        matr[col, 1] = (mat0[0, 1]*mat1[col, 0] + mat0[1, 1]*mat1[col, 1]
                        + mat0[2, 1]*mat1[col, 2])
        matr[col, 2] = (mat0[0, 2]*mat1[col, 0] + mat0[1, 2]*mat1[col, 1]
                        + mat0[2, 2]*mat1[col, 2])
    return matr