import numpy as np
from sparse import DOK
from thermca.tests.numpy_vindex import VindexArray

from thermca._utils import sparse_dok_indexer


def test_aix():
    assert_array_equal = np.testing.assert_array_equal
    x = np.arange(3 * 4 * 5).reshape((3, 4, 5)).view(VindexArray)
    d = DOK.from_numpy(x)
    # # test getitem and setitem
    d.aix[[0, 1], [1, 1], [0, 0]] = [1, 2]
    x.vindex[[0, 1], [1, 1], [0, 0]] = [1, 2]
    assert_array_equal(d.todense(), x)
    d.aix[[0, 1], [0, 0], [0, 0]] = 2
    x[[0, 1], [0, 0], [0, 0]] = 2
    assert_array_equal(d.todense(), x)
    d.aix[[0, 1], [1, 1], [0, 0]] = d.aix[[0, 1], [0, 0], [0, 0]]
    x.vindex[[0, 1], [1, 1], [0, 0]] = x.vindex[[0, 1], [0, 0], [0, 0]]
    assert_array_equal(d.todense(), x)
