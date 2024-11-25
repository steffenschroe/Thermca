import numpy as np
from sparse import DOK
from thermca.tests.numpy_vindex import VindexArray

from thermca._utils import sparse_dok_indexer


def test_vix():
    assert_array_equal = np.testing.assert_array_equal
    x = np.arange(3*4*5).reshape((3, 4, 5)).view(VindexArray)
    d = DOK.from_numpy(x)

    # # Test getitem
    # sequences
    assert_array_equal(d.vix[[]].todense(), x[[]])
    assert_array_equal(d.vix[[], []].todense(), x[[], []])
    assert_array_equal(d.vix[[2]].todense(), x[[2]])
    assert_array_equal(d.vix[[1, 2]].todense(), x[[1, 2]])
    assert_array_equal(d.vix[[1, 2], [1, 2]].todense(), x[[1, 2], [1, 2]])
    assert_array_equal(d.vix[[1, 2], [1, 2], [1, 2]].todense(), x[[1, 2], [1, 2], [1, 2]])
    # slices
    assert_array_equal(d.vix[[0, 1], [0, 1], :].todense(), x.vindex[[0, 1], [0, 1], :])
    assert_array_equal(d.vix[[0, 1], :, [0, 1]].todense(), x.vindex[[0, 1], :, [0, 1]])
    assert_array_equal(d.vix[:, [0, 1], [0, 1]].todense(), x.vindex[:, [0, 1], [0, 1]])
    assert_array_equal(d.vix[:, :, [0, 1]].todense(), x.vindex[:, :, [0, 1]])
    assert_array_equal(d.vix[:, [0, 1], :].todense(), x.vindex[:, [0, 1], :])
    assert_array_equal(d.vix[[0, 1], :, :].todense(), x.vindex[[0, 1], :, :])
    assert_array_equal(d.vix[[0, 1], [0, 1], 1].todense(), x.vindex[[0, 1], [0, 1], 1])
    assert_array_equal(d.vix[[0, 1], 1, [0, 1]].todense(), x.vindex[[0, 1], 1, [0, 1]])
    assert_array_equal(d.vix[1, [0, 1], [0, 1]].todense(), x.vindex[1, [0, 1], [0, 1]])
    assert_array_equal(d.vix[1, 1, [0, 1]].todense(), x.vindex[1, 1, [0, 1]])
    assert_array_equal(d.vix[1, [0, 1], 1].todense(), x.vindex[1, [0, 1], 1])
    assert_array_equal(d.vix[[0, 1], 1, 1].todense(), x.vindex[[0, 1], 1, 1])
    # different behaviour between scalar indices and slices resulting in one scalar
    assert_array_equal(d.vix[[0, 1], [0, 1], 0:1].todense(), x.vindex[[0, 1], [0, 1], 0:1])
    assert_array_equal(d.vix[[0, 1], 0:1, [0, 1]].todense(), x.vindex[[0, 1], 0:1, [0, 1]])
    assert_array_equal(d.vix[0:1, [0, 1], [0, 1]].todense(), x.vindex[0:1, [0, 1], [0, 1]])
    assert_array_equal(d.vix[0:1, 0:1, [0, 1]].todense(), x.vindex[0:1, 0:1, [0, 1]])
    assert_array_equal(d.vix[0:1, [0, 1], 0:1].todense(), x.vindex[0:1, [0, 1], 0:1])
    assert_array_equal(d.vix[[0, 1], 0:1, 0:1].todense(), x.vindex[[0, 1], 0:1, 0:1])
    # test empty slice
    assert_array_equal(d.vix[[0, 1], [0, 1], 1:1].todense(), x.vindex[[0, 1], [0, 1], 1:1])
    # test fallback to slice_getitem
    assert_array_equal(d.vix[1, 1, 1], x.vindex[1, 1, 1])
    assert_array_equal(d.vix[1, 1, :].todense(), x.vindex[1, 1, :])
    assert_array_equal(d.vix[1, :, :].todense(), x.vindex[1, :, :])
    assert_array_equal(d.vix[:, :, :].todense(), x.vindex[:, :, :])

    # # test setitem
    d.vix[[0, 1], [1, 1], [0, 0]] = [1, 2]
    x.vindex[[0, 1], [1, 1], [0, 0]] = [1, 2]
    assert_array_equal(d.todense(), x)
    d.vix[[0, 1], [0, 0], [0, 0]] = 2
    x[[0, 1], [0, 0], [0, 0]] = 2
    assert_array_equal(d.todense(), x)
    d.vix[[0, 1], [1, 1], [0, 0]] = d.vix[[0, 1], [0, 0], [0, 0]].todense()
    x.vindex[[0, 1], [1, 1], [0, 0]] = x.vindex[[0, 1], [0, 0], [0, 0]]
    assert_array_equal(d.todense(), x)


def test_aix():
    assert_array_equal = np.testing.assert_array_equal
    x = np.arange(3*4*5).reshape((3, 4, 5)).view(VindexArray)
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