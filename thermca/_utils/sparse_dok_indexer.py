"""Vector indexer for sparse.DOK array"""

# Copyright 2019 Steffen Schroeder

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import product, repeat
from numbers import Integral

import numpy as np
from sparse import DOK


class DOKArrayIndexer:
    """Indexing with vectors only for sparse.DOK

    This is a fast version for vectorized or "fancy" indexing. Only
    vectors (1d sequences) are accepted for indexing. The getitem
    method always returns dense ndarrays in contrast to normal
    fancy indexing of sparse arrays which create new copies of sparse
    arrays. The setitem method allows only vectors and scalars as
    values. This allows fast calculations over ndarray methods
    without the expensive creation of temporary sparse arrays between
    operations. This is of course only useful for small vectors, which
    fit easily in to memory.

    Example::
        @property
        def aix(self):
            return DOKArrayIndexer(self)

        DOK.aix = aix
        d = DOK((2, 2), dtype=np.int64)
        # setitem
        d.aix[[0, 1],[1, 1]] = [1, 2]
        d.aix[[0, 1],[1, 1]] = 2
    """

    def __init__(self, dok):
        self._dok = dok

    def _normalize_idxs(self, idxs):
        if not isinstance(idxs, tuple):  # one dimension, one argument
            idxs = (idxs,)
        if len(idxs) != self._dok.ndim:
            raise NotImplementedError(
                f'Index sequences for all {self._dok.ndim} array dimensions needed!'
            )
        idxs = tuple(np.asanyarray(idxs) for idxs in idxs)
        if not (isinstance(k.dtype, Integral) for k in idxs):
            raise IndexError('Indices must be sequences of integer types!')
        if not all(idxs[0].shape == k.shape for k in idxs[1:]):
            raise IndexError('Unequal length of index sequences!')
        if idxs[0].ndim != 1:
            raise IndexError('Indices are not 1d sequences!')
        return idxs

    def __getitem__(self, idxs):
        idxs = self._normalize_idxs(idxs)
        fill_value = self._dok.fill_value
        data_keys = self._dok.data.keys()
        data = self._dok.data
        return np.array(
            [data[idx] if idx in data_keys else fill_value for idx in zip(*idxs)],
            self._dok.dtype,
        )

    def __setitem__(self, idxs, values):
        idxs = self._normalize_idxs(idxs)
        values = np.asanyarray(values, self._dok.dtype)
        if values.ndim == 0:
            values = np.full(idxs[0].size, values, self._dok.dtype)
        elif values.ndim > 1:
            raise ValueError(f'Dimension of values ({values.ndim}) must be 0 or 1!')
        if not idxs[0].shape == values.shape:
            raise ValueError(
                f'Shape mismatch of indices ({idxs[0].shape}) and values ({values.shape})!'
            )
        fill_value = self._dok.fill_value
        data = self._dok.data
        for idx, value in zip(zip(*idxs), values):
            # self._dok._setitem(k, value)
            if not value == fill_value:
                data[idx] = value
            elif idx in data:
                del data[idx]


def add_array_indexer_to_sparse_dok():
    """Add array indexer to sparse.DOK array"""

    @property
    def aix(self):
        return DOKArrayIndexer(self)

    DOK.aix = aix


add_array_indexer_to_sparse_dok()
