from numpy import array, int64, zeros, empty_like, asarray
from scipy.sparse import csr_matrix, spmatrix
from numba import njit


try:
    from sparse_dot_mkl import dot_product_mkl
    sparse_dot = dot_product_mkl
except ImportError:
    sparse_dot = csr_matrix.__matmul__


@njit
def coo_diag_data_idxs(shape, data_shape, row, col):
    """Diagonal element indices of the data attribute of coo matrix"""
    diag_idxs = []
    for row_idxs in range(shape[0]):
        for data_idxs in range(data_shape[0]):
            if row[data_idxs] == row_idxs and col[data_idxs] == row_idxs:
                diag_idxs.append(data_idxs)
                break
    return array(diag_idxs)


@njit
def sparse_diag_dot_dense(diag_mat_data, dense_mat):
    """Multiply sparse diagonal matrix with dense matrix

    Sparse diagonal matrix must have all diagonal elements including
    zeros in its `data` attribute
    """
    ret_mat = empty_like(dense_mat)
    for rcol, dcol in zip(ret_mat.T, dense_mat.T):
        rcol[:] = diag_mat_data * dcol
    return ret_mat


@njit
def _csr_data_idxs(indptr, indices, dense_idx0, dense_idx1):
    sparse_idxs = []
    for row, col in zip(dense_idx0, dense_idx1):
        for idx in range(indptr[row], indptr[row + 1]):
            if indices[idx] == col:
                sparse_idxs.append(idx)
                break
        else:
            raise IndexError("No value stored for given index!")  # f'No value stored for index ({row}, {col})!'
    return asarray(sparse_idxs, dtype=indices.dtype)


def to_csr_data_idxs(csr_matrix, dense_idx: tuple):
    """Converts indices from dense two-dimensional matrix
    to indices for the sparse csr matrix data array"""
    dense_idx0, dense_idx1 = (
        asarray(dense_idx[0], csr_matrix.indices.dtype),
        asarray(dense_idx[1], csr_matrix.indices.dtype)
    )
    # print(f"{csr_matrix.indptr=}, {csr_matrix.indices=}, {dense_idx0=}, {dense_idx1=}")
    #print(f"{type(csr_matrix.indptr)=}, {type(csr_matrix.indices)=}, {type(dense_idx0)=}, {type(dense_idx1)=}")
    return _csr_data_idxs(csr_matrix.indptr, csr_matrix.indices, dense_idx0, dense_idx1)


@njit
def _sub_matrix_calc(row_start, row_stop, col_start, col_stop, indptr, col_indices, data, shape0):
    sub_indptr = []
    sub_indices = []
    sub_data = []
    orig_data_idxs = []
    sub_data_idx = 0
    for row_i in range(shape0):
        if row_start <= row_i < row_stop:
            sub_indptr.append(sub_data_idx)
            for data_i in range(indptr[row_i], indptr[row_i + 1]):
                col_i = col_indices[data_i]
                if col_start <= col_i < col_stop:
                    orig_data_idxs.append(data_i)
                    sub_indices.append(col_i - col_start)
                    sub_data.append(data[data_i])
                    sub_data_idx += 1
    # if sub_indptr:
    sub_indptr.append(sub_data_idx)
    # print(matrix.shape, row_start, row_stop, col_start, col_stop, sub_data, sub_indices, sub_indptr)
    col_len = (col_stop - col_start)
    row_len = (row_stop - row_start)
    if col_len == 0 or row_len == 0:
        shape = (0, 0)
    else:
        shape = (row_len, col_len)
    return array(sub_data), array(sub_indices), array(sub_indptr), array(shape), array(orig_data_idxs, dtype=col_indices.dtype)


def csr_sub_matrix(matrix, row_start, row_stop, col_start, col_stop):
    """Creates sub matrix from a csr matrix without sparsity change.

    This means the zeros remain in the data. The function returns the
    sub matrix and the indices of the data in the data array of the
    source matrix"""
    row_start, row_stop, _ = slice(row_start, row_stop).indices(matrix.shape[0])
    col_start, col_stop, _ = slice(col_start, col_stop).indices(matrix.shape[1])
    sub_data, sub_indices, sub_indptr, shape, orig_data_idxs = _sub_matrix_calc(
        row_start, row_stop, col_start, col_stop, matrix.indptr, matrix.indices, matrix.data, matrix.shape[0])
    sub_matrix = csr_matrix((sub_data, sub_indices, sub_indptr), shape=shape)
    return sub_matrix, orig_data_idxs


@njit
def coo_to_csr(row, col, data, shape):
    """Convert COO sparse matrix to CSR format

    The algorithm is taken from scipy.
    Warning: it doesn't remove double entries!

    Returns:
        CSR sparse matrix properties and
        additionally the transformation information
        between both data arrays: to_csr_data_idxs
    """
    indptr = zeros(shape[0] + 1, dtype=int64)
    indices = empty_like(col, dtype=int64)
    csr_data = empty_like(data)
    # to_coo_data_idxs = empty_like(data, dtype=int64)
    to_csr_data_idxs = empty_like(data, dtype=int64)
    nnz = len(data)
    for n in range(nnz):
        indptr[row[n]] += 1

    cumsum = 0
    for i in range(shape[0]):
        temp = indptr[i]
        indptr[i] = cumsum
        cumsum += temp
    indptr[shape[0]] = nnz

    for n in range(nnz):
        row_idx = row[n]
        dest_idx = indptr[row_idx]

        indices[dest_idx] = col[n]
        csr_data[dest_idx] = data[n]
        # to_coo_data_idxs[n] = dest_idx
        to_csr_data_idxs[dest_idx] = n

        indptr[row_idx] += 1

    last = 0
    for i in range(shape[0] + 1):
        temp = indptr[i]
        indptr[i] = last
        last = temp

    return indptr, indices, csr_data, shape, to_csr_data_idxs
