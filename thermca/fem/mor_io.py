"""Generates a thermal FE-system of reduced degree of freedom

Alexander Galant developed this method and implemented it in MATLAB:
Alexander Galant, Knut Großmann, Andreas Mühl:
Thermo-Elastic Simulation of Entire Machine Tool.
(2015). 10.1007/978-3-319-12625-8_7.

Translated into efficient Python code by Michael Bauer
Copyright (C) 2020 Michael Bauer
License: GNU GENERAL PUBLIC LICENSE Version 3

Integrated into Thermca by Steffen Schroeder
"""


from warnings import catch_warnings, simplefilter

from numpy import array, ndarray, sqrt, zeros, full
from numpy.linalg import norm, pinv
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

from thermca.fem.fe_system import FESystem
from thermca._utils.func_tools import disk_cache


class MORIO:
    """Finite element state space IO-system of reduced dimension

    It is called MOR-system (reduced model order)

    Args:
        fe_sys: Full FE-system of part
        mor_dof: Dimension of reduced system
        ref_temp: MOR reference temperature
        films: Boundary film coefficients on linked surfaces
        film_surf_idxs: Indices of linked surfaces regarding 'films'
        surf_net_idxs: Network indices of all surfaces of this part
        link_elem_net_idxs: Network indices of element nodes linked to
            the surfaces of this part
        link_self_net_idxs: Network indices of nodes representing the
            linked surfaces of this part by mean surface temperatures
    """

    def __init__(
        self,
        fe_sys: FESystem,
        mor_dof: int,
        ref_temp,
        films,
        film_surf_idxs,
        surf_net_idxs: ndarray,
        link_elem_net_idxs,
        link_self_net_idxs,
    ):
        self.fe_sys = fe_sys
        self.ref_temp = ref_temp
        self.films = films
        self.film_surf_idxs = film_surf_idxs
        self.surf_net_idxs = surf_net_idxs
        self.link_elem_net_idxs = link_elem_net_idxs
        self.link_self_net_idxs = link_self_net_idxs
        self.mor_dof = mor_dof

        # FEM -> Statespace -> MOR
        init_films = full(
            len(films), 1.0
        )  # Initialize MOR model with film of 1. for now
        M_is, A_body, A_film, A, B, C_surf, C_body = to_mor_state_space(
            fe_sys.M_diag,
            fe_sys.L,
            fe_sys.Qs,
            fe_sys.C_surf,
            fe_sys.C_body,
            init_films,
            film_surf_idxs,
        )
        A_body, A_films, B_T, C_surfs, VT0, VT1 = transform(
            M_is, A_body, A_film, A, B, C_surf, mor_dof
        )
        self.A_body = A_body
        self.A_films = A_films
        self.A = A_body + sum(f * A for f, A in zip(films, A_films))
        self.B_T = B_T
        self.C_surfs = C_surfs
        self.C_body = C_body
        self.VT0 = VT0
        self.VT1 = VT1

    def update_state_matrix(self):
        """Update state matrix for changed films or body properties"""
        self.A = self.A_body + sum(f * A for f, A in zip(self.films, self.A_films))


def to_mor_state_space(M_diag, L, Qs, C_surf, C_body, films, film_idxs):
    """Creates a state space representation for model order reduction

    Args:
        M_diag: Dense vector of the diagonal of capacity matrix
        L: Body conductance matrix
        Qs: Load Matrix with columns for coupling surfaces
            containing node related surface area
        C_surf: Output matrix of IO-system to get the mean surf
            temperature
        films: Coupling surface film coefficients
        film_idxs: Coupling surface indices of film coefficients

    Returns:
        M_is: Inverted square root of diagonalised capacity matrix
        A_body: State matrix of IO-system containing information of
            the solid state body only
        A_film: State matrices for surf Robin BC containing a film
            of 1.
        A: A_body + sum(A_film) as input to Arnoldi process
        B: Input Matrix with columns for coupling surfaces
        C_surf: Output matrix of IO-system to get the mean coupling
            surface temperature
        C_body: Output matrix of IO-system to get the mean body
            temperature
    """
    M_is = diags(1.0 / sqrt(M_diag), format='csr')

    A_body = M_is @ L @ M_is
    if len(films) > 0:
        A_film = [M_is @ diags(q, format='csr') @ M_is for q in Qs.T[film_idxs]]
        A = A_body + sum(A_film)  # Input to Arnoldi process
    else:
        A_film = None
        A = A_body

    B = M_is @ Qs  # Will be column-wise reduced

    C_surf = M_is @ C_surf  # M_is for temperature re-transformation, M_is⋅C=(Cᵀ⋅M_is)ᵀ
    C_body = C_body @ M_is  # 1d

    return M_is, A_body, A_film, A, B, C_surf, C_body


@disk_cache
def transform(M_is, A_body, A_film, A, B, C, dim):
    VTs = transformation_matrices(A, B, dim)

    # MOR System
    # Store the n_pin matrices as 3-dim tensor.
    Ar_body = array([VT.T @ A_body @ VT for VT in VTs])  # => (surf num, dim, dim)
    # 4-dim tensor
    if A_film is not None:
        Ar_film = array(
            [array([VT.T @ A @ VT for VT in VTs]) for A in A_film]
        )  # => (film num, surf num, dim, dim)
    else:
        Ar_film = None

    # Column-wise reduction
    Br_T = array([VT.T @ B[:, i] for i, VT in enumerate(VTs)])  # => (surf num, dim)

    Crs = array([VT.T @ C for VT in VTs])  # => (surf num, dim, surf num)

    # Transformation matrix for initial temperatures
    # T = VT1 * Tr
    VT1 = array([M_is @ VT for VT in VTs])
    # Transformation matrix for results
    # Tr = pinv(VT1) * T0
    VT0 = array([pinv(VT) for VT in VT1])

    return Ar_body, Ar_film, Br_T, Crs, VT0, VT1


def transformation_matrices(A, B, dim):
    """Calculate the transformation matrices to the Krylov subspace with
    dimension dim. One for each column vector in B.
    Based on https://en.wikipedia.org/wiki/Arnoldi_iteration
    """

    def arnoldi_iteration(A, b, n: int):
        """Computes a basis of the (n + 1)-Krylov subspace of A: the space
        spanned by {b, Ab, ..., A^n b}.

        From https://en.wikipedia.org/wiki/Arnoldi_iteration
        Modified to create the Krylov subspace of A⁻¹ instead of A.
        Instead of computing v=A·q A·v=q needs to be solved.
        Using LU decomposition from scipy.sparse.linalg.factorized for sparse A.

        Args:
          A: m × m array
          b: initial vector (length m)
          n: dimension of Krylov subspace, must be >= 1

        Returns:
          m x (n + 1) array, the columns are an orthonormal basis of the
            Krylov subspace.
        """
        m = b.shape[0]
        with catch_warnings():
            simplefilter("ignore")
            # SparseEfficiencyWarning, "splu requires CSC matrix format"
            # at the moment it internally transforms A to CSC format
            A = factorized(A)
        Q = zeros((m, n + 1))
        q = b / norm(b)  # Normalize the input vector
        Q[:, 0] = q  # Use it as the first Krylov vector

        for k in range(n):
            # v = A.dot(q)  # Generate a new candidate vector
            # Solve A·v=q; throws a warning because it has to convert A to csc format
            v = A(q)
            for j in range(k + 1):  # Subtract the projections on previous vectors
                h = Q[:, j].conj() @ v
                v = v - h * Q[:, j]

            h = norm(v)
            eps = 1e-12  # If v is shorter than this threshold it is the zero vector
            if h > eps:  # Add the produced vector to the list, unless
                q = v / h  # the zero vector is produced.
                Q[:, k + 1] = q
            else:  # If that happens, stop iterating.
                return Q
        return Q

    return [arnoldi_iteration(A, B[:, i], dim - 1) for i in range(B.shape[1])]
