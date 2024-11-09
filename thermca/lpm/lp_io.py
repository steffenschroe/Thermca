from scipy.sparse import diags
from numpy import ndarray, zeros, add

from thermca._utils.sparse import sparse_dot
from thermca.lpm.lp_system import LPSystem


class LPIO:
    """Lumped parameter state space IO-system

    Args:
        lp_sys: Lumped parameter system of part
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
        lp_sys: LPSystem,
        films,
        film_surf_idxs,
        surf_net_idxs: ndarray,
        link_elem_net_idxs,
        link_self_net_idxs,
    ):
        self.lp_sys = lp_sys
        self.films = films
        self.film_surf_idxs = film_surf_idxs
        self.surf_net_idxs = surf_net_idxs
        self.link_elem_net_idxs = link_elem_net_idxs
        self.link_self_net_idxs = link_self_net_idxs
        self.L_film_margs = film_marg_series(lp_sys, films, film_surf_idxs)
        self.L_film_margs_sum = self.L_film_margs.sum(axis=0)
        A_body, A_film_margs, M_inv = to_state_space(
            lp_sys.M_diag, lp_sys.L[: lp_sys.dof], self.L_film_margs
        )

        # Fast runtime version of: A = A_body + sum(A_film_margs)
        A_body_coo = A_body.tocoo(copy=False)
        A_diag_data_idxs = (A_body_coo.row == A_body_coo.col).nonzero()[0]
        # Vectors of diagonals
        A_film_margs_diag = M_inv.data * self.L_film_margs.sum(axis=1)
        A = A_body.copy()
        A.data[A_diag_data_idxs] += A_film_margs_diag

        self.A_body = A_body
        self.A_diag_data_idxs = A_diag_data_idxs
        self.A = A
        # self.B = B
        self.M_inv = M_inv
        self.C_marg = lp_sys.C_marg
        self.C_body = lp_sys.C_body

    def update_material_properties(self, body_temps):
        """Update temperature dependent material properties"""
        lp_sys = self.lp_sys
        lp_sys.update_material_properties(body_temps)
        self.M_inv.data = 1.0 / lp_sys.M_diag
        # self.B = sparse_diag_dot_dense(self.M_inv.data, fe_sys.Qs)
        A_body = sparse_dot(self.M_inv, lp_sys.L[: lp_sys.dof])
        self.A_body.data = A_body.data

    def update_state_matrix(self):
        """Update state matrix for changed films or body properties

        Returns summed conductance's of film coefficients and margins of
        each surface
        """
        L_film_margs = film_marg_series(self.lp_sys, self.films, self.film_surf_idxs)
        self.L_film_margs = L_film_margs
        self.L_film_margs_sum = L_film_margs.sum(axis=0)
        A_film_margs_diag = self.M_inv.data * L_film_margs.sum(axis=1)
        self.A.data = self.A_body.data.copy()
        self.A.data[self.A_diag_data_idxs] += A_film_margs_diag
        return self.L_film_margs_sum


def film_marg_series(lp_sys, films, film_idxs):
    """Series connection of margin and film for every surface"""
    L_film_margs = zeros((lp_sys.dof, len(films)))
    for i, (face_areas, film, smxdxs, marg_idxs) in enumerate(
        zip(
            (lp_sys.surf_face_areas[fi] for fi in film_idxs),
            films,
            (lp_sys.surf_to_marg_data_idxs[fi] for fi in film_idxs),
            (lp_sys.marg_dof_idxs[fi] for fi in film_idxs),
        )
    ):
        # Film and margin conductance for each surface face
        film_conds = face_areas * film
        marg_conds = -lp_sys.L.data[smxdxs]
        # Compute series conductance and sum it up for each margin dof
        add.at(
            L_film_margs[:, i],
            marg_idxs,
            film_conds * marg_conds / (film_conds + marg_conds),
        )
    return L_film_margs


def to_state_space(M_diag, L_body, L_film_margs):
    """Creates a state space representation

    Args:
        M_diag: Dense vector of the diagonal of capacity matrix
        L_body: Body conductance matrix
        L_film_margs: Series connection of body margin and film
            conductance for every surface

    Returns:
        A_body: State matrix of IO-system containing information of the
            solid state body only
        A_film_marg: State matrices for surf Robin BC containing series
            connection of body margin and film coefficients
        M_inv: Inverse of capacity matrix
    """
    M_inv = diags(1.0 / M_diag, format='csr')

    A_body = sparse_dot(M_inv, L_body)  # A_body = M_inv @ L_body
    if len(L_film_margs) > 0:
        A_film_margs = [M_inv @ diags(L, format='csr') for L in L_film_margs.T]
    else:
        A_film_margs = None

    return A_body, A_film_margs, M_inv
