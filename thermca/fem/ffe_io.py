from numpy import ndarray
from scipy.sparse import diags

from thermca.fem.fe_system import FESystem
from thermca._utils.sparse import sparse_dot, sparse_diag_dot_dense


class FFEIO:
    """Full finite element state space IO-system

    It may be a full FE-system or MOR-system (reduced model order)

    Args:
        fe_sys: Full FE system of part
        films: Boundary film coefficients of linked surfaces
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
        films,
        film_surf_idxs,
        surf_net_idxs: ndarray,
        link_elem_net_idxs,
        link_self_net_idxs,
    ):
        self.fe_sys = fe_sys
        self.films = films
        self.film_surf_idxs = film_surf_idxs
        self.surf_net_idxs = surf_net_idxs
        self.link_elem_net_idxs = link_elem_net_idxs
        self.link_self_net_idxs = link_self_net_idxs

        A_body, A_films, B, M_inv = to_state_space(
            fe_sys.M_diag, fe_sys.L, fe_sys.Qs, film_surf_idxs
        )
        self.A_body = A_body
        A_body_coo = A_body.tocoo(
            copy=False
        )  # If A_body.data changes, A_body_coo.data will still hold reference to old A_body.data!
        self.A_body_coo = A_body_coo
        self.A_films = A_films
        # Fast runtime version of: A = A_body + sum(f * A for f, A in zip(films, A_films))
        # TODO: create the following only in case of Part.temp_dep = True to save memory
        A_body_coo = A_body.tocoo(copy=False)
        A_diag_idxs = (A_body_coo.row == A_body_coo.col).nonzero()[0]
        A_films_diag_idxs = [
            Af.tocoo().col for Af in A_films
        ]  # Only some elements on diagonal are occupied
        A_data_films_idxs = [A_diag_idxs[Afdi] for Afdi in A_films_diag_idxs]
        A = A_body.copy()
        for film, data_film_idxs, A_film in zip(films, A_data_films_idxs, A_films):
            A.data[data_film_idxs] += film * A_film.data
        self.A = A
        self.A_data_films_idxs = A_data_films_idxs
        self.B = B
        self.M_inv = M_inv
        self.C_surf = fe_sys.C_surf
        self.C_body = fe_sys.C_body

    def update_material_properties(self, body_temps):
        """Update temperature dependent material properties"""
        fe_sys = self.fe_sys
        fe_sys.update_material_properties(body_temps)
        self.M_inv.data = 1.0 / fe_sys.M_diag
        self.B = sparse_diag_dot_dense(self.M_inv.data, fe_sys.Qs)
        A_body = sparse_dot(self.M_inv, fe_sys.L)
        self.A_body.data = A_body.data

    def update_state_matrix(self):
        """Update state matrix for changed films or body properties"""
        self.A.data = self.A_body.data.copy()
        for film, data_film_idxs, A_film in zip(
            self.films, self.A_data_films_idxs, self.A_films
        ):
            self.A.data[data_film_idxs] += film * A_film.data


def to_state_space(M_diag, L, Qs, film_idxs):
    """Creates a full DOF state space representation

    This is a modified version of a function written by Michael Bauer
    Copyright (C) 2020 Michael Bauer
    License: GNU GENERAL PUBLIC LICENSE Version 3

    Args:
        M_diag: Dense vector of the diagonal of capacity matrix
        L: Conductance matrix
        Qs: Array columns of connection surface node "areas"
        film_idxs: Coupling surface indices of film coefficients

    Returns:
        A_body: State matrix of IO-system containing information of
            the solid state body only
        A_films: State matrices for surf Robin BC containing film
            coefficients
        B: Input matrix of IO-system containing columns of surface
            area information of surfs
        M_inv: Inverse of capacity matrix
    """
    M_inv = diags(1.0 / M_diag, format='csr')
    A_body = sparse_dot(M_inv, L)  # A_body = M_inv @ L
    if len(film_idxs) > 0:
        # First use only constant surface data Qs
        A_films = [M_inv @ diags(q, format='csr') for q in Qs.T[film_idxs]]
    else:
        A_films = None

    B = sparse_diag_dot_dense(M_inv.data, Qs)  # B = M_inv @ Qs

    return A_body, A_films, B, M_inv
