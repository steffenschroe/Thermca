"""Solver for transient solution"""

from time import perf_counter as time_now

# fmt: off
from numpy import (
    array, arange, vstack, hstack, full, sum as asum, nonzero, isclose, add, dtype,
    float64
)
# fmt: on
from scipy.sparse import csr_matrix, diags
from scipy.integrate import RK23, RK45, LSODA
from scipy.optimize import OptimizeResult
import numba as nb
from sparse import DOK

try:
    from sparse_dot_mkl import sparse_qr_solve_mkl
except ImportError:
    sparse_qr_solve_mkl = None

from thermca._utils.sparse import to_csr_data_idxs, csr_sub_matrix, sparse_dot
from thermca.resultdata import _ResultCollector


class OdeResult(OptimizeResult):
    pass


@nb.jit(nopython=True, cache=True, fastmath=True)
def solve_stationary_nodes(
    stationary_elem_idxs, temp, heat_src, cond_indptr, cond_indices, cond_data
):
    for i in stationary_elem_idxs:
        sum_cond = 0.0
        ap = heat_src[i]
        for j in range(cond_indptr[i], cond_indptr[i + 1]):
            sum_cond += cond_data[j]
            ap += cond_data[j] * temp[cond_indices[j]]
        if sum_cond == 0:  # Capacity without connection to other capacities
            continue
        temp[i] -= temp[i] - ap / sum_cond


@nb.jit(nopython=True, cache=True, fastmath=True)
def cond_to_differential_equation_representation(
    calc_cond_data, calc_cond_indptr, calc_cond_diag_data_idxs, run_cond_data, m_lgth
):
    """Conductance matrix from node-network to differential equation representation

    Negate conductance entries and insert positive conductance sum on diagonal
    """
    calc_cond_data[:] = -run_cond_data
    for row_i in range(m_lgth):
        sum_cond = 0.0
        for data_i in range(calc_cond_indptr[row_i], calc_cond_indptr[row_i + 1]):
            sum_cond += calc_cond_data[data_i]
        calc_cond_data[calc_cond_diag_data_idxs[row_i]] = -sum_cond


def transient_solution(
    time_span,
    net,
    rel_tol,
    abs_tol,
    method,
    num_skip,  # Number of results to skip over before collecting next result
    **options,
):
    """Simulation with time step control

    The simulation uses Scipy time integration routines with time step
    control. The network system is split into known (k) and unknown (u)
    temperatures.

    ⎡C_uu C_uk⎤ ⎧Ṫ_u⎫ + ⎡L_uu L_uk⎤ ⎧T_u⎫ = ⎧Q̇_u⎫
    ⎣C_ku C_kk⎦ ⎩Ṫ_k⎭   ⎣L_ku L_kk⎦ ⎩T_k⎭   ⎩Q̇_k⎭

    Known temperatures represent temperature bound nodes and current
    mean temperatures of coupling surfaces of parts. The mean
    temperatures are calculated from the last time step and serve as
    boundary temperatures for connected network elements in the current
    time step. The integration routine takes the time derivative as
    input:

    Ṫ_u = C_uu⁻¹ ⋅(Q̇_u - L_uk⋅T_k - C_uu⋅T_u)

    The IO-systems of LP-, FE- and MOR-parts (p) are solved together
    with the network system. The time derivatives of all systems are
    stacked on top of each other in one solve vector Ṫ and given to the
    integration routine:

    {Ṫ} = [{Ṫ_u}ᵀ, {Ṫ_p₁}ᵀ ⋯ {Ṫ_pₙ}ᵀ]ᵀ
    """

    # Network
    net_temp = net.init_temp.copy()
    heat_src = net.heat_src
    run_cond = net.run_cond
    u_lgth = net._free_temp_lgth
    set_time_dep_prps = net._set_time_dependent_properties
    stationary_elem_idxs = net.stationary_elem_idxs
    run_cond_indptr = run_cond.indptr
    run_cond_indices = run_cond.indices
    run_cond_data = run_cond.data
    # The solution needs a state space representation of the
    # conductance matrix. run_cond holds positive conductance values
    # and zeros on diagonal. A state space representation calc_cond
    # is created. It holds negative elements and a positive
    # conductance sum on the diagonal.
    calc_cond = csr_matrix(
        (run_cond.data.copy(), run_cond.indices, run_cond.indptr), run_cond.shape
    )
    calc_cond_data = calc_cond.data
    calc_cond_indptr = calc_cond.indptr
    diag_idxs = arange(u_lgth)
    calc_cond_diag_data_idxs = to_csr_data_idxs(calc_cond, (diag_idxs, diag_idxs))
    inv_capy_uu = diags(1.0 / net.capy[:u_lgth]).tocsr()
    cond_uu, cond_uu_orig_data_idxs = csr_sub_matrix(calc_cond, 0, u_lgth, 0, u_lgth)
    cond_uk, cond_uk_orig_data_idxs = csr_sub_matrix(
        calc_cond, 0, u_lgth, u_lgth, net._node_count
    )
    heat_src_u = heat_src[:u_lgth]
    temp_u = net_temp[:u_lgth]
    temp_k = net_temp[u_lgth : net._node_count]

    # Part IOs
    lp_ios = net._lp_ios
    ffe_ios = net._ffe_ios
    mor_ios = net._mor_ios

    # Initialisation temperature as contiguous temperature vector
    io_init_temps = []
    for lp_io in lp_ios:
        io_init_temps.append(full(lp_io.lp_sys.dof, lp_io.lp_sys.init_temp))
    for ffe_io in ffe_ios:
        io_init_temps.append(full(ffe_io.fe_sys.dof, ffe_io.fe_sys.init_temp))
    for mor_io in mor_ios:
        io_init_temps.append(
            vstack(mor_io.VT0)
            @ (
                full(mor_io.fe_sys.dof, mor_io.fe_sys.init_temp - mor_io.ref_temp)
                / len(mor_io.fe_sys.surf_names)
            )
        )
    if io_init_temps:
        io_init_temps = hstack(io_init_temps)
    else:
        io_init_temps = array([])

    init_temps = hstack((temp_u, io_init_temps))

    # Record array for fast access to contiguous solve temperature vector
    net_dt = dtype([('0', float64, (u_lgth,))])
    lp_dt = dtype(
        [(str(i), float64, (lp_io.lp_sys.dof,)) for i, lp_io in enumerate(lp_ios)]
    )
    ffe_dt = dtype(
        [(str(i), float64, (ffe_io.fe_sys.dof,)) for i, ffe_io in enumerate(ffe_ios)]
    )
    mor_dt = dtype(
        [
            (str(i), float64, (len(mor_io.fe_sys.surf_names), mor_io.mor_dof))
            for i, mor_io in enumerate(mor_ios)
        ]
    )

    solve_temp_dt = dtype(
        [
            ('net', net_dt),
            ('lp', lp_dt),
            ('ffe', ffe_dt),
            ('mor', mor_dt),
        ]
    )

    net.time = time_span[0]

    # Initialise result collection skipping
    result_step = 0
    next_result_coll_step = num_skip

    result_collector = _ResultCollector(
        io_temp_dt=dtype(
            [
                ('lp', lp_dt),
                ('ffe', ffe_dt),
                ('mor', mor_dt),
            ]
        )
    )

    def update_time_derivative(time, solve_temp):
        """Update time and temperature dependent values of the
        differential equation representation and return time derivative;
        The function is called by the solver
        """
        # TODO: Step end event handler in net.sim to get current solve time
        net.time = time

        # # Get components out of continuous solved temp. vector
        (
            net_solve_temp,
            lp_solve_temp,
            fe_solve_temp,
            mor_solve_temp,
        ) = views_into_flat_temps(solve_temp, solve_temp_dt)
        temp_u[:] = net_solve_temp[0]

        # # Mean surface or margin temps. as boundary conditions for connected network elements
        for lp_io, part_temp in zip(lp_ios, lp_solve_temp):
            net_temp[lp_io.surf_net_idxs] = lp_io.C_marg.T @ part_temp
        for ffe_io, part_temp in zip(ffe_ios, fe_solve_temp):
            net_temp[ffe_io.surf_net_idxs] = ffe_io.C_surf.T @ part_temp
        for mor_io, part_temp in zip(mor_ios, mor_solve_temp):
            mean_surf_temps = (
                asum([temps @ C for temps, C in zip(part_temp, mor_io.C_surfs)], axis=0)
                + mor_io.ref_temp
            )
            net_temp[mor_io.surf_net_idxs] = mean_surf_temps

        # # Update time and temperature dependent properties in network
        set_time_dep_prps(time, net_temp, lp_solve_temp, fe_solve_temp, mor_solve_temp)
        solve_stationary_nodes(
            stationary_elem_idxs,
            net_temp,
            heat_src,
            run_cond_indptr,
            run_cond_indices,
            run_cond_data,
        )
        cond_to_differential_equation_representation(
            calc_cond_data,
            calc_cond_indptr,
            calc_cond_diag_data_idxs,
            run_cond.data,
            u_lgth,
        )  # Use run_cond_data?
        cond_uu.data[:] = calc_cond_data[cond_uu_orig_data_idxs]
        cond_uk.data[:] = calc_cond_data[cond_uk_orig_data_idxs]
        heat_src_u[:] = heat_src[:u_lgth]  # Could also be a view?
        # solve_temps[:] = temp[:u_lgth]
        # temp_k[:] = temp[u_lgth:net._condensed_lgth]
        inv_capy_uu.data[:] = (
            1.0 / net.capy[:u_lgth]
        )  # Values on fully occupied diagonal only

        # # Temperature derivatives
        temp_deriv = []

        # Frst network
        if temp_u.size > 0:
            temp_deriv.append(
                inv_capy_uu @ (heat_src_u - cond_uk @ temp_k - cond_uu @ temp_u)
            )

        # Second parts
        for lp_io, part_temp in zip(lp_ios, lp_solve_temp):
            # Heat flux density as inputs
            surf_flux = heat_src[lp_io.surf_net_idxs] / lp_io.lp_sys.surf_areas
            flow = (lp_io.lp_sys.Qs @ surf_flux).ravel()
            flow += (lp_io.L_film_margs @ net_temp[lp_io.link_elem_net_idxs]).ravel()
            B = lp_io.M_inv.data * flow
            temp_deriv.append(B - sparse_dot(lp_io.A, part_temp))
        for ffe_io, part_temp in zip(ffe_ios, fe_solve_temp):
            # Heat flux density as inputs
            surf_flux = heat_src[ffe_io.surf_net_idxs] / ffe_io.fe_sys.surf_areas
            # ẋ = -(A_body + ɑ A_film)x + Bu(ɑ)
            # surf_flux[fe_io.film_idxs] += fe_io.films * net_temp[fe_io.conn_node_idxs]
            add.at(
                surf_flux,
                ffe_io.film_surf_idxs,
                ffe_io.films * net_temp[ffe_io.link_elem_net_idxs],
            )
            # TODO: speed up by only multiply with changed fluxes and used surfs
            #   otherwise use old b; Another possibility is to use known indices
            #   of connection surfaces to prevent multiplication with many zero
            #   areas in B columns or use sparse vectors (Suitesparse)
            load_idxs = nonzero(surf_flux)[0]
            b = (ffe_io.B[:, load_idxs] @ surf_flux[load_idxs]).ravel()
            temp_deriv.append(b - sparse_dot(ffe_io.A, part_temp))
        for mor_io, part_temp in zip(mor_ios, mor_solve_temp):
            # Heat flux density as inputs
            surf_flux = heat_src[mor_io.surf_net_idxs] / mor_io.fe_sys.surf_areas
            # ẋ = -(A_body + ɑ A_film)x + Bu(ɑ)
            # surf_flux[fe_io.film_idxs] += fe_io.films * (net_temp[fe_io.conn_node_idxs] - fe_io.ref_temp)
            add.at(
                surf_flux,
                mor_io.film_surf_idxs,
                mor_io.films * (net_temp[mor_io.link_elem_net_idxs] - mor_io.ref_temp),
            )
            temp_deriv.append(
                (
                    array(
                        [-a @ surf_temps for a, surf_temps in zip(mor_io.A, part_temp)]
                    )
                    + [b * flux for b, flux in zip(mor_io.B_T, surf_flux)]
                ).ravel()
            )

        # Return continuous vector of derivatives
        return hstack(temp_deriv)

    # Initialise solve speed calculation
    old_times = (time_span[0], time_now())

    def step_end_func(time, solve_temp):
        """Routines after each successful time step"""

        nonlocal result_step, next_result_coll_step, old_times

        # Save results or skip if specified
        if next_result_coll_step == result_step or isclose(time, time_span[1]):
            temp_u[:] = solve_temp[:u_lgth]
            io_temp = solve_temp[u_lgth:]
            store_results(net, time, net_temp, io_temp, result_collector)
            next_result_coll_step += num_skip + 1
        result_step += 1

        # Calculate solve speed as sim. time vs real time progress
        now_times = (time, time_now())
        net._solve_step_speed = (old_times[0] - now_times[0]) / (
            old_times[1] - now_times[1]
        )
        net._solve_step_time = time
        old_times = now_times

    # # Simulation loop
    update_time_derivative(time_span[0], init_temps)
    store_results(net, time_span[0], net_temp, io_init_temps, result_collector)
    if time_span[1] > time_span[0]:
        ode_result = simple_solve_ivp(
            update_time_derivative,
            time_span,
            init_temps,
            method,
            step_end_func=step_end_func,
            rtol=rel_tol,
            atol=abs_tol,
            **options,
        )
        if not ode_result.success:
            raise Exception("Solver failed" + ode_result.message)

    return result_collector


def simple_solve_ivp(
    func, t_span, y0, method='RK45', func_args=None, step_end_func=None, **options
):
    """Solve initial value problem

    Simplified version of scipy.integrate.solve_ivp
    with additional hook at end of solver step
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    if func_args is not None:
        func = lambda t, x, func=func: func(t, x, *func_args)
    METHODS = {'RK23': RK23, 'RK45': RK45, 'LSODA': LSODA}
    if method in METHODS:
        method = METHODS[method]
    solver = method(func, t0, y0, tf, vectorized=False, **options)
    status = None
    while status is None:
        message = solver.step()
        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break
        if step_end_func is not None:
            step_end_func(solver.t, solver.y)
    MESSAGES = {
        0: "The solver successfully reached the end of the integration interval.",
        1: "A termination event occurred.",
    }
    message = MESSAGES.get(status, message)
    return OdeResult(
        t=None,
        y=None,
        sol=None,
        t_events=None,
        y_events=None,
        nfev=solver.nfev,
        njev=solver.njev,
        nlu=solver.nlu,
        status=status,
        message=message,
        success=status >= 0,
    )


def store_results(net, time, net_temp, io_temp, res_collector):
    """Store current node net data in a result data container.

    Args:
        res_collector: Result container
    """
    res_collector.times.append(time)
    res_collector.net_temps.append(net_temp.copy())
    res_collector.net_capys.append(net.capy.copy())
    res_collector.net_heat_srcs.append(net.heat_src.copy())
    # # build res_cond as sparse.DOK array
    # first copy all constant values
    # emulate DOK.copy() because it does not exist,
    # don't transfer data to DOK.__init__ because internal copy is expensive
    cond = net.cond
    res_cond = DOK(shape=cond.shape, dtype=cond.dtype, fill_value=cond.fill_value)
    res_cond.data = cond.data.copy()
    # second copy all values stored in run_cond, don't overwrite anything else
    run_cond_coo = net.run_cond.tocoo()
    row_idx, col_idx = run_cond_coo.row, run_cond_coo.col
    res_cond.aix[row_idx, col_idx] = run_cond_coo.data[:]
    res_collector.net_conds.append(res_cond)
    posn = net.posn.copy()
    res_collector.net_clean_conds.append(net.clean_run_cond.todense())
    res_collector.net_posns.append(posn)
    if io_temp.size > 0:
        res_collector.io_temps.append(io_temp)
    if net._lp_ios:
        lp_M_diag = [lp_io.lp_sys.M_diag.copy() for lp_io in net._lp_ios]
        res_collector.lp_M_diagss.append(lp_M_diag)
        lp_L = [lp_io.lp_sys.L.copy() for lp_io in net._lp_ios]
        res_collector.lp_Lss.append(lp_L)


def views_into_flat_temps(temps, temps_dt):
    view = temps.view(dtype=temps_dt)
    return (
        view[0]['net'],
        view[0]['lp'],
        view[0]['ffe'],
        view[0]['mor'],
    )
