"""3D plotting functions to visualise the model elements and the node net."""
from __future__ import annotations
from math import tau
from typing import TYPE_CHECKING, Optional

from numpy import (
    zeros, logical_and, nonzero, where, pi,inf, mean, sum as asum, min as amin, add,
    max as amax, abs as aabs, int64, array, vstack, hstack, column_stack, sqrt, arange,
    full, errstate, isinf, isfinite, minimum, maximum, tril, setdiff1d, unique
)
from scipy.sparse import tril
from sparse import tril as stril
import pyvista as pv

from thermca.plot import primitives as pp
from thermca.plot.primitives import (
    AXES, DEFAULT, color_themes, CAPY, HEAT, COND, CONN, TEMP, LUMP_NODE_IDX, COLOR_BAR, HEAT_SRC, VOL, AREA, EDGE
)
from thermca.plot.model import elements
if TYPE_CHECKING:
    from thermca.model import Model
    from thermca.mesh import Mesh
    from thermca.resultdata import ResultData

# Plot entities
# Element specific
LINK = 'link'
HEAT_SRC_ELEM = 'heat src elem'

# time scales
SIM_SPAN = 'sim span'
POINT_IN_TIME = 'point in time'


def get_lp_part_dat(lp_part, lp_io, M_diag, L, temps, net_clean_cond, net_temp, net_heat_src, net_posn):
    """Get flat arrays of lumped parameters from result of lp_part"""

    from thermca._utils.sparse import to_csr_data_idxs

    lp_sys = lp_io.lp_sys

    # Inner parameters
    # Inner conductances as a lower triangle matrix to prevent double entries
    inner_conds = -tril(L[:lp_sys.dof], -1).tocoo()
    plot_row = [inner_conds.row]
    plot_col = [inner_conds.col]
    plot_cond = [inner_conds.data]
    plot_heat = [inner_conds.data * (temps[inner_conds.row] - temps[inner_conds.col])]
    plot_posn = lp_sys.posns + lp_part._net_dat.posn

    # Parameters of linked margins and surfaces
    # Film dof indices as single nodes to the end
    num_cond_dof = L.shape[0]
    film_dof_idxs = arange(len(lp_io.films)) + num_cond_dof
    conn_net_node_posns = []
    for surf_idx, film_cond, film_temp, film_dof_idx, link_node_posn in zip(
            lp_io.film_surf_idxs,  # Indices of surfaces linked with films
            net_clean_cond[lp_io.link_elem_net_idxs, lp_io.link_self_net_idxs],  # Conducances of film links
            net_temp[lp_io.link_elem_net_idxs],  # Temperatures of linked network elements
            film_dof_idxs,  # Indices of connected elements as points added to the end
            net_posn[lp_io.link_elem_net_idxs],  # Posn. of linked network elements
    ):
        row_idxs = lp_sys.surf_dof_idxs[surf_idx]
        col_idxs = lp_sys.marg_dof_idxs[surf_idx]
        # Margin conductances to invisible surface nodes
        plot_row.append(row_idxs)
        plot_col.append(col_idxs)
        data_idxs = to_csr_data_idxs(L, (row_idxs, col_idxs))
        marg_conds = -L.data[data_idxs]
        plot_cond.append(marg_conds)
        # Film conductances to connected node or surface doc
        face_areas = lp_sys.surf_face_areas[surf_idx]
        film_conds = film_cond * face_areas/face_areas.sum()
        num_faces = len(row_idxs)
        plot_row.append(full(num_faces, film_dof_idx))
        plot_col.append(row_idxs)
        plot_cond.append(film_conds)
        # Heat
        marg_film_conds = 1. / (1. / marg_conds + 1. / film_conds)
        marg_film_heats = marg_film_conds * (film_temp - temps[col_idxs])
        plot_heat.append(marg_film_heats)  # For margin
        plot_heat.append(marg_film_heats)  # Same heat for film
        # Posns
        conn_net_node_posns.append(link_node_posn)  # Connected network node

    # Heat sources on margin nodes
    src_heat = []
    src_heat_idxs = []
    src_surfc_idxs = net_heat_src[lp_io.surf_net_idxs].nonzero()[0]
    obs_net_heat_src_idx = lp_io.surf_net_idxs[src_surfc_idxs]
    # src_surfc_idxs = isin(lp_io.surf_net_idxs, net_srcs_idx).nonzero()
    for src_surfc_idx in src_surfc_idxs:
        heat = net_heat_src[lp_io.surf_net_idxs[src_surfc_idx]]
        face_areas = lp_io.lp_sys.surf_face_areas[src_surfc_idx]
        src_heat.append(heat * face_areas / face_areas.sum())
        src_heat_idxs.append(lp_io.lp_sys.marg_dof_idxs[src_surfc_idx])

    plot_row = hstack(plot_row)
    plot_col = hstack(plot_col)
    plot_cond = hstack(plot_cond)
    plot_heat = hstack(plot_heat)
    plot_capy = M_diag
    # Heat sources from different surfaces may feed heat into the same nodes.
    # Sum heat in cases of duplicate indices.
    src_heat = hstack(src_heat) if src_heat else array([])
    src_heat_idxs = hstack(src_heat_idxs) if src_heat_idxs else array([], dtype=int64)
    unique_idx, inv_idx = unique(src_heat_idxs, return_inverse=True)
    if len(unique_idx) < len(src_heat_idxs):
        unique_vals = zeros(len(unique_idx))
        add.at(unique_vals, inv_idx, src_heat)
        src_heat_idxs = unique_idx
        src_heat = unique_vals
    plot_posn = vstack((plot_posn, array(conn_net_node_posns)))
    return plot_row, plot_col, plot_cond, plot_heat, plot_capy, plot_posn, src_heat, src_heat_idxs, obs_net_heat_src_idx


def _get_net_para_idxs(res):
    """Indices for network parameter types"""
    # Indices for "ordinary" conductance, only upper triangle to prevent redundancy
    conds_sym_idx = []
    conds_asym_idx = []
    for i in range(res.net_conds.shape[0]):
        cond = res.net_conds.vix[i].to_coo()
        conds_sym_idx.append(((cond == cond.T)*(stril(cond) != 0)).nonzero())
        conds_asym_idx.append(((cond != cond.T)*(cond != 0)).nonzero())

    capys_inf_idx = []
    capys_zero_idx = []
    capys_ord_idx = []
    heat_srcs_idx = []
    for capy, temp, pls in zip(res.net_capys, res.net_temps, res.net_heat_srcs):
        capys_inf_idx.append(where(capy == inf)[0])  # Infinite capacities of temp. BCs
        capys_zero_idx.append(where(capy == 0.)[0])  # Zero capacities of stat. nodes
        capy_ord_mask_steps = logical_and(logical_and((capy != inf), (capy != 0.)),
                                          (capy != -inf))  # Ordinary nodes
        capys_ord_idx.append(nonzero(capy_ord_mask_steps)[0])
        heat_srcs_idx.append(where(pls != 0)[0])
    return (conds_sym_idx, conds_asym_idx, capys_inf_idx, capys_zero_idx,
            capys_ord_idx, heat_srcs_idx)


def _calc_para_scale_dat(
        res: ResultData,
        fe_parts,
        part_capys,
        part_sfc_conds,
        lp_parts,
        lp_conds,
        lp_heats,
        lp_capys,
        para_index_dat,
        symbol_size,
        geo_scale_type,
):
    """Data to scale the symbol volumes of the parameter plot

    Scale only for one result time point because only one can be shown
    at a time.
    Args:
        res: result data
        para_index_dat: Indices for parameter types
        symbol_size: (total symbol volume)/(total available volume)
        geo_scale_type: set to VOLU, if the value of the properties
            correspond to the volume of the symbols, set to AREA,
            if the value of the properties should correspond to the
            cross-sectional area of the symbols
    """
    from thermca.lpm.lp_construction import create_lp_elems
    # Get total volume to draw in
    # Part posns are static for now -> use only initial posns
    mins, maxs = [array([inf, inf, inf])], [array([-inf, -inf, -inf])]
    for lp_part in lp_parts:
        lp_blocks, lp_cyls, lp_link_surfs, force_cont_pairs = create_lp_elems(lp_part.asm)
        for body in lp_blocks + lp_cyls:
            min, max = body.bounding_box()
            mins.append(min)
            maxs.append(max)
    for fe_part in fe_parts:
        min, max = fe_part._body_mesh.bounding_box()
        mins.append(min)
        maxs.append(max)
    min = amin(vstack(mins), axis=0)
    max = amin(vstack(maxs), axis=0)
    # Net posns are handled across all times -> min and max for all result times
    pos_mins = amin(res.net_posns, axis=1)
    pos_mins = minimum(pos_mins, min)
    xmins = pos_mins[:, 0]
    ymins = pos_mins[:, 1]
    zmins = pos_mins[:, 2]
    pos_maxs = amax(res.net_posns, axis=1)
    pos_maxs = maximum(pos_maxs, max)
    xmaxs = pos_maxs[:, 0]
    ymaxs = pos_maxs[:, 1]
    zmaxs = pos_maxs[:, 2]
    dxs = xmaxs - xmins
    dys = ymaxs - ymins
    dzs = zmaxs - zmins
    dmaxs = amax([dxs, dys, dzs], axis=0)
    dmaxs = where(dmaxs > 0., dmaxs, 1.)  # all elements on the same position?
    dmins = dmaxs*.1  # aspect ratio of volume dims. at least .1
    dxs = where(dxs > dmins, dxs, dmins)
    dys = where(dys > dmins, dys, dmins)
    dzs = where(dzs > dmins, dzs, dmins)
    if geo_scale_type == VOL:  # volume
        draw_geos = dxs*dys*dzs
    elif geo_scale_type == AREA:
        draw_geos = dxs*dys
    else:
        raise ValueError(f"'geo_scale_type' argument {geo_scale_type} is not supported!")

    (conds_sym_idx, conds_asym_idx, capys_inf_idx, capys_zero_idx,
     capys_ord_idx, heat_srcs_idx) = para_index_dat

    symbol_volus = draw_geos*symbol_size  # Volume of all symbols combined
    capys_sum = zeros(len(res))
    conds_sum = zeros(len(res))
    heats_sum = zeros(len(res))
    time_const_sum = zeros(len(res))
    # self.type = scale_type
    capys_inf = zeros(len(res))
    capys = zeros(len(res))
    heats = zeros(len(res))
    conds = zeros(len(res))
    time_consts = zeros(len(res))
    time_consts_inf = zeros(len(res))
    for ti, (capy_glob, hsrc, time_const_glob) in enumerate(
            zip(res.net_capys, res.net_heat_srcs, res.net_time_consts)):
        lp_capy = lp_capys[ti] if lp_capys else array([])
        lp_cond = lp_conds[ti] if lp_conds else array([])
        lp_heat = lp_heats[ti] if lp_heats else array([])
        # Sum of capacities, conductances and heat flows each drawn with equal total volume
        cond_glob, heat_glob = res.net_conds.vix[ti], res.net_heats.vix[ti]
        # Infinite capacities get average volume symbols
        capys_inf[ti] = mean(hstack((capy_glob[capys_ord_idx[ti]], part_capys, lp_capy)))
        # Sum of "ordinary" and infinite capacities
        capys_sum[ti] = (
                sum(capy_glob[capys_ord_idx[ti]])
                + capys_inf[ti]*len(capys_inf_idx[ti])
                + sum(part_capys)
                + sum(lp_capy)
        )
        cond_sym = cond_glob.aix[conds_sym_idx[ti]]
        cond_inf_mask = cond_sym != float('inf')
        conds_sum[ti] = (
                sum(cond_sym[cond_inf_mask])
                + sum(cond_glob.aix[conds_asym_idx[ti]])
                + sum(sum(conds) for conds in part_sfc_conds)
                + sum(lp_cond)
        )
        # Heat transport not considered in heat flow symbols
        heat_sym = heat_glob.aix[conds_sym_idx[ti]]
        heats_sum[ti] = (
                sum(abs(heat_sym[cond_inf_mask]))
                + sum(hsrc[heat_srcs_idx[ti]])
                + sum(lp_heat)
        )
        # Reference: 1/3 of total volume for capacity symbols
        capys[ti] = symbol_volus[ti]/3./capys_sum[ti]
        if heats_sum[ti] != 0.:
            heats[ti] = capys_sum[ti]/heats_sum[ti]*capys[ti]
        conds[ti] = capys_sum[ti]/conds_sum[ti]*capys[ti]
        # Time constants
        if capys_ord_idx[ti].size != 0:
            time_consts_inf[ti] = mean(time_const_glob[capys_ord_idx[ti]])
            time_const_sum[ti] = (
                    asum(time_const_glob[capys_ord_idx[ti]])
                    + time_consts_inf[ti]*len(capys_inf_idx[ti])
            )
        else:
            time_consts_inf[ti] = 1.
            time_const_sum[ti] = 1.*len(capys_inf_idx[ti])
        # TODO: include parts to time_consts, time_consts deactivated for now
        # time_consts[ti] = symbol_volus[ti]/3./time_const_sum[ti]

    # Global scale data: scale data over all time steps
    capy_inf_glob = mean(capys_inf)
    mean_capy_sum = mean(capys_sum)
    capy_glob = mean(symbol_volus)/3./mean_capy_sum
    mean_heats_sum = mean(heats_sum)
    if mean_heats_sum != 0.:
        heat_glob = mean_capy_sum/mean_heats_sum*capy_glob
    else:
        heat_glob = 0.
    cond_glob = mean_capy_sum/mean(conds_sum)*capy_glob

    time_const_inf_glob = mean(time_consts_inf)
    # TODO: include parts into time_const_glob, time_const_glob deactivated for now
    # time_const_glob = mean(symbol_volus)/3./mean(time_const_sum)

    return (capys_inf, capys, heats, conds, time_consts, time_consts_inf,
            capy_inf_glob, capy_glob, heat_glob, cond_glob, time_const_glob,
            time_const_inf_glob)


def params(
        res_data: ResultData,
        time: Optional[float] = None,
        symbol_size: float = .05,
        time_scale_type: str = SIM_SPAN,
        geo_scale_type: str = AREA,
        hide: tuple[str, ...] = (LUMP_NODE_IDX, AXES, LINK, HEAT_SRC_ELEM, EDGE),
        dpi: float = 95,
        draw_2d: bool = False,
        draw_elements: bool = True,
        color_theme: str = DEFAULT,
        colored_sfcs: bool = False,
        plotter: Optional[pv.Plotter] = None
):
    """
    Plots the parameters of the node net

    Draws model parameters as geometric primitives. The the magnitude
    of the parameter values correspond to the size of the geometric
    primitives.

    Args:
        res_data: result data
        time: Time; if None, last time from results are taken
        symbol_size: total_symbol_volume / total_bounding_box_volume
        time_scale_type: SIM_SPAN optimises the scaling of over the
            entire simulation time span; POINT_IN_TIME scales optimal
            for the given point in simulation time
        geo_scale_type: set to VOLU, if the value of the properties
            correspond to the volume of the symbols, set to AREA,
            if the value of the properties should correspond to the
            cross-sectional area of the symbols
        hide: plot objects to hide, the following are valid:
            CAPY, HEAT, COND, CONN, LUMP_NODE_IDX,
            COLOR_BAR, HEAT_SRC, TEMP as well as NAME, EDGE and AXES
            if the elements are drawn
        dpi: Screen resolution in dots per inch
        draw_2d: View of x-y-plane with parallel projection
        draw_elements: Specifies, if the surfaces of the elements are
            drawn
        color_theme: DEFAULT with grey background,
            BRIGHT with white background
        colored_sfcs: Draws lumped Part conductances in colors that
            match the colors of the element surfaces.
        plotter: PyVista plotter to draw in; if None a new plotter will
            be created
    Returns:
         PyVista Plotter
    """
    from thermca.fem.ffe_io import FFEIO

    time = res_data.times[-1] if time is None else time
    if time < res_data.times[0] or time > res_data.times[-1]:
        raise ValueError("'time' argument out of result time range!")

    net = res_data.network

    # # Drawing stuff
    clr_theme = color_themes[color_theme]
    if plotter is None:
        plotter = pv.Plotter()
    plotter.set_background(clr_theme['bg'])
    if draw_elements:
        elements(net.model, hide, dpi, draw_2d, color_theme, colored_sfcs, plotter)
    font_size, small_line, thick_line = pp._adjust_drawings_to_screen_dpi(dpi)

    ti = aabs(res_data.times - time).argmin()  # nearest result index for given time
    net_cond = res_data.net_conds.vix[ti]
    net_heat = res_data.net_heats.vix[ti]
    net_capy = res_data.net_capys[ti]
    net_temp = res_data.net_temps[ti]
    net_hsrc = res_data.net_heat_srcs[ti]
    net_posn = res_data.net_posns[ti]
    net_time_const = res_data.net_time_consts[ti]

    para_index_dat = _get_net_para_idxs(res_data)
    (conds_sym_idx, conds_asym_idx, capys_inf_idx, capys_zero_idx,
     capys_ord_idx, heat_srcs_idx) = para_index_dat

    lp_parts = net._lp_parts
    lp_conds = []
    lp_heats = []
    lp_capys = []
    lp_temps = []
    lp_temp_min = [float('nan')]
    lp_temp_max = [float('nan')]
    lp_src_heats = []
    lp_src_heat_idxs = []
    obs_src_heat_idxs = []
    if lp_parts:
        lp_ios = net._lp_ios
        # Iterate over result time points
        for M_diags, Ls, io_temp, net_clean_cond, net_temp, net_heat_src, net_posn in zip(
                res_data.lp_M_diagss,
                res_data.lp_Lss,
                res_data.io_temps,
                res_data.net_clean_conds,
                res_data.net_temps,
                res_data.net_heat_srcs,
                res_data.net_posns,
        ):
            # Collect params in flat arrays over all LPParts
            full_offsset = 0
            capy_offsset = 0  # Only internal capacities, without linked nodes
            lp_row, lp_col, lp_cond, lp_heat, lp_capy, lp_posn, lp_capy_posn, lp_src_heat, lp_src_heat_idx, obs_src_heat_idx = [], [], [], [], [], [], [], [], [], []
            obs_link_idxs0, obs_link_idxs1 = [], []
            io_temp_view = io_temp.view(dtype=res_data.io_temp_dt)
            lp_temp = io_temp_view[0]['lp']
            for lp_part, lp_io, M_diag, L, temp in zip(lp_parts, lp_ios, M_diags, Ls, lp_temp):
                row, col, cond, heat, capy, posn, src_heat, src_heat_idx, obs_net_src_heat_idx = get_lp_part_dat(
                    lp_part, lp_io, M_diag, L, temp, net_clean_cond, net_temp, net_heat_src, net_posn)
                # TODO: extract row, col, posn and redundant_link_idxs because they need only be processed once
                lp_capy_posn.append(posn[:len(capy)])
                lp_row.append(row + full_offsset)
                lp_col.append(col + full_offsset)
                lp_cond.append(cond)
                lp_heat.append(heat)
                lp_capy.append(capy)
                lp_posn.append(posn)
                lp_temp_min.append(amin(temp))
                lp_temp_max.append(amax(temp))
                obs_link_idxs0.append(lp_io.link_elem_net_idxs)
                obs_link_idxs1.append(lp_io.link_self_net_idxs)
                lp_src_heat.append(src_heat)
                lp_src_heat_idx.append(src_heat_idx + capy_offsset)
                obs_src_heat_idx.append(obs_net_src_heat_idx)
                full_offsset += len(posn)
                capy_offsset += len(capy)
            lp_conds.append(hstack(lp_cond))
            lp_heats.append(hstack(lp_heat))
            lp_capys.append(hstack(lp_capy))
            lp_temps.append(hstack(lp_temp))
            lp_src_heats.append(hstack(lp_src_heat))
            lp_src_heat_idxs.append(hstack(lp_src_heat_idx))
            obs_src_heat_idxs.append(hstack(obs_src_heat_idx))
            # Constant over time
            lp_row = hstack(lp_row)
            lp_col = hstack(lp_col)
            lp_posn = vstack(lp_posn)
            lp_capy_posn = vstack(lp_capy_posn)
            obs_link_idxs0 = hstack(obs_link_idxs0)
            obs_link_idxs1 = hstack(obs_link_idxs1)

        # Links with LPParts are drawn with LPParts not with the network
        # Therefore remove links from network
        lp_link_idxs = (
            hstack((obs_link_idxs0, obs_link_idxs1)),
            hstack((obs_link_idxs1, obs_link_idxs0)),
        )  # Could be in both directions
        conds_sym_idx = [
            sub_idxs_2d(idxs, lp_link_idxs)
            for idxs in conds_sym_idx
        ]

        # Heat sources on LPPart surfaces are drawn on margin nodes.
        # Therefore, remove heat source in network.
        heat_srcs_idx = [
            setdiff1d(idxs, obs_idxs)
            for idxs, obs_idxs in zip(heat_srcs_idx, obs_src_heat_idxs)
        ]
    lp_temp_glob_min = min(lp_temp_min)
    lp_temp_glob_max = max(lp_temp_max)

    cond_sym_idx = conds_sym_idx[ti]
    cond_asym_idx = conds_asym_idx[ti]
    capy_ord_idx = capys_ord_idx[ti]
    capy_inf_idx = capys_inf_idx[ti]
    capy_zero_idx = capys_zero_idx[ti]
    heat_src_idx = heat_srcs_idx[ti]

    # FEParts: initial lumped parameters are plotted
    # Time argument is therefore ignored for now
    fe_parts = net._fe_parts
    fe_ios = [net.fe_part_to_io[part] for part in fe_parts]
    fe_syss = [io.fe_sys for io in fe_ios]
    fe_capys = array([fe_sys.lump_capy for fe_sys in fe_syss])
    fe_surf_conds = [fe_sys.lump_surf_conds for fe_sys in fe_syss]

    (
        capys_inf_scale, capys_scale, heats_scale, conds_scale, time_consts_scale,
        time_consts_inf_scale, capy_inf_glob_scale, capy_glob_scale, heat_glob_scale,
        cond_glob_scale, time_const_glob_scale, time_const_inf_glob_scale
     ) = _calc_para_scale_dat(
        res_data,
        fe_parts,
        fe_capys,
        fe_surf_conds,
        lp_parts,
        lp_conds,
        lp_heats,
        lp_capys,
        para_index_dat,
        symbol_size,
        geo_scale_type
    )

    if time_scale_type == SIM_SPAN:
        temp_min = min(amin(res_data.net_temps), lp_temp_glob_min)
        temp_max = max(amax(res_data.net_temps), lp_temp_glob_max)
        capy_inf_scale = capy_inf_glob_scale
        capy_scale = capy_glob_scale
        heat_scale = heat_glob_scale
        cond_scale = cond_glob_scale
        # time_const_scale = time_const_glob_scale
        # time_const_inf_scale = time_const_inf_glob_scale
        # temp_scale = temp_glob_scale
    elif time_scale_type == POINT_IN_TIME:
        temp_min = min(amin(res_data.net_temps[ti]), lp_temp_min[ti])
        temp_max = max(amax(res_data.net_temps[ti]), lp_temp_max[ti])
        capy_inf_scale = capys_inf_scale[ti]
        capy_scale = capys_scale[ti]
        heat_scale = heats_scale[ti]
        cond_scale = conds_scale[ti]
        # time_const_scale = time_consts_scale[ti]
        # time_const_inf_scale = time_consts_inf_scale[ti]
    else:
        raise ValueError(f"'time_scale_type' argument {time_scale_type} is not supported!")

    if geo_scale_type == VOL:
        calc_capy_rad = lambda volus: (volus*3/(2*tau))**.3333
        calc_heat_rad = lambda volus: (6*aabs(volus)/(1.732*tau))**.3333  # 3**.5 = 1.732
        calc_cond_rad = lambda volus: (.6666*volus/tau)**.3333
    elif geo_scale_type == AREA:
        calc_capy_rad = lambda areas: .5*(4.*areas/pi)**.5
        calc_heat_rad = lambda areas: .5*(4.*aabs(areas)/1.732)**.5
        calc_cond_rad = lambda areas: (.1666*areas)**.5
    else:
        raise ValueError(f"'geo_scale_type' argument {geo_scale_type} is not supported!")

    # FEParts
    if fe_parts:
        part_capy_posns = []
        part_link_posns = []
        part_link_conns = []
        part_conds_colors = []
        start_idx = 0
        for part in fe_parts:
            part_dock = part._net_dat.posn + part._net_dat.dock_lctn
            part_capy_posns.append(part_dock)
            for i, surf in enumerate(part.surf):
                idx = start_idx + i
                part_link_posns.append(part_dock)
                surf_dock = surf._net_dat.posn + surf._net_dat.dock_lctn
                part_link_posns.append(surf_dock)
                part_link_conns.append((idx * 2, idx * 2 + 1))
                color = pp.tab10_colors[idx % len(pp.tab10_colors)]
                part_conds_colors.append(color)
            start_idx += i + 1

        part_capy_posns = array(part_capy_posns)
        part_link_posns = array(part_link_posns)
        part_link_conns = array(part_link_conns, dtype=int64)
        part_conds = hstack(fe_surf_conds)  # array(part_conds).ravel()

    # Colored capacity spheres
    capy_rad = zeros(len(res_data.net_capys[0]))
    capy_rad[capy_ord_idx] = calc_capy_rad(net_capy[capy_ord_idx] * capy_scale)
    capy_rad[capy_inf_idx] = calc_capy_rad(capy_scale * capy_inf_scale)
    capy_rad[capy_zero_idx] = 1e-16

    if CAPY not in hide:
        capy_sphere_posn = []
        capy_sphere_rad = []
        capy_sphere_temp = []

        # Network
        sphere_idxs = capy_ord_idx  # hstack((capy_ord_idx, capy_inf_idx))
        capy_sphere_posn.append(net_posn[sphere_idxs])
        capy_sphere_rad.append(capy_rad[sphere_idxs])
        capy_sphere_temp.append(net_temp[sphere_idxs])

        # FEParts
        if fe_parts:
            capy_sphere_posn.append(part_capy_posns)
            capy_sphere_rad.append(
                array([calc_capy_rad(capy * capy_scale) for capy in fe_capys]))
            # Temperature for now as mean surface temperatures
            view = res_data.io_temps.ravel().view(dtype=res_data.io_temp_dt)
            fe_part_capy_temps = []
            for fe_io in fe_ios:
                if isinstance(fe_io, FFEIO):
                    ffe_io = fe_io
                    temps = view['ffe'][str(net._ffe_ios.index(ffe_io))]
                else:
                    mor_io = fe_io
                    temps = view[:]['mor'][str(net._mor_ios.index(mor_io))]
                    s = temps.shape
                    temps = temps.reshape(s[0], s[1] * s[2])
                    temps = hstack(mor_io.VT1) @ temps.T + mor_io.ref_temp
                    temps = temps.T
                mean_temps = fe_io.fe_sys.C_body @ temps.T
                fe_part_capy_temps.append(mean_temps[ti])
            capy_sphere_temp.append(array(fe_part_capy_temps))

        # LPParts
        if lp_parts:
            capy_sphere_posn.append(lp_capy_posn)
            lp_capy_rad = calc_capy_rad(lp_capys[ti] * capy_scale)
            capy_sphere_rad.append(lp_capy_rad)
            capy_sphere_temp.append(lp_temps[ti])

        capy_sphere_posn = vstack(capy_sphere_posn)
        capy_sphere_rad = hstack(capy_sphere_rad)
        capy_sphere_temp = hstack(capy_sphere_temp)

        # Capacity spheres and temperature as colored intensity
        if TEMP not in hide:
            pp.glowing_scaled_spheres(
                capy_sphere_posn,
                capy_sphere_rad,
                capy_sphere_temp,
                clim=(temp_min, temp_max),
                show_color_bar=False,
                plotter=plotter,
            )
            # Capacity points
            pp.colored_points(
                net_posn[capy_zero_idx],
                net_temp[capy_zero_idx],
                thick_line * 5,
                clim=(temp_min, temp_max),
                show_color_bar=False,
                plotter=plotter,
            )
            if COLOR_BAR not in hide:
                pp.color_bar("Temperature", font_size, clim=(temp_min, temp_max), plotter=plotter)
        else:
            pp.scaled_spheres(
                capy_sphere_posn,
                capy_sphere_rad,
                color=clr_theme['temp_clamp'],
                plotter=plotter
            )
            # Capacity points
            pp.points(
                net_posn[capy_zero_idx],
                clr_theme['temp_clamp'],
                thick_line * 3,
                plotter=plotter
            )
        # Temperature clamping
        pp.glowing_temp_clamping(
            net_posn[capy_inf_idx],
            capy_rad[capy_inf_idx],
            net_temp[capy_inf_idx],
            clim=(temp_min, temp_max),
            plotter=plotter,
        )

    # Network
    # conductance lines from capacity to capacity
    # draw asymmetric conductances in both directions
    if cond_sym_idx or cond_asym_idx:
        link_idxs0 = hstack([idxs[0] for idxs in [cond_sym_idx, cond_asym_idx] if idxs])
        link_idxs1 = hstack([idxs[1] for idxs in [cond_sym_idx, cond_asym_idx] if idxs])
        cond_is_sym = hstack([
            full(len(idxs[0]), is_sym)
            for idxs,  is_sym
            in zip([cond_sym_idx, cond_asym_idx], [True, False])
            if idxs
        ])
        draw_link_idxs = column_stack((link_idxs0, link_idxs1))

        # heat flow cones:
        # Don't show forced heat flow because the heat flow is dependent on
        # the absolute temperature. In contrast, the heat flow over 'real'
        # conductances is dependent on temperature differences. So it makes no
        # sense to plot the heat flow value of forced flows in the current
        # visualisation context. Additionally, the forced heat flows are
        # typically much bigger than the 'normal' ones.
        # The conductance of forced heat flows is a capacity flow:
        # mass flow * spec. heat capacity.
        # This value is plotted as a directed conductance.
        heat_flow = net_heat.aix[link_idxs0, link_idxs1]
        sym_heat_flow = where(cond_is_sym, heat_flow, 0.)
        # also don't draw inf heat flow at inf conductance
        sym_heat_flow = where(isfinite(sym_heat_flow), sym_heat_flow, 0.)
        forward_flow = (sym_heat_flow > 0)[:, None]
        posn0 = where(forward_flow, net_posn[link_idxs0], net_posn[link_idxs1])
        posn1 = where(forward_flow, net_posn[link_idxs1], net_posn[link_idxs0])
        dirn = posn1 - posn0
        lgth = sqrt((dirn*dirn).sum(axis=1))[:, None]  # node-node direction
        xdir = array([1, 0, 0])
        with errstate(invalid='ignore'):
            heat_dirn = where(lgth == 0., xdir, dirn/lgth)
        heat_rad = calc_heat_rad(sym_heat_flow*heat_scale)
        heat_lgth = 1.732*heat_rad
        # to prevent lut manager errors in case of small lengths in mlab.plot3d, length should be at least 1e-16
        heat_lgth = where(heat_lgth == 0., 1e-16, heat_lgth)

        # conductance cylinders
        c = net_cond.aix[link_idxs0, link_idxs1]
        c[isinf(c)] = 0.  # don't draw infinity conductance of MergeLink
        cond_dirn = -heat_dirn
        cond_rad = calc_cond_rad(c*cond_scale)
        cond_lgth = 3*cond_rad
        heat_posn = cond_posn = posn0 + .5*dirn - heat_dirn*(.5*(cond_lgth + heat_lgth) - cond_lgth)[:, None]

        if CONN not in hide:  # conductance lines
            pp.lines(net_posn, draw_link_idxs, small_line, clr_theme['cond_line'], opacity=1, plotter=plotter)
        # Conductance cylinders and heat cones
        if COND not in hide:
            pp.cond_cylinders(cond_posn, cond_rad, cond_lgth, cond_dirn, color=clr_theme['cond_cyl'], opacity=.85,
                              plotter=plotter)
        if HEAT not in hide:
            # Heat cones
            pp.tube_cones(heat_posn, heat_rad, heat_lgth, heat_dirn, color=clr_theme['heat_cone'], opacity=.85,
                          plotter=plotter)
            # Heat sources
            heat_src_rad = calc_heat_rad(net_hsrc[heat_src_idx] * heat_scale)
            heat_src_posn = array([]) if HEAT_SRC in hide else net_posn[heat_src_idx]
            heat_src_dirn = array([.7071, .7071, 0]) if draw_2d else array([.577, .577, -.577])
            pp.heat_sources(
                heat_src_posn,
                capy_rad[heat_src_idx],
                heat_src_rad,
                heat_src_dirn,
                thick_line,
                heat_color=clr_theme['heat_cone'],
                line_color=clr_theme['heat_cone'],
                stripe_color=clr_theme['heat_cone_stripe'],
                opacity=.85,
                plotter=plotter,
            )
            # Arrows for forced heat flow
            asym_start = len(cond_sym_idx[0]) if cond_sym_idx else 0
            pp.forced_flow_arrows(
                cond_posn[asym_start:],
                cond_rad[asym_start:],
                cond_lgth[asym_start:],
                cond_dirn[asym_start:],
                color=clr_theme['forced_heat'],
                opacity=.85,
                plotter=plotter,
            )

    if lp_parts:
        if CONN not in hide:  # conductance lines
            pp.lines(
                lp_posn,
                column_stack((lp_row, lp_col)),
                small_line,
                clr_theme['cond_line'],
                opacity=1,
                plotter=plotter
            )

        # Heat flow cones
        # Indices of inner conductances
        forward_flow = (lp_heats[ti] > 0)[:, None]
        posn0 = where(forward_flow, lp_posn[lp_row], lp_posn[lp_col])
        posn1 = where(forward_flow, lp_posn[lp_col], lp_posn[lp_row])
        dirn = posn1 - posn0
        lgth = sqrt((dirn * dirn).sum(axis=1))[:, None]  # node-node direction
        xdir = array([1, 0, 0])
        with errstate(invalid='ignore'):
            heat_dirn = where(lgth == 0., xdir, dirn / lgth)
        heat_rad = calc_heat_rad(lp_heats[ti] * heat_scale)
        heat_lgth = 1.732 * heat_rad
        # To prevent lut manager errors, length should be at least 1e-16
        heat_lgth = where(heat_lgth < 1e-16, 1e-16, heat_lgth)

        # Conductance cylinders
        cond_dirn = -heat_dirn
        cond_rad = calc_cond_rad(lp_conds[ti] * cond_scale)
        cond_lgth = 3 * cond_rad
        heat_posn = cond_posn = posn0 + .5 * dirn - heat_dirn * (.5 * (cond_lgth + heat_lgth) - cond_lgth)[:, None]

        # Draw heat and conductance
        if COND not in hide:
            pp.cond_cylinders(cond_posn, cond_rad, cond_lgth, cond_dirn, color=clr_theme['cond_cyl'], opacity=.85,
                              plotter=plotter)

        if HEAT not in hide:
            pp.tube_cones(heat_posn, heat_rad, heat_lgth, heat_dirn, color=clr_theme['heat_cone'], opacity=.85,
                          plotter=plotter)

            heat_src_dirn = array([.7071, .7071, 0]) if draw_2d else array([.577, .577, -.577])
            lp_src_idxs = lp_src_heat_idxs[ti]
            pp.heat_sources(
                array([]) if HEAT_SRC in hide else lp_capy_posn[lp_src_idxs],
                zeros(len(lp_src_idxs)) if CAPY in hide else lp_capy_rad[lp_src_idxs],
                full(len(lp_src_idxs), calc_heat_rad(lp_src_heats[ti] * heat_scale)),
                heat_src_dirn,
                thick_line,
                heat_color=clr_theme['heat_cone'],
                line_color=clr_theme['heat_cone'],
                stripe_color=clr_theme['heat_cone_stripe'],
                opacity=.85,
                plotter=plotter,
            )

    if fe_parts:
        conn_pts = part_link_posns[part_link_conns]
        pcond_dirn = conn_pts[:, 1] - conn_pts[:, 0]
        # When points overlay each other --> dirn. == [0, 0, 0]
        # To give a direction and prevent division by zero set dirn. to [1, 0, 0]
        pcond_dirn[(pcond_dirn == 0.).all(axis=1), :] = array([1., 0., 0.])
        conn_lgth = sqrt((pcond_dirn * pcond_dirn).sum(axis=1))  # node-node direction
        part_cond_dirn = pcond_dirn/conn_lgth[:, None]
        part_cond_rad = calc_cond_rad(part_conds * cond_scale)
        part_cond_lgth = 3 * part_cond_rad
        part_cond_posn = conn_pts[:, 0] + .5*pcond_dirn - part_cond_dirn*(part_cond_lgth*.5)[:, None]
        # Draw
        pp.lines(part_link_posns, part_link_conns, small_line, clr_theme['cond_line'], opacity=1, plotter=plotter)
        if COND not in hide:
            if colored_sfcs:
                pp.colored_cond_cylinders(
                    part_cond_posn,
                    part_cond_rad,
                    part_cond_lgth,
                    part_cond_dirn,
                    array(part_conds_colors),
                    opacity=1.,
                    plotter=plotter
                )
            else:
                pp.cond_cylinders(
                    part_cond_posn,
                    part_cond_rad,
                    part_cond_lgth,
                    part_cond_dirn,
                    color=clr_theme['cond_cyl'],
                    opacity=.85,
                    plotter=plotter
                )

    if LUMP_NODE_IDX not in hide:
        pp.labels(net_posn, arange(len(net_posn)), clr_theme['fg'], clr_theme['text_bg'], font_size, plotter)

    pp._set_initial_view(plotter, draw_2d)
    return plotter


def sub_idxs_2d(a, b):
    """Subtract index pairs given in b from a"""
    diff = array(
        list(set(map(tuple, array(a).T))
             - set(map(tuple, array(b).T)))
    ).T
    if diff.size == 0:
        return ()
    return diff[0], diff[1]


if __name__ == '__main__':
    from thermca import *
    import pyvista as pv

    simple_block = Mesh(
        points=array([
            [0.8, 0.8, 0.8, 0.8, 0., 0., 0., 0., 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
            [0.6, 0.6, 0., 0., 0.6, 0.6, 0., 0., 0.6, 0., 0.6, 0., 0.3, 0.3],
            [0., 0.4, 0., 0.4, 0., 0.4, 0., 0.4, 0.4, 0.4, 0., 0., 0.4, 0.]
        ]).T,
        cell_blocks=[
            array([
                [2, 13, 2, 8, 12, 13, 2, 8, 8, 13, 12, 8, 13, 12, 12, 13, 12, 8],
                [12, 5, 12, 13, 11, 7, 12, 13, 1, 7, 11, 13, 7, 1, 11, 5, 11, 13],
                [13, 4, 0, 0, 13, 5, 3, 5, 0, 11, 7, 10, 6, 0, 9, 6, 2, 12],
                [0, 10, 3, 10, 7, 12, 9, 12, 12, 6, 9, 5, 5, 3, 2, 4, 13, 0]]).T,
            array([
                [6, 3, 9, 11],
                [11, 9, 7, 2],
                [7, 2, 11, 9]]).T,
            array([
                [6, 11, 4, 10, 2, 13],
                [4, 6, 10, 0, 11, 0],
                [13, 13, 13, 13, 13, 2]]).T,
            array([
                [3, 1],
                [2, 3],
                [0, 0]]).T,
            array([
                [10, 10, 5, 1],
                [4, 8, 8, 0],
                [5, 0, 10, 8]]).T,
            array([
                [6, 5],
                [7, 4],
                [5, 6]]).T,
            array([
                [5, 8, 7, 1, 9, 3],
                [7, 5, 9, 8, 3, 1],
                [12, 12, 12, 12, 12, 12]]).T
        ],
        block_types=['tetra', 'triangle', 'triangle', 'triangle', 'triangle', 'triangle', 'triangle'],
        block_names=['noname1', 'bottom', 'back', 'right', 'top', 'left', 'front'],
    )
    mesh = Mesh.read("tests/flat_cuboid_fine.cgns")  # surfaces: 'boden', 'luft'

    with Model() as model:
        cuboid = FEPart(mesh, solids.al_alloy, init_temp=20., name='cuboid')
        env = BoundNode(temp=20., name='environment', posn=(-.3, .5, .025))
        HeatSource(cuboid.surf.boden, 1000.)
        FilmLink(
            cuboid.surf.luft,  # cuboid.sfc.front, cuboid.sfc.back, cuboid.sfc.left, cuboid.sfc.right, cuboid.sfc.top],
            env,
            10.
        )
        '''
        body0 = Node(
            capy=1.,
            posn=(1., 1.),
            name='body0')
        body1 = LinkNode(
            #capy=.5,
            posn=(.0,),
            name='body1')
        CondLink(
            body0,
            body1,
            cond=.5)

        HeatSource(
            body1,
            heat=.5)
        body2 = Node(
            capy=.75,
            posn=(0., 1.),
            name='body2')
        HeatSource(
            body2,
            heat=1.)

        env0 = Node.bound(
            temp=1.,
            posn=(2., 2.),
            name='bound_temp0')

        env1 = Node.bound(
            temp=1.,
            posn=(0., 2.),
            name='bound_temp1')        

        CondLink(
            body0,
            body2,
            cond=.75)
        CondLink(
            body1,
            env0,
            cond=.1)
        CondLink(
            body2,
            env1,
            cond=.1)
        Block(
            posn=(0., 0.),
            matl=solids.steel,
            width=.5,
            hgt=.5,
            depth=.5
        )
        Block(
            posn=(.5, 0.),
            matl=solids.steel,
            width=.5,
            hgt=.5,
            depth=.5,
        )
        Block(
            posn=(-.5, 0.),
            matl=solids.steel,
            width=.5,
            hgt=.5,
            depth=.5,
        )
        Block(
            posn=(0., .5),
            matl=solids.steel,
            width=.5,
            hgt=.5,
            depth=.5,
        )
        Block(
            posn=(0., -.5),
            matl=solids.steel,
            width=.5,
            hgt=.5,
            depth=.5,
        )
        b0 = Block(
            posn=(0., 0., .5),
            matl=solids.steel,
            width=.5,
            hgt=.5,
            depth=.5,
        )
        b1 = Block(
            posn=(0., 0, -.5),
            matl=solids.steel,
            width=.5,
            hgt=.5,
            depth=.5,
        )
        HeatSource(
            b0, 1,
        )
        HeatSource(
            b1, 2,
        )
        
        Cyl(
            posn=(1.5, .5),
            matl=solids.steel,
            inner_rad=.2,
            outer_rad=1.,
            lgth=1.,
            rad_div=2,
            lgth_div=2,
            name='Cyl0'
        )
        Cyl(
            posn=(-1.5, .5),
            matl=solids.steel,
            #inner_rad=.2,
            outer_rad=1.,
            lgth=1.,
            rad_div=1,
            lgth_div=1,
            name='Cyl1'
        )  
        # Water with source and sink
        MatlNode.matl = fluids.water
        cs_area = .05
        vol = cs_area*.5
        MatlNode.init_temp = 20
        source = MatlNode.bound(
            temp=20.,
            posn=(.0,),
            name='Source')
        MatlNode.is_temp_dep = True
        water0 = MatlNode(
            posn=(.125, ),
            vol=vol,
            name='Water0')
        water1 = MatlNode(
            posn=(.25 + .125, ),
            vol=vol,
            name='Water1')
        FlowLink(
            water0, source, .0001)
        FlowLink(
            source, water0, .00001)
        FlowLink(
            water1, water0, .0001)
        FlowLink(
            water0, water1, .00001)
    '''
    network = Network(model)
    result = network.sim((0., 1.))

    pl = pv.Plotter()
    #elements(model, figure=fig, dpi=230)
    params(
        time=1,
        res_data=result.condensed().data,
        #symbol_size=.05,
        time_scale_type=SIM_SPAN,
        hide=( CAPY, AXES),
        geo_scale_type=AREA,
        # dpi=160,
        draw_2d=True,
        # color_theme=BRIGHT,
        plotter=pl,
    )
    # pl.show()
    Mesh.read("../../sandbox/test_block_tilted.med").plot(
        plotter=pl,
        # dpi=160,
        # color_theme=BRIGHT,
    ).show()

