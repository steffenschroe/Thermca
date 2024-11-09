from __future__ import annotations
from math import tau
from typing import TYPE_CHECKING, Optional

from numpy import array, abs as aabs, zeros, full, mean, hstack, vstack, arange, where
from numpy import sqrt, errstate, column_stack
from scipy.sparse import tril as stril
from scipy import spatial
import pyvista as pv

from thermca.plot import primitives as pp
from thermca.plot.primitives import CAPY, HEAT, COND, CONN, TEMP, COLOR_BAR
from thermca.plot.primitives import HEAT_SRC, VOL, AREA, color_themes, AXES
from thermca.plot.primitives import glowing_temp_clamping

if TYPE_CHECKING:
    from thermca.lpm.lp_system import StaticResult


def static_result(
        result: StaticResult,
        symbol_size: float = .05,
        hide: tuple[str, ...] = (AXES, ),
        geo_scale_type: str = AREA,
        dpi: float = 95,
        draw_asm: bool = True,
        draw2d: bool = False,
        color_theme: str = pp.DEFAULT,
        plotter: Optional[pv.Plotter] = None
):
    from thermca._utils.sparse import to_csr_data_idxs
    from thermca.plot.asm import bodies_and_surfs

    clr_theme = color_themes[color_theme]
    if plotter is None:
        plotter = pv.Plotter()
    plotter.set_background(clr_theme['bg'])
    if draw_asm:
        bodies_and_surfs(result.lp_sys.asm, hide=hide, dpi=dpi, color_theme=color_theme, plotter=plotter)
    font_size, small_line, thick_line = pp._adjust_drawings_to_screen_dpi(dpi)

    (
        bc_temp_idxs, bc_temps,
        bc_heat_idxs, bc_heats,
        bc_film_idxs, bc_films, bc_film_temps,
    ) = result.flat_bcs

    # # Build the plot system

    # Plot the whole system for better understanding of the mechanics of the lp system
    lp_sys = result.lp_sys
    # Save conductances as a lower triangle matrix to prevent double entries
    # Include margins on surfaces with BCs and film conductances
    # First inner conductances
    inner_conds = -stril(lp_sys.L[:lp_sys.dof], -1).tocoo()
    # conds_sym_idx = one_dir_conds.nonzero()
    row = [inner_conds.row]
    col = [inner_conds.col]
    cond_data = [inner_conds.data]
    heat_data = [inner_conds.data * (result.temps[inner_conds.row] - result.temps[inner_conds.col])]
    # Second BC conductances margin to surface
    for surf_idx, surf_temp in zip(bc_temp_idxs, bc_temps):
        row_idxs = lp_sys.surf_dof_idxs[surf_idx]
        col_idxs = lp_sys.marg_dof_idxs[surf_idx]
        row.append(row_idxs)
        col.append(col_idxs)
        data_idxs = lp_sys.surf_to_marg_data_idxs[surf_idx]
        # data_idxs = to_csr_data_idxs(lp_part.L, (row_idxs, col_idxs))
        marg_conds = -lp_sys.L.data[data_idxs]
        cond_data.append(marg_conds)
        heat_data.append(marg_conds * (surf_temp - result.temps[col_idxs]))

    # Film dof indices as single nodes to the end
    num_cond_dof = lp_sys.L.shape[0]
    film_dof_idxs = arange(len(bc_films)) + num_cond_dof
    for surf_idx, film, film_temp, film_dof_idx in zip(
            bc_film_idxs,
            bc_films,
            bc_film_temps,
            film_dof_idxs
    ):
        row_idxs = lp_sys.surf_dof_idxs[surf_idx]
        col_idxs = lp_sys.marg_dof_idxs[surf_idx]
        # Margin conductances
        row.append(row_idxs)
        col.append(col_idxs)
        data_idxs = to_csr_data_idxs(lp_sys.L, (row_idxs, col_idxs))
        marg_conds = -lp_sys.L.data[data_idxs]
        cond_data.append(marg_conds)
        # Film conductances
        film_conds = lp_sys.surf_face_areas[surf_idx] * film
        row.append(full(len(row_idxs), film_dof_idx))
        col.append(row_idxs)
        cond_data.append(film_conds)
        # Heat
        marg_film_conds = 1./(1./marg_conds + 1./film_conds)
        marg_film_heats = marg_film_conds * (film_temp - result.temps[col_idxs])
        heat_data.append(marg_film_heats)  # For margin
        heat_data.append(marg_film_heats)  # Same heat for film

    row = hstack(row)
    col = hstack(col)
    cond_data = hstack(cond_data)
    heat_data = hstack(heat_data)

    # Add positions of film nodes
    posns = lp_sys.posns
    film_posns = []
    # Positions for film node as extension of margin connection
    for surf_idx in bc_film_idxs:
        surf_dof_idxs = lp_sys.surf_dof_idxs[surf_idx]
        sp = posns[surf_dof_idxs]
        center_point_idx = spatial.KDTree(sp).query(sp.mean(axis=0))[1]
        surf_posn = posns[surf_dof_idxs[center_point_idx]]
        marg_posn = posns[lp_sys.marg_dof_idxs[surf_idx][center_point_idx]]
        # film_node_posn = surf_posn + (surf_posn - marg_posn)
        film_posns.append(surf_posn + (surf_posn - marg_posn))
    posns = vstack((posns, array(film_posns)))

    # # Scaling of plot symbols
    if geo_scale_type == VOL:
        draw_geos = lp_sys.vol
    elif geo_scale_type == AREA:
        draw_geos = (lp_sys.vol**.333)**2.
    else:
        raise ValueError(f"'geo_scale_type' argument {geo_scale_type} is not supported!")
    symbol_volume = draw_geos * symbol_size

    # Give capacity, conductance and heat symbols 1/3 of symbol volume each
    capy_scale = symbol_volume / (3. * sum(lp_sys.M_diag))
    cond_scale = symbol_volume / (3. * sum(cond_data))
    heat_scale = symbol_volume / (3. * sum(abs(heat_data)))

    if geo_scale_type == VOL:
        calc_capy_rad = lambda volus: (volus*3/(2*tau))**.3333
        calc_heat_rad = lambda volus: (6*aabs(volus)/(1.732*tau))**.3333  # 3**.5 = 1.732
        calc_cond_rad = lambda volus: (.6666*volus/tau)**.3333
    elif geo_scale_type == AREA:
        calc_capy_rad = lambda areas: .5*(4.*areas/(tau/2.))**.5
        calc_heat_rad = lambda areas: .5*(4.*aabs(areas)/1.732)**.5
        calc_cond_rad = lambda areas: (.1666*areas)**.5
    else:
        raise ValueError(f"'geo_scale_type' argument {geo_scale_type} is not supported!")

    # # Draw symbols

    # Conductance lines from capacity to capacity
    if CONN not in hide:
        # Combine link indices
        pp.lines(
            posns,
            column_stack((row, col)),
            small_line,
            clr_theme['cond_line'],
            opacity=1,
            plotter=plotter
        )

    # Heat flow cones
    # Indices of inner conductances
    forward_flow = (heat_data > 0)[:, None]
    posn0 = where(forward_flow, posns[row], posns[col])
    posn1 = where(forward_flow, posns[col], posns[row])
    dirn = posn1 - posn0
    lgth = sqrt((dirn * dirn).sum(axis=1))[:, None]  # node-node direction
    xdir = array([1, 0, 0])
    with errstate(invalid='ignore'):
        heat_dirn = where(lgth == 0., xdir, dirn / lgth)
    heat_rad = calc_heat_rad(heat_data * heat_scale)
    heat_lgth = 1.732 * heat_rad
    # To prevent lut manager errors, length should be at least 1e-16
    heat_lgth = where(heat_lgth < 1e-16, 1e-16, heat_lgth)

    # Conductance cylinders
    cond_dirn = -heat_dirn
    cond_rad = calc_cond_rad(cond_data * cond_scale)
    cond_lgth = 3 * cond_rad
    heat_posn = cond_posn = posn0 + .5 * dirn - heat_dirn * (.5 * (cond_lgth + heat_lgth) - cond_lgth)[:, None]

    # Draw heat and conductance
    if COND not in hide:
        pp.cond_cylinders(cond_posn, cond_rad, cond_lgth, cond_dirn, color=clr_theme['cond_cyl'], opacity=.85, plotter=plotter)  # opacity=.85,

    if HEAT not in hide:
        pp.tube_cones(heat_posn, heat_rad, heat_lgth, heat_dirn, color=clr_theme['heat_cone'], opacity=.85, plotter=plotter)

    # Capacity spheres with temperature glow
    capy_rad = calc_capy_rad(lp_sys.M_diag * capy_scale)
    if CAPY not in hide:
        if TEMP not in hide:
            # Don't draw 'zero' capacity nodes on surfaces
            clim = (min(result.temps), max(result.temps))
            # Ordinary nodes as spheres
            pp.glowing_scaled_spheres(
                lp_sys.posns[:lp_sys.dof],
                capy_rad,
                result.temps,
                clim=clim,
                show_color_bar=False,
                plotter=plotter,
            )

            if COLOR_BAR not in hide:
                pp.color_bar("Temperature", font_size, clim=clim, plotter=plotter)
        else:
            pp.scaled_spheres(
                lp_sys.posns[:lp_sys.dof],
                capy_rad,
                color=clr_theme['temp_clamp'],
                plotter=plotter
            )
        # Temperature clamping of temperature bcs
        # posn[temp_bc_dof_idxs]
        cone_hgt = mean(capy_rad)
        for surf_idx, bc_temp in zip(bc_temp_idxs, bc_temps):
            dof_idxs = lp_sys.surf_dof_idxs[surf_idx]
            glowing_temp_clamping(
                posns[dof_idxs],
                full(len(dof_idxs), cone_hgt),
                full(len(dof_idxs), bc_temp),
                clim,
                plotter=plotter,
            )
        # Temperature clamping of film bcs
        glowing_temp_clamping(
            posns[film_dof_idxs],
            full(len(bc_film_temps), cone_hgt),
            bc_film_temps,
            clim,
            plotter=plotter,
        )

    # Heat sources
    if HEAT_SRC not in hide:
        heat_src_dirn = array([.7071, .7071, 0]) if draw2d else array([.577, .577, -.577])

        for surf_idx, bc_heat in zip(bc_heat_idxs, bc_heats):
            dof_idxs = lp_sys.marg_dof_idxs[surf_idx]
            pp.heat_sources(
                posns[dof_idxs],
                zeros(len(dof_idxs)) if CAPY in hide else capy_rad[dof_idxs],
                full(len(dof_idxs), calc_heat_rad(bc_heat/len(dof_idxs) * heat_scale)),
                heat_src_dirn,
                thick_line,
                heat_color=clr_theme['heat_cone'],
                line_color=clr_theme['heat_cone'],
                stripe_color=clr_theme['heat_cone_stripe'],
                opacity=.85,
                plotter=plotter,
            )

    pp._set_initial_view(plotter, draw2d)
    return plotter


if __name__ == '__main__':
    from thermca.lpm.asm import Asm
    from thermca.lpm.cube import Cube
    from thermca.lpm.cyl import Cyl
    from thermca.lpm.asm import Surf, ForceConts
    from thermca.lpm.lp_system import LPSystem
    from thermca.materials import Solid
    from thermca.static_bcs import TempBC, HeatBC, FilmBC, FluxBC

    with Asm() as asm:
        left_block = Cube(
            'left_block',
            posn=(0., 0., 0.),
            width=2.,
            width_div=5,
            hgt=3.,
            hgt_div=2,
            depth=4.,
            depth_div=3,
        )
        right_block = Cube(
            'right_block',
            posn=(left_block.width, 0., 0.),
            width=2.,
            width_div=5,
            hgt=3.,
            hgt_div=2,
            depth=4.,
            depth_div=3,
        )
        left = Surf(
            name='left',
            faces=[left_block.face.left]
        )
        right = Surf(
            name='right',
            faces=[right_block.face.right]
        )
        top = Surf(
            name='top',
            faces=[right_block.face.top]
        )

    test_matl = Solid(
        condy=1.5,
        dens=1.,
        spec_heat=1.)

    # pl = asm.plot(dpi=250)
    part = LPSystem(asm, 0., test_matl)
    result = part.solve([HeatBC(left, 1.), TempBC(right, 1.), FilmBC(top, 1., 1.)])
    static_result(result, dpi=250).show()

