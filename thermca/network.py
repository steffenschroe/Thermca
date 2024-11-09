"""Solves the model"""
from collections import defaultdict, namedtuple
from itertools import chain
import threading
import warnings
from typing import Union, Tuple, List
from functools import wraps

from numpy import zeros, ones, mean, array, int64, float64, hstack, arange, unique
from scipy import sparse
from sparse import DOK

from thermca.pointnodes import Node, StatNode, MatlNode, BoundNode
from thermca.fem.fe_part import FEPart, FEPartSurf
from thermca.fem.fe_system import FESystem
from thermca.fem.ffe_io import FFEIO
from thermca.fem.mor_io import MORIO
from thermca.lpm.lp_io import LPIO
from thermca.lpm.lp_part import LPPart, LPPartSurf
from thermca.lpm.lp_system import LPSystem
from thermca.links import CondLink, FilmLink, FlowLink
from thermca.source import HeatSource, FluxSource
from thermca.model import Model
from thermca.progressbar import TextProgressBar
from thermca.resultdata import ResultData, Result, _ResultCollector
import thermca.solver as solver
from thermca.input import Input
from thermca._utils.sparse import to_csr_data_idxs
import thermca.plot.result as plot

PartSurf = FEPartSurf | LPPartSurf

_SfcSfcDocks = namedtuple('_SfcSfcDocks', ['body0', 'surf0', 'surf1', 'body1'])

_SfcNodeDocks = namedtuple('_SfcNodeDocks', ['body', 'surf', 'node'])


def _get_link_elems(model):
    return {
        CondLink: model._get_all_elems_of_types(CondLink),
        FilmLink: model._get_all_elems_of_types(FilmLink),
        FlowLink: model._get_all_elems_of_types(FlowLink),
    }


def _get_boundary_elems(model):
    """Get elements with boundary conditions. Boundary conditions are
    bound temperatures as well as heat and flux sources."""
    return {
        HeatSource: model._get_all_elems_of_types(HeatSource),
        FluxSource: model._get_all_elems_of_types(FluxSource),
        BoundNode: model._get_all_elems_of_types(BoundNode),
    }


def _get_sorted_node_elems(model, all_temp_bound_elems):
    """Returns a list of all node elements of a Net sorted by element
    types.

    The elements are ordered to enable time optimisations e.g. in
    solution process.

    Args:
        model (Model)
    Returns:
        dict[List[NodeElements]]: dict of node element lists, keys
        are element types
    """
    # The network object builds an equation system to simulate the
    # temperature field over time.
    # ⎡C_uu C_uk⎤ ⎧Ṫ_u⎫ + ⎡L_uu L_uk⎤ ⎧T_u⎫ = ⎧Q̇_u⎫
    # ⎣C_ku C_kk⎦ ⎩Ṫ_k⎭   ⎣L_ku L_kk⎦ ⎩T_k⎭   ⎩Q̇_k⎭
    # Nodes with free (Unknown) temperatures T_u have to be on the
    # first rows. Afterward nodes with bound temperatures (Known) T_k
    # are inserted. All stationary nodes with zero capacities like
    # surfaces and stationary nodes are treated like bound
    # temperatures. Their temperatures are calculated in a preceding
    # pass. Surface nodes on adiabatic surfaces are called unbound
    # surfaces. They are placed on the very end because they are not
    # needed to solve the temperature field, only for visualisation.

    node_elems = {}
    temp_unbound_node_elems = {}
    temp_bound_node_elems = {}
    for elem_type in (MatlNode, Node, StatNode, BoundNode, FEPartSurf, LPPartSurf):
        elems = model._get_all_elems_of_types(elem_type)
        # Bound and unbound in sense of solution process
        bound = []
        unbound = []
        for elem in elems:
            # Stationary nodes and part surfaces are also temp bound because they get
            # precalculated
            if elem in all_temp_bound_elems or isinstance(elem, (StatNode, PartSurf)):
                bound.append(elem)
            else:
                unbound.append(elem)
        node_elems[elem_type] = elems
        temp_unbound_node_elems[elem_type] = unbound
        temp_bound_node_elems[elem_type] = bound
    return node_elems, temp_unbound_node_elems, temp_bound_node_elems


def _get_input_element_set_methods(model):
    """Get time dependent input elements"""
    inputs = model._get_all_elems_of_types(Input)
    return [input._set_time for input in inputs]


class Network:
    """Creates a runnable node network out of the model.

    Args:
        model (Model): Net of elements
    """

    def __init__(self, model):
        if not isinstance(model, Model):
            raise TypeError('"model" argument must be of type Model')
        self.model = model
        self.time = None  # Only used during simulation, start value not known now

        self._boundary_elems = _get_boundary_elems(model)
        all_temp_bound_elems = set(self._boundary_elems[BoundNode])
        (
            self._node_elems,
            temp_unbound_node_elems,
            temp_bound_node_elems,
        ) = _get_sorted_node_elems(model, all_temp_bound_elems)
        self._link_elems = _get_link_elems(model)
        (
            self.elem_idx,
            self._node_count,
            self.elem_slice,
            self._free_temp_lgth,
            self.body_marg_idxs,
        ) = self._count_and_assign_elem_nodes(
            temp_unbound_node_elems, temp_bound_node_elems
        )
        self.input_set_methods = _get_input_element_set_methods(model)
        self.posn = zeros((self._node_count, 3))
        self.init_temp = zeros((self._node_count,))
        self._vol_capy = zeros((self._node_count,))
        self._vol = zeros((self._node_count,))
        self.capy = ones((self._node_count,)) * float("-inf")
        self.condy = zeros(self._node_count)
        self.cond = DOK((self._node_count, self._node_count))
        self.heat_src = zeros((self._node_count,))
        # fe_parts
        fe_parts = model._get_all_elems_of_types(FEPart)
        self._fe_parts = fe_parts
        fe_syss = [
            FESystem(
                part._body_mesh,
                part._surf_meshes,
                part.init_temp,
                part.matl,
                part.lump_cond_meth,
                part.fe_assembly,
            )
            for part in fe_parts
        ]
        # lp_parts
        lp_parts = model._get_all_elems_of_types(LPPart)
        self._lp_parts = lp_parts
        lp_syss = [LPSystem(part.asm, part.init_temp, part.matl) for part in lp_parts]

        self._sim_timer = None
        self._solve_speed = 0.0

        # Material elements
        matl_idxs = defaultdict(list)  # Node indices for every material
        # Node indices for temperature dependent material
        self._all_temp_dependent_matl_idxs = defaultdict(list)
        for matl_elem in self._node_elems[MatlNode]:
            matl_idxs[matl_elem.matl].extend(self.elem_idx[matl_elem])
            if matl_elem.temp_dependent:
                self._all_temp_dependent_matl_idxs[matl_elem.matl].extend(
                    self.elem_idx[matl_elem]
                )

        # All indices of temperature dependent material elements
        self._all_temp_dependent_matl_idx_values = array(
            list(chain.from_iterable(self._all_temp_dependent_matl_idxs.values())),
            dtype=int64,
        )

        # Node elements: names, positions, initial temperatures, volumes
        for node_elem_list in self._node_elems.values():
            for node_elem in node_elem_list:
                self.posn[self.elem_idx[node_elem], :] = (
                    node_elem._net_dat.posn + node_elem._net_dat.dock_lctn
                )
                if not isinstance(node_elem, BoundNode):
                    self.init_temp[self.elem_idx[node_elem]] = node_elem.init_temp
                if isinstance(node_elem, MatlNode):
                    self._vol[self.elem_idx[node_elem]] = node_elem.vol

        # Set material properties of MatlKnots, temperature field must be initialized
        for matl, idx in matl_idxs.items():
            temp = self.init_temp[idx]
            matl.check_limits(temp)
            self._vol_capy[idx] = matl.vol_capy_interp(temp)
            # self.condy[idx] = matl.condy_interp(temp)

        # Capacity of MatlNodes
        self.capy[:] = self._vol[:] * self._vol_capy[:]

        # Capacity of "ordinary" point node elements
        for ord_node_cls in (Node, BoundNode, StatNode):
            for node_elem in self._node_elems[ord_node_cls]:
                self.capy[self.elem_idx[node_elem]] = node_elem.capy

        run_cond_idxs0 = []
        run_cond_idxs1 = []

        # FlowLink
        flow_link_func_idxs = (defaultdict(list), defaultdict(list))
        flow_link_func_vol_flow = defaultdict(list)
        flow_link_func_fluid_info = defaultdict(list)
        flow_link_idxs = ([], [])
        flow_link_vol_flow = []
        for flow_link in self._link_elems[FlowLink]:
            idx0, idx1 = self.link_idx_pairs(flow_link)
            run_cond_idxs0.extend(idx0)
            run_cond_idxs1.extend(idx1)
            vol_flow = flow_link.vol_flow
            dest_node, src_node = flow_link.dest_node, flow_link.src_node
            # Time dependent calculation of function as well as volumetric capacity
            if callable(vol_flow):
                flow_link_func_idxs[0][vol_flow].extend(idx0)
                flow_link_func_idxs[1][vol_flow].extend(idx1)
                flow_link_func_vol_flow[vol_flow].append(vol_flow)
                # Store index of link with fluid
                # to later sort indices with the same fluids
                idx = len(flow_link_func_vol_flow[vol_flow])
                if flow_link.matl is not None:
                    if (
                        not isinstance(dest_node, BoundNode)
                        and dest_node.matl != flow_link.matl
                    ):
                        warnings.warn(
                            "FlowLink material does not match the material of the linked node!"
                        )
                    if (
                        not isinstance(src_node, BoundNode)
                        and src_node.matl != flow_link.matl
                    ):
                        warnings.warn(
                            "FlowLink material does not match the material of the linked node!"
                        )
                flow_link_func_fluid_info[vol_flow].append((flow_link.matl, idx - 1))
            # Time dependent calculation of function as well as volumetric capacity
            elif (
                not isinstance(dest_node, BoundNode) and dest_node.temp_dependent
            ) or (not isinstance(src_node, BoundNode) and src_node.temp_dependent):
                # already done: self._vol_capy[idx] = matl.vol_capy_interp(temp)
                flow_link_idxs[0].extend(idx0)
                flow_link_idxs[1].extend(idx1)
                flow_link_vol_flow.append(vol_flow)
            else:
                self.cond.aix[idx0, idx1] = (
                    vol_flow * (self._vol_capy[idx0] + self._vol_capy[idx1]) / 2
                )

        # Prepare for volumetric heat temperature dependent calculations
        self._flow_link_data = (
            array([], dtype=int64),  # sparse indexes filled later
            array(flow_link_idxs[0], dtype=int64),
            array(flow_link_idxs[1], dtype=int64),
            array(flow_link_vol_flow, dtype=float64),
        )
        # Prepare efficient FlowLink function calls during simulation
        # Combine indices with the same function and fluid
        flow_link_func_fluid_idxs = ({}, {})
        flow_link_func_fluid_flow = {}
        # Sort fluids
        for flow_func in flow_link_func_idxs[0]:
            idxs0 = flow_link_func_fluid_idxs[0][flow_func] = defaultdict(list)
            idxs1 = flow_link_func_fluid_idxs[1][flow_func] = defaultdict(list)
            vol_flow = flow_link_func_fluid_flow[flow_func] = defaultdict(list)
            for fluid, idx in flow_link_func_fluid_info[flow_func]:
                idxs0[fluid].append(flow_link_func_idxs[0][flow_func][idx])
                idxs1[fluid].append(flow_link_func_idxs[1][flow_func][idx])
                vol_flow[fluid].append(flow_link_func_fluid_flow[flow_func][idx])
        self._flow_link_func_data = {}
        for (flow_func, idxs0), (flow_func, idxs1), (flow_func, vol_flow) in zip(
            flow_link_func_fluid_idxs[0].items(),
            flow_link_func_fluid_idxs[1].items(),
            flow_link_func_fluid_flow.items(),
        ):
            self._flow_link_func_data[flow_func] = {}
            for fluid in idxs0:
                self._flow_link_func_data[flow_func][fluid] = (
                    array(idxs0[fluid], dtype=int64),
                    array(idxs1[fluid], dtype=int64),
                    array(vol_flow[fluid], dtype=float64),
                )

        # FilmLink and CondLink
        # Treat CondLink like FilmLink; for physical compatibility its
        # conductance is stored in 'film' and the 'area' is set to 1.
        # Collect film indices and node areas for every film function
        film_link_func_idxs = (defaultdict(list), defaultdict(list))
        film_link_func_areas = defaultdict(list)
        film_link_func_fluid_info = defaultdict(list)
        # Get all linked PartSfc needed for part creation
        linked_part_surf_idxs = defaultdict(list)  # Index to surf field of part
        linked_part_cond_idxs0 = defaultdict(list)
        linked_part_cond_idxs1 = defaultdict(list)
        # Links of LPPart to LPPart have a three-part series conductance of
        # margin, cond. or film and margin. This needs special handling.
        lp_to_lp_link_elems: list[tuple] = []  # Pairs of two connected LPParts
        lp_to_lp_surf_idxs: list[tuple] = []  # Pairs of part surface indices
        lp_to_lp_link_idxs_idxs: list[
            tuple
        ] = []  # Pairs of indices into part surface index array
        # Same as above but for run-time changing links
        self.run_lp_to_lp_link_elems: list[tuple] = []
        self.run_lp_to_lp_surf_idxs: list[tuple] = []
        self.run_lp_to_lp_link_idxs_idxs: list[tuple] = []
        # Because film coefficients get included in parts,
        # parts have to change if films during simulation
        var_film_parts = set()
        for link in self._link_elems[FilmLink] + self._link_elems[CondLink]:
            elem0, elem1 = link.elem0, link.elem1
            idxs0, idxs1 = self.link_idx_pairs(link)
            if isinstance(link, FilmLink):
                film = link.film
                areas = self.link_areas(link)
            elif isinstance(link, CondLink):
                film = link.cond
                areas = ones(len(idxs0))
            run_cond_idxs0.extend(idxs0)
            run_cond_idxs1.extend(idxs1)
            film_is_func = True if callable(film) else False
            if isinstance(elem0, PartSurf):
                part0 = elem0.part
                linked_part_surf_idxs[part0].append(elem0._idx())
                linked_part_cond_idxs0[part0].extend(idxs0)
                linked_part_cond_idxs1[part0].extend(idxs1)
                if film_is_func:
                    var_film_parts.add(part0)
            if isinstance(elem1, PartSurf):
                part1 = elem1.part
                linked_part_surf_idxs[part1].append(elem1._idx())
                linked_part_cond_idxs0[part1].extend(idxs1)
                linked_part_cond_idxs1[part1].extend(idxs0)
                if film_is_func:
                    var_film_parts.add(part1)
            if isinstance(elem0, LPPartSurf) and isinstance(elem1, LPPartSurf):
                lp_to_lp_link_elems.append((part0, part1))
                lp_to_lp_surf_idxs.append((elem0._idx(), elem1._idx()))
                lp_to_lp_link_idxs_idxs.append(
                    (
                        len(linked_part_surf_idxs[part0]) - 1,
                        len(linked_part_surf_idxs[part1]) - 1,
                    )
                )
                if (
                    film_is_func
                    or elem0.part.temp_dependent
                    or elem1.part.temp_dependent
                ):
                    self.run_lp_to_lp_link_elems.append(lp_to_lp_link_elems[-1])
                    self.run_lp_to_lp_surf_idxs.append(lp_to_lp_surf_idxs[-1])
                    self.run_lp_to_lp_link_idxs_idxs.append(lp_to_lp_link_idxs_idxs[-1])
            if film_is_func:
                matl = link.matl
                film_link_func_idxs[0][film].extend(idxs0)
                film_link_func_idxs[1][film].extend(idxs1)
                film_link_func_areas[film].extend(areas)
                # Store slice of sub links with the same fluid,
                # for sorting of indices with the same fluids in the next step
                l = len(film_link_func_areas[film])
                film_link_func_fluid_info[film].append((matl, slice(l - len(areas), l)))
            else:
                self.cond.aix[idxs0, idxs1] = self.cond.aix[idxs1, idxs0] = film * areas

        # Prepare efficient Part simulation: sort indices
        fe_link_surf_idxs = []
        self._fe_cond_idxs0 = []
        self._fe_cond_idxs1 = []
        for part in fe_parts:
            fe_link_surf_idxs.append(array(linked_part_surf_idxs[part]))
            self._fe_cond_idxs0.append(array(linked_part_cond_idxs0[part]))
            self._fe_cond_idxs1.append(array(linked_part_cond_idxs1[part]))
        lp_link_surf_idxs = []
        self._lp_cond_idxs0 = []
        self._lp_cond_idxs1 = []
        for part in lp_parts:
            lp_link_surf_idxs.append(array(linked_part_surf_idxs[part]))
            self._lp_cond_idxs0.append(array(linked_part_cond_idxs0[part]))
            self._lp_cond_idxs1.append(array(linked_part_cond_idxs1[part]))

        # Prepare efficient FilmLink and CondLink function calls during simulation
        # Combine indices with the same function and material
        film_link_func_matl_idxs = ({}, {})
        film_link_func_matl_areas = {}
        # Sort fluids
        for link_func in film_link_func_idxs[0]:
            idxs0 = film_link_func_matl_idxs[0][link_func] = defaultdict(list)
            idxs1 = film_link_func_matl_idxs[1][link_func] = defaultdict(list)
            areas = film_link_func_matl_areas[link_func] = defaultdict(list)
            for matl, slice_ in film_link_func_fluid_info[link_func]:
                idxs0[matl].extend(film_link_func_idxs[0][link_func][slice_])
                idxs1[matl].extend(film_link_func_idxs[1][link_func][slice_])
                areas[matl].extend(film_link_func_areas[link_func][slice_])
        self._film_link_func_data = {}
        for (link_func, idxs0), idxs1, areas in zip(
            film_link_func_matl_idxs[0].items(),
            film_link_func_matl_idxs[1].values(),
            film_link_func_matl_areas.values(),
        ):
            self._film_link_func_data[link_func] = {}
            for matl in idxs0:
                self._film_link_func_data[link_func][matl] = (
                    array(idxs0[matl], dtype=int64),
                    array(idxs1[matl], dtype=int64),
                    array(areas[matl], dtype=float64),
                )

        # HeatSource
        # Collect the heat indices for every function in a list
        heat_srce_func_idxs = defaultdict(list)
        heat_srce_func_count = defaultdict(list)
        for heat_srce in self._boundary_elems[HeatSource]:
            heat = heat_srce.heat
            idxs = self.elem_idx[heat_srce.net_elem]
            if callable(heat):
                heat_srce_func_idxs[heat].append(idxs)
                heat_srce_func_count[heat].append(len(idxs))
            else:
                self.heat_src[idxs] = heat / len(idxs)
        # Prepare for efficient function calls during simulation
        self._heat_srce_func_data = {}
        for heat_func in heat_srce_func_idxs:
            self._heat_srce_func_data[heat_func] = (
                array(heat_srce_func_idxs[heat_func], dtype=int64),
                array(heat_srce_func_count[heat_func], dtype=float64),
            )

        # FluxSource
        # Collect the flux indices for every function in a list
        flux_srce_func_idxs = defaultdict(list)
        flux_srce_func_area = defaultdict(list)
        for flux_srce in self._boundary_elems[FluxSource]:
            flux = flux_srce.flux
            idxs = self.elem_idx[flux_srce.surf]
            areas = flux_srce.surf._net_dat.areas
            if callable(flux):
                flux_srce_func_idxs[flux].extend(idxs)
                flux_srce_func_area[flux].extend(areas)
            else:
                self.heat_src[idxs] = flux * areas
        # Prepare for efficient function calls during simulation
        self._flux_srce_func_data = {}
        for flux_func in flux_srce_func_idxs:
            self._flux_srce_func_data[flux_func] = (
                array(flux_srce_func_idxs[flux_func], dtype=int64),
                array(flux_srce_func_area[flux_func], dtype=float64),
            )

        # BoundNode
        # Collect the bound temp indices for every function in a list
        self._bound_temp_func_idxs = defaultdict(list)
        bound_temp_indxs = []
        for bound_node in self._boundary_elems[BoundNode]:
            temp = bound_node.temp
            idxs = self.elem_idx[bound_node]
            if callable(temp):
                self._bound_temp_func_idxs[temp].extend(idxs)
            else:
                self.init_temp[idxs] = temp
            self.capy[idxs] = float('inf')
            bound_temp_indxs.extend(idxs)
        # prepare for efficient function calls and data access during simulation
        for temp_func, idxs in self._bound_temp_func_idxs.items():
            self._bound_temp_func_idxs[temp_func] = array(idxs, dtype=int64)
        # 'inf' capacities are not allowed to be recalculated during simulation
        self._all_temp_dependent_capy_idxs = array(
            list(set(self._all_temp_dependent_matl_idx_values) - set(bound_temp_indxs)),
            dtype=int64,
        )

        # StatNode
        # Nodes for stationary solution
        stationary_elems = set(self._node_elems[StatNode]).difference(
            all_temp_bound_elems
        )
        self.stationary_elem_idxs = hstack(
            [self.elem_idx[elem] for elem in stationary_elems]
            + [array([], dtype=int64)]
        )

        # LPIO
        for part_surf in self._node_elems[LPPartSurf]:
            self.capy[self.elem_idx[part_surf]] = 0.0  # float('inf')
        self._lp_ios = []
        for part, lp_sys, cond_idxs0, cond_idxs1, surf_idxs in zip(
            lp_parts,
            lp_syss,
            self._lp_cond_idxs0,
            self._lp_cond_idxs1,
            lp_link_surf_idxs,
        ):
            if cond_idxs0.size == 0:
                raise Exception(
                    f'Cannot create Network without connecting link to LPPart "{part.name}".'
                )
            lp_io = LPIO(
                lp_sys,
                film_surf_idxs=surf_idxs,
                films=self.cond.aix[cond_idxs1, cond_idxs0]
                / lp_sys.surf_areas[surf_idxs],
                surf_net_idxs=array(
                    [self.elem_idx[surf] for surf in part.surf]
                ).ravel(),
                link_elem_net_idxs=cond_idxs1,
                link_self_net_idxs=cond_idxs0,
            )
            self._lp_ios.append(lp_io)

        # Two linked LPIOs -> film is series connection of film and margin of linked part
        for link_elem, surf_idx, idx_idx in zip(
            lp_to_lp_link_elems,
            lp_to_lp_surf_idxs,
            lp_to_lp_link_idxs_idxs,
        ):
            lp_io0 = self._lp_ios[self._lp_parts.index(link_elem[0])]
            lp_io1 = self._lp_ios[self._lp_parts.index(link_elem[1])]

            film0 = (
                lp_io1.L_film_margs_sum[idx_idx[1]]
                / lp_io0.lp_sys.surf_areas[surf_idx[0]]
            )
            lp_io0.films[idx_idx[0]] = film0

            film1 = (
                lp_io0.L_film_margs_sum[idx_idx[0]]
                / lp_io1.lp_sys.surf_areas[surf_idx[1]]
            )
            lp_io1.films[idx_idx[1]] = film1

            lp_io0.update_state_matrix()
            lp_io1.update_state_matrix()

        self.clean_cond = DOK(
            shape=self.cond.shape,
            dtype=self.cond.dtype,
            fill_value=self.cond.fill_value,
        )
        self.clean_cond.data = self.cond.data.copy()  # Without leak of LPParts
        # LPPart margins leak into conductance matrix:
        # rewrite with series connection of body margin and film coefficient
        for lp_io, cond_idxs0, cond_idxs1 in zip(
            self._lp_ios,
            self._lp_cond_idxs0,
            self._lp_cond_idxs1,
        ):
            film_marg = lp_io.L_film_margs_sum
            self.cond.aix[cond_idxs1, cond_idxs0] = film_marg
            self.cond.aix[cond_idxs0, cond_idxs1] = film_marg

        # FEIO and MORIO
        for part_surf in self._node_elems[FEPartSurf]:
            self.capy[self.elem_idx[part_surf]] = 0.0  # float('inf')
        self._ffe_ios = []
        self._mor_ios = []
        self.fe_part_to_io = {}
        for part, fe_sys, cond_idxs0, cond_idxs1, surf_idxs in zip(
            fe_parts,
            fe_syss,
            self._fe_cond_idxs0,
            self._fe_cond_idxs1,
            fe_link_surf_idxs,
        ):
            if cond_idxs0.size == 0:
                raise Exception(
                    f'Cannot create Network without connecting link to FEPart "{part.name}".'
                )
            if part.mor_dof is None:
                ffe_io = FFEIO(
                    fe_sys,
                    films=self.cond.aix[cond_idxs0, cond_idxs1]
                    / fe_sys.surf_areas[surf_idxs],
                    film_surf_idxs=surf_idxs,
                    surf_net_idxs=array(
                        [self.elem_idx[surf] for surf in part.surf]
                    ).ravel(),
                    link_elem_net_idxs=cond_idxs1,
                    link_self_net_idxs=cond_idxs0,
                )
                self.fe_part_to_io[part] = ffe_io
                self._ffe_ios.append(ffe_io)
            else:
                mor_io = MORIO(
                    fe_sys,
                    mor_dof=part.mor_dof,
                    ref_temp=part.mor_ref_temp,
                    films=(
                        self.cond.aix[cond_idxs0, cond_idxs1]
                        / fe_sys.surf_areas[surf_idxs]
                    ),
                    film_surf_idxs=surf_idxs,
                    surf_net_idxs=array(
                        [self.elem_idx[surf] for surf in part.surf]
                    ).ravel(),
                    link_elem_net_idxs=cond_idxs1,
                    link_self_net_idxs=cond_idxs0,
                )
                self.fe_part_to_io[part] = mor_io
                self._mor_ios.append(mor_io)

        self.run_cond, self.clean_run_cond = self._make_sparse_run_cond(
            run_cond_idxs0, run_cond_idxs1
        )

        # Prepare for time changing LPParts
        self._var_film_lp_ios = []
        self._var_lp_run_cond_film_idxs = []
        self._var_body_lp_ios = []
        self._var_lp_run_cond_fwd_idxs = []
        self._var_lp_run_cond_bwd_idxs = []
        self._var_lp_ios = []
        for part, lp_io, lp_sys, cond_idxs0, cond_idxs1, surf_idxs in zip(
            lp_parts,
            self._lp_ios,
            lp_syss,
            self._lp_cond_idxs0,
            self._lp_cond_idxs1,
            lp_link_surf_idxs,
        ):
            if part in var_film_parts:
                self._var_film_lp_ios.append(lp_io)
                self._var_lp_run_cond_film_idxs.append(
                    to_csr_data_idxs(self.run_cond, dense_idx=(cond_idxs1, cond_idxs0))
                )
            if part.temp_dependent:
                self._var_body_lp_ios.append(lp_io)
            if part in var_film_parts or part.temp_dependent:
                self._var_lp_run_cond_fwd_idxs.append(
                    to_csr_data_idxs(self.run_cond, dense_idx=(cond_idxs0, cond_idxs1))
                )
                self._var_lp_run_cond_bwd_idxs.append(
                    to_csr_data_idxs(self.run_cond, dense_idx=(cond_idxs1, cond_idxs0))
                )
                self._var_lp_ios.append(lp_io)

        # Prepare for time changing FEParts
        self._var_film_ffe_io = []
        self._var_body_ffe_ios = []
        # Network indices of variable conductance to parts
        self._var_ffe_run_cond_idxs = []
        self._var_film_mor_ios = []
        # self._var_body_ffe_io = []
        # Network indices of variable conductance to parts
        self._var_mor_run_cond_idxs = []
        for part, fe_sys, cond_idxs0, cond_idxs1, surf_idxs in zip(
            fe_parts,
            fe_syss,
            self._fe_cond_idxs0,
            self._fe_cond_idxs1,
            fe_link_surf_idxs,
        ):
            if part.mor_dof is None:
                ffe_io = self.fe_part_to_io[part]
                if part in var_film_parts:
                    self._var_film_ffe_io.append(ffe_io)
                    self._var_ffe_run_cond_idxs.append(
                        to_csr_data_idxs(
                            self.run_cond, dense_idx=(cond_idxs0, cond_idxs1)
                        )
                    )
                if part.temp_dependent:
                    self._var_body_ffe_ios.append(ffe_io)
            else:
                mor_io = self.fe_part_to_io[part]
                if part in var_film_parts:
                    self._var_film_mor_ios.append(mor_io)
                    self._var_mor_run_cond_idxs.append(
                        to_csr_data_idxs(
                            self.run_cond, dense_idx=(cond_idxs0, cond_idxs1)
                        )
                    )
        self._var_ffe_ios = set(self._var_film_ffe_io + self._var_body_ffe_ios)

    def link_idx_pairs(self, link: Union[CondLink, FilmLink]):
        return self.elem_idx[link.elem0], self.elem_idx[link.elem1]

    def link_areas(self, link):
        """Get link areas of the connection surface.

        Returns mean areas if the areas of connected surfaces have
        different sizes.
        """
        if isinstance(link.elem0, (FEPartSurf, LPPartSurf)):
            areas = link.elem0._net_dat.areas
            if isinstance(link.elem1, (FEPartSurf, LPPartSurf)):
                areas = (areas + link.elem1._net_dat.areas) / 2.0  # Mean areas
        elif isinstance(link.elem1, (FEPartSurf, LPPartSurf)):
            areas = link.elem1._net_dat.areas
        else:
            raise ValueError("Non of the elements has a surface!")
        return areas

    def _make_sparse_run_cond(self, run_cond_idxs0, run_cond_idxs1):
        """Make a sparse conductance matrix for use in time iteration."""
        # Use indices to create the matrix, because some conductances may be zero at this time
        diag_idxs = list(range(self._node_count))
        run_cond_idxs = (
            array(run_cond_idxs0 + run_cond_idxs1 + diag_idxs, dtype=int64).ravel(),
            array(run_cond_idxs1 + run_cond_idxs0 + diag_idxs, dtype=int64).ravel(),
        )

        run_cond_idxs = unique(
            run_cond_idxs, axis=1
        )  # remove double entries because they get summed up during csr_matrix creation
        run_cond_idxs = (run_cond_idxs[0], run_cond_idxs[1])
        run_cond = sparse.csr_matrix((self.cond.aix[run_cond_idxs], run_cond_idxs))
        clean_run_cond = sparse.csr_matrix(
            (self.clean_cond.aix[run_cond_idxs], run_cond_idxs)
        )

        # Generate indices for sparse conductance matrix access for time dependent behaviour during simulation time
        # Flow link temperature dependent volumetric capacity
        _, idxs0, idxs1, vol_flow = self._flow_link_data
        sparse_idxs = to_csr_data_idxs(run_cond, (idxs0, idxs1))
        self._flow_link_data = sparse_idxs, idxs0, idxs1, vol_flow
        # Volume flow functions
        for flow_func, flow_data in self._flow_link_func_data.items():
            for fluid, (idxs0, idxs1, vol_flow) in flow_data.items():
                data_idxs = to_csr_data_idxs(run_cond, (idxs0, idxs1))
                flow_data[fluid] = (data_idxs, idxs0, idxs1, vol_flow)
        # Film coefficient functions
        for film_func, fluid_data in self._film_link_func_data.items():
            for fluid, (idxs0, idxs1, areas) in fluid_data.items():
                fluid_data[fluid] = (
                    to_csr_data_idxs(run_cond, (idxs0, idxs1)),
                    to_csr_data_idxs(run_cond, (idxs1, idxs0)),
                    idxs0,
                    idxs1,
                    areas,
                )

        return run_cond, clean_run_cond

    def _set_time_dependent_properties(
        self, time, net_temp, lp_solve_temp, fe_solve_temp, mor_solve_temp
    ):
        """sets time changing properties of the node net during simulation time"""
        cond_data = self.run_cond.data
        clean_cond_data = self.clean_run_cond.data
        condy = self.condy
        vol_capy = self._vol_capy

        # Needs to be called before the time dependent functions
        for input_set_method in self.input_set_methods:
            input_set_method(time)

        # Set time dependent material properties
        for matl, idx in self._all_temp_dependent_matl_idxs.items():
            matl_temp = net_temp[idx]
            matl.check_limits(matl_temp)
            vol_capy[idx] = matl.vol_capy_interp(matl_temp)
            condy[idx] = matl.condy_interp(matl_temp)
        # Capacity of solid elements and MatlNodes
        capy_idxs = self._all_temp_dependent_capy_idxs
        self.capy[capy_idxs] = self._vol[capy_idxs] * vol_capy[capy_idxs]
        # Flow link temperature dependent volumetric capacity
        data_idxs, idxs0, idxs1, vol_flow = self._flow_link_data
        cond_data[data_idxs] = vol_flow * (vol_capy[idxs0] + vol_capy[idxs1]) / 2
        # Volume flow functions
        for flow_func, flow_data in self._flow_link_func_data.items():
            for fluid, (data_idxs, idxs0, idxs1, vol_flow) in flow_data.items():
                cond_data[data_idxs] = (
                    flow_func(net_temp[idxs0], net_temp[idxs1], fluid)
                    * (vol_capy[idxs0] + vol_capy[idxs1])
                    / 2
                )
        # Film coefficient functions
        for film_func, fluid_data in self._film_link_func_data.items():
            for matl, (fwd_didxs, bwd_didxs, idxs0, idxs1, areas) in fluid_data.items():
                cond = film_func(net_temp[idxs0], net_temp[idxs1], matl) * areas
                clean_cond_data[fwd_didxs] = clean_cond_data[bwd_didxs] = cond
                cond_data[fwd_didxs] = cond_data[bwd_didxs] = cond

        # Heat source functions
        for hsrc_func, (idxs_arr, node_count_arr) in self._heat_srce_func_data.items():
            for idxs, node_count in zip(idxs_arr, node_count_arr):
                self.heat_src[idxs] = hsrc_func(mean(net_temp[idxs])) / node_count
        # Flux source functions
        for flux_func, (idxs, areas) in self._flux_srce_func_data.items():
            self.heat_src[idxs] = flux_func(net_temp[idxs]) * areas
        # Boundary temperature functions
        for temp_func, idxs in self._bound_temp_func_idxs.items():
            net_temp[idxs] = temp_func()

        # # Part IOs
        # For now all film coefficients of time variable parts are computed.
        # This could be sped up if only changing film coefficients are considered

        # Update changing films of parts; get film from updated network conductance
        for lp_io, run_cond_idxs in zip(
            self._var_film_lp_ios, self._var_lp_run_cond_film_idxs
        ):
            dock_areas = lp_io.lp_sys.surf_areas[lp_io.film_surf_idxs]
            lp_io.films = clean_cond_data[run_cond_idxs] / dock_areas
        for ffe_io, run_cond_idxs in zip(
            self._var_film_ffe_io, self._var_ffe_run_cond_idxs
        ):
            dock_areas = ffe_io.fe_sys.surf_areas[ffe_io.film_surf_idxs]
            ffe_io.films = clean_cond_data[run_cond_idxs] / dock_areas
        for mor_io, run_cond_idxs in zip(
            self._var_film_mor_ios, self._var_mor_run_cond_idxs
        ):
            dock_areas = mor_io.fe_sys.surf_areas[mor_io.film_surf_idxs]
            mor_io.films = clean_cond_data[run_cond_idxs] / dock_areas

        # Update changing body material properties of parts
        for lp_io in self._var_body_lp_ios:
            body_temps = lp_solve_temp[self._lp_ios.index(lp_io)]
            lp_io.update_material_properties(body_temps)
        for ffe_io in self._var_body_ffe_ios:
            body_temps = fe_solve_temp[self._ffe_ios.index(ffe_io)]
            ffe_io.update_material_properties(body_temps)

        # Update IO-system of parts in case of changing film coefficient and body material
        for lp_io in self._var_lp_ios:
            lp_io.update_state_matrix()

        # Two linked LPIOs -> film is series connection of film and margin of the linked part
        for link_elem, surf_idx, idx_idx in zip(
            self.run_lp_to_lp_link_elems,
            self.run_lp_to_lp_surf_idxs,
            self.run_lp_to_lp_link_idxs_idxs,
        ):
            lp_io0 = self._lp_ios[self._lp_parts.index(link_elem[0])]
            lp_io1 = self._lp_ios[self._lp_parts.index(link_elem[1])]

            film0 = (
                lp_io1.L_film_margs_sum[idx_idx[1]]
                / lp_io0.lp_sys.surf_areas[surf_idx[0]]
            )
            lp_io0.films[idx_idx[0]] = film0
            film1 = (
                lp_io0.L_film_margs_sum[idx_idx[0]]
                / lp_io1.lp_sys.surf_areas[surf_idx[1]]
            )
            lp_io1.films[idx_idx[1]] = film1

            lp_io0.update_state_matrix()
            lp_io1.update_state_matrix()

        for lp_io, fwd_idxs, bwd_idxs in zip(
            self._var_lp_ios,
            self._var_lp_run_cond_fwd_idxs,
            self._var_lp_run_cond_bwd_idxs,
        ):
            # Margin conductances from LP parts leak into the network system to get the
            # proper conductance from the connected network node to margin nodes of the
            # LP parts.
            cond_data[fwd_idxs] = cond_data[bwd_idxs] = lp_io.L_film_margs_sum

        for ffe_io in self._var_ffe_ios:
            ffe_io.update_state_matrix()

        for mor_io in self._var_film_mor_ios:
            mor_io.update_state_matrix()

    def _count_and_assign_elem_nodes(
        self, temp_unbound_node_elems, temp_bound_node_elems
    ):
        """Counts the number of nodes needed for the node net and
        assigns the nodes of each element to its place in the parameter
        arrays of the node net."""
        elem_node_idxs = {}
        # make index arrays for every node element; at first locale indices
        for node_elem_lists in self._node_elems.values():
            for node_elem in node_elem_lists:
                elem_node_idxs[node_elem] = arange(1)

        def process_node_elems(node_elems, elem_slice, net_node_count):
            for node_elem in node_elems:
                elem_node_count = 1
                elem_node_idxs[node_elem] += net_node_count
                elem_slice[node_elem] = slice(
                    net_node_count, net_node_count + elem_node_count
                )  # For result access
                net_node_count += elem_node_count
            return net_node_count

        temp_unbound_elems = [
            item for sublist in temp_unbound_node_elems.values() for item in sublist
        ]
        temp_bound_elems = [
            item for sublist in temp_bound_node_elems.values() for item in sublist
        ]
        net_node_count = 0
        elem_slice = {}
        net_node_count = process_node_elems(temp_unbound_elems, elem_slice, net_node_count)
        free_temp_lgth = net_node_count
        net_node_count = process_node_elems(temp_bound_elems, elem_slice, net_node_count)
        # Indices of the margin body nodes connected to the specified surface
        body_marg_idxs = {}
        return elem_node_idxs, net_node_count, elem_slice, free_temp_lgth, body_marg_idxs

    def __len__(self):
        """Number of nodes"""
        return self._node_count

    def _show_progress_bar(self, result_times):
        def show_progress():
            self._sim_timer = threading.Timer(2.0, show_progress)  # Restart timer
            self._sim_timer.daemon = (
                True  # Kill timer thread at shut down of main process
            )
            self._sim_timer.start()
            # if hasattr(self.nodes, 'time'):
            self._progress_bar.speed = self._solve_step_speed
            self._progress_bar.value = self._solve_step_time
            if self._progress_bar.value >= self._progress_bar.max:
                self._progress_bar.finish()
                self._sim_timer.cancel()

        self._solve_step_speed = 0.0
        self._solve_step_time = 0.0
        self._progress_bar = TextProgressBar(
            min=self.time,
            max=result_times[-1],
            description="Simulation run progress, speed as sim. time vs. real time:",
        )
        show_progress()

    def sim(
        self,
        time_span: Union[Tuple[float, float], List[float]],
        rel_tol: float = 1e-3,
        abs_tol: float = 1e-6,
        method='RK45',
        progress_bar: bool = True,
        num_res_skip: int = 0,
        **options,
    ) -> Result:
        """Runs the model simulation over time.

        Args:
            time_span: Points in time for start and end of simulation.
                If start >= end only the initial state of the network
                is saved in the results.
            rel_tol: The solver keeps the local error estimates less
                than abs_threshold + rel_threshold * abs(temp_result).
            abs_tol: The solver keeps the local error estimates less
                than abs_tol + rel_tol * abs(temp_result).
            method: Integration method, use ‘RK45’ (default) for
                the explicit Runge-Kutta method of order 5(4),
                use ‘RK23’ for the explicit Runge-Kutta method of
                order 3(2), use 'LSODA' for Fortran solver from ODEPACK
                with automatic stiffness detection and switching
            progress_bar: Shows a progress_bar to indicate the
                solution progress.
            num_res_skip: Number of results to skip over before collecting
                next result.
            options: Additional solver options, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        Returns:
            Simulation results over time
        """
        if len(time_span) != 2:
            raise ValueError("time_span must be a sequence of length 2")
        time_span = float(time_span[0]), float(time_span[1])
        self.time = time_span[0]
        if time_span[1] > time_span[0]:
            if progress_bar:
                self._show_progress_bar(time_span)
        else:
            progress_bar = False
        try:
            result_collector = solver.transient_solution(
                time_span, self, rel_tol, abs_tol, method, num_res_skip, **options
            )
        finally:
            if progress_bar:
                if not self._progress_bar.finished:
                    self._sim_timer.cancel()
                    self._progress_bar.finish()
        return Result(ResultData.from_collector(self, result_collector))

    @wraps(plot.params)
    def plot(self, *args, **kwargs):
        time = kwargs.get('time')
        if time is not None:
            warnings.warn("Plotting initial state of network with given time only!")
            sim_time = time
        else:
            sim_time = 0.0
        result = self.sim(time_span=(sim_time, sim_time))
        return plot.params(result._data, sim_time, *args, **kwargs)
