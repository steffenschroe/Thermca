from __future__ import annotations
import warnings
from functools import wraps
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from numpy import (
    array,
    atleast_2d,
    empty,
    inf,
    divide,
    float64,
    ndarray,
    asarray,
    hstack,
    dtype,
)
from sparse import DOK, COO
from scipy import spatial
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix

try:
    import pandas as pd

    pandas_installed = True
    from pandas import DataFrame
except ImportError:
    pandas_installed = False

import thermca.plot.result as plot
from thermca.links import BaseLink

if TYPE_CHECKING:
    from thermca.network import Network

from thermca.pointnodes import StatNode, MatlNode, BoundNode, Node
from thermca.fem.fe_part import FEPart, FEPartSurf
from thermca.lpm.lp_part import LPPart, LPPartSurf

PointNode = Node | MatlNode | StatNode | BoundNode
PartSurf = FEPartSurf | LPPartSurf
NetElement = PointNode | PartSurf  # Linkable network nodes
Part = FEPart | LPPart


@dataclass
class _ResultCollector:
    """Temporary container for collecting results during simulation.

    Values are stored as time series collections
    """

    io_temp_dt: dtype
    times: list[float] = field(default_factory=list)
    net_temps: list[ndarray] = field(default_factory=list)
    net_capys: list[ndarray] = field(default_factory=list)
    net_heat_srcs: list[ndarray] = field(default_factory=list)
    net_posns: list[ndarray] = field(default_factory=list)
    net_conds: list[ndarray] = field(default_factory=list)
    net_clean_conds: list[ndarray] = field(default_factory=list)
    io_temps: list[ndarray] = field(default_factory=list)
    lp_M_diagss: list[ndarray] = field(default_factory=list)
    lp_Lss: list[csr_matrix] = field(default_factory=list)


class Result:
    """Contains the result data and provides convenient access

    Results are selected regarding model elements or regarding the
    links between the elements:

    * Regarding model elements, temperatures capacities, and heat
      inputs are available, see: :class:`ElementProcessing`
    * Regarding links, conductances, film coefficients, heat flows
      and heat fluxes are available, see: :class:`LinkProcessing`

    Data is accessed by specifying the associated model elements or
    links as keys. This is followed by a method for the desired
    physical quantity.

    Example:

        >>> from numpy import geomspace
        >>> import matplotlib.pyplot as plt
        >>> from thermca import *
        >>>
        >>> steel = Solid(dens=8000., spec_heat=500., condy=50.)
        >>>
        >>> # Create a sheet geometry with two coupling surfaces
        >>> with Asm() as cuboid:
        ...     block = Cube(width=.5, hgt=1., depth=.05, hgt_div=10)
        ...     face = block.face
        ...     Surf(
        ...         name='upper',
        ...         faces=[face.left, face.right, face.front, face.back, face.top],
        ...     )
        ...     Surf(name='btm', faces=[face.btm])
        ...
        >>> # Create a steel sheet with a link from the upper surface to the
        >>> # environment via a film BC and a forced heat flow into the bottom
        >>> # surface.
        >>> with Model() as model:
        ...     sheet = LPPart(asm=cuboid, matl=steel, name='sheet')
        ...     env = BoundNode(temp=20., posn=(-.3, .5, .025), name='env')
        ...     upper_to_env = FilmLink(sheet.surf.upper, env, film=10.)
        ...     HeatSource(sheet.surf.btm, heat=1000., name='src')
        ...
        >>> # Simulate over 36000 seconds
        >>> sim_time = 36000.
        >>> result = Network(model).sim((0., sim_time))
        Simulation run progress, speed as sim. time vs. real time:
         0% 100% done.
        >>> # Get the temperatures on the bottom and top corner and plot it.
        >>> temps = result[sheet].temp(lctn=[(0., 0., 0.), (0., 1., 0.)])
        >>> plt.plot(result.time(), temps)
        [<matplotlib.lines.Line2D object at 0x1337b6050>, <matplotlib.lines.Line2D object at 0x1337b6ec0>]
        >>> plt.show()
        >>> # Get a data frame of the temperatures at certain points in time.
        >>> frame = result[sheet].temp_frame(
        ...     time=geomspace(.1, sim_time, num=5),
        ...     lctn=[(0., 0., 0.), (0., 1., 0.)])
        >>> print(frame)
        Temperature   sheet lctn=(0., 0., 0.)  sheet lctn=(0., 1., 0.)
        Time
        0.100000                     0.010217                 0.000268
        2.449490                     0.249103                 0.006545
        60.000000                    5.769231                 0.157884
        1469.693846                 83.300245                 3.310066
        36000.000000               253.019457                45.451702
    """

    def __init__(self, result_data):
        self._data = result_data

    def __getitem__(self, elem_or_link):
        """Returns interpolation objects for convenient result access.

        Args:
            elem_or_link:
                Data is accessed by specifying the associated model
                elements or links as keys.
        """

        if isinstance(elem_or_link, (NetElement, Part)):
            return ElementProcessing(self._data, elem_or_link)
        elif isinstance(elem_or_link, BaseLink):
            return LinkProcessing(self._data, elem_or_link)
        else:
            raise TypeError('Allowed types are elements or links!')

    def time(self):
        """Time stamps for result values"""
        return self._data.times

    @wraps(plot.params)
    def plot(self, *args, **kwargs):
        return plot.params(self._data, *args, **kwargs)


class ElementProcessing:
    def __init__(self, result_data, elem):
        self.data = result_data
        self.elem = elem
        if isinstance(self.elem, PartSurf):
            self.elem_name = self.elem.part.name + '.' + self.elem.name
        else:
            self.elem_name = self.elem.name

    def temp(
        self,
        time: ndarray | list[float] = None,
        lctn: ndarray | list[tuple[float, float, float]] = None,
    ) -> ndarray:
        """Returns the simulation temperatures of the element

        Args:
            time: Time values for which the temperatures are returned.
                The values need ascending order. For given time
                points between simulation time steps, the resulting
                temperatures are linearly interpolated.
                If _no_ times are given, the solver times are used.
            lctn: Locations within geometric defined elements to
                get the temperature for. The temperatures of the
                nearest nodes to the given locations are used. The
                locations are locale coordinates. Multiple points may
                be given: e.g. [(0., 1., .0), (2., 4., 0.)].
                The default value is None. In this case the caloric
                mean temperature of the element is returned.

        Returns:
            Temperatures on given times and locale coordinates;
            If _no_ locale coordinates are given a one dimensional
            array will be returned with one value for each given time.
            Otherwise, the array is two-dimensional. The first dimension
            represents the given times, the second dimension the
            given body locations (time in row, location in column).
        """
        if lctn is not None:
            lctn = asarray(lctn)
            if lctn.shape[1] != 3:
                raise Exception(
                    "'lctns' argument must be a sequence of vectors of length 3!"
                )
        if time is not None:
            time = asarray(time)

        elem = self.elem
        net = self.data.network

        if isinstance(elem, NetElement):
            if lctn is not None:
                warnings.warn(
                    "'lctn' argument gets ignored because nodes have lumped properties!"
                )
            lctn_temps = self.data.net_temps[:, net.elem_idx[elem]].ravel()
        elif isinstance(elem, LPPart):
            part_idx = net._lp_parts.index(elem)
            view = self.data.io_temps.ravel().view(dtype=self.data.io_temp_dt)
            temps = view[:]['lp'][str(part_idx)]
            lp_sys = net._lp_ios[part_idx].lp_sys
            node_lctns = lp_sys.posns[: lp_sys.dof]
            if lctn is None:
                lctn_temps = lp_sys.C_body @ temps.T
            else:
                near_dists, nearest_point_idxs = spatial.KDTree(node_lctns).query(lctn)
                lctn_temps = temps[:, nearest_point_idxs]
        elif isinstance(elem, FEPart):
            view = self.data.io_temps.ravel().view(dtype=self.data.io_temp_dt)
            if elem.mor_dof is None:
                ffe_io = net.fe_part_to_io[elem]
                fe_sys = ffe_io.fe_sys
                temps = view['ffe'][str(net._ffe_ios.index(ffe_io))]
            else:
                mor_io = net.fe_part_to_io[elem]
                fe_sys = mor_io.fe_sys
                temps = view[:]['mor'][str(net._mor_ios.index(mor_io))]
                s = temps.shape
                temps = temps.reshape(s[0], s[1] * s[2])
                temps = hstack(mor_io.VT1) @ temps.T + mor_io.ref_temp
                temps = temps.T
            if lctn is None:
                lctn_temps = fe_sys.C_body @ temps.T
            else:
                temps = temps[:, fe_sys.vert_to_dof_idx]
                near_dists, nearest_point_idxs = spatial.KDTree(
                    elem._body_mesh.points
                ).query(lctn)
                lctn_temps = temps[:, nearest_point_idxs]

        if time is None:
            result = lctn_temps
        else:
            result = interp1d_time(self.data.times, time, lctn_temps)
        return result

    def temp_frame(
        self,
        time: ndarray | list[float] = None,
        lctn: ndarray | list[tuple[float, float, float]] = None,
    ) -> DataFrame:
        """Returns a data frame of the simulation temperatures of the
        element

        Args:
            time: Time values for which the temperatures are returned.
                The values need ascending order. For given time
                points between simulation time steps, the resulting
                temperatures are linearly interpolated.
                If _no_ times are given, the solver times are used.
            lctn: Locations within geometric defined elements to
                get the temperature for. The temperatures of the
                nearest nodes to the given locations are used. The
                locations are locale coordinates. Multiple points may
                be given: e.g. [(0., 1., .0), (2., 4., 0.)].
                The default value is None. In this case the caloric
                mean temperature of the element is returned.

        Returns:
            Dataframe with times and temperatures
        """
        return _create_data_frame(
            self.temp, time, lctn, 'Temperature', self.elem_name, self.data
        )

    def capy(self, time: ndarray | list[float] = None) -> ndarray:
        """Returns the simulation capacities of the element

        Args:
            time: Time values for which the capacities are returned.
                The values need ascending order. For given time
                points between simulation time steps, the resulting
                capacities are linearly interpolated.
                If _no_ times are given, the solver times are used.
        """
        if time is not None:
            time = asarray(time)

        elem = self.elem
        net = self.data.network

        if isinstance(elem, PointNode):
            capy = self.data.net_capys[:, net.elem_idx[elem]].sum(axis=1)
        elif isinstance(elem, LPPart):
            part_idx = net._lp_parts.index(elem)
            capy = array([diag[part_idx].sum() for diag in self.data.lp_M_diagss])
        elif isinstance(elem, FEPart):
            raise NotImplementedError()
        else:
            raise TypeError("Capacity is available for point nodes and parts!")

        if time is None:
            result = capy
        else:
            result = interp1d_time(self.data.times, time, capy)
        return result

    def capy_frame(self, time: ndarray | list[float] = None) -> DataFrame:
        """Returns a data frame of simulation capacities of the element

        Args:
            time: Time values for which the capacities are returned.
                The values need ascending order. For given time
                points between simulation time steps, the resulting
                capacities are linearly interpolated.
                If _no_ times are given, the solver times are used.
        Returns:
            Dataframe with times and capacities
        """
        return _create_data_frame(
            self.capy, time, None, 'Capacity', self.elem_name, self.data
        )

    def heat_src(self, time: ndarray | list[float] = None) -> ndarray:
        """Returns the heat flow from heat sources into the element

        Args:
            time: Time values for which the heat flow values are
                returned. The values need ascending order. For given
                time points between simulation time steps, the
                heat flow values are linearly interpolated.
                If _no_ times are given, the solver times are used.
        """
        if time is not None:
            time = asarray(time)

        elem = self.elem
        net = self.data.network

        if not isinstance(elem, NetElement):
            raise TypeError("Heat can only be fed into point nodes and surfaces!")

        heat_flow = self.data.net_heat_srcs[:, net.elem_idx[elem]].ravel()
        if time is None:
            result = heat_flow
        else:
            result = interp1d_time(self.data.times, time, heat_flow)
        return result

    def heat_src_frame(self, time: ndarray | list[float] = None) -> DataFrame:
        """Returns a data frame of heat flow from heat sources into the
        element

        Args:
            time: Time values for which the heat flow values are
                returned. The values need ascending order. For given
                time points between simulation time steps, the
                resulting heat flow values are linearly interpolated.
                If _no_ times are given, the solver times are used.
        Returns:
            Dataframe with times and heat flow from heat sources
        """
        return _create_data_frame(
            self.heat_src, time, None, 'Heat from source', self.elem_name, self.data
        )


class LinkProcessing:
    def __init__(self, result_data, link):
        self.data = result_data
        self.link = link
        self.idxs0, self.idxs1 = self._get_node_link_idxs(link.elem0, link.elem1)
        if link.name == '':
            if isinstance(link.elem0, PartSurf):
                elem0_name = link.elem0.part.name + '.' + link.elem0.name
            else:
                elem0_name = link.elem0.name
            if isinstance(link.elem1, PartSurf):
                elem1_name = link.elem1.part.name + '.' + link.elem1.name
            else:
                elem1_name = link.elem1.name
            self.link_name = elem0_name + '-' + elem1_name
        else:
            self.link_name = link.name

    def cond(self, time: ndarray | list[float] = None) -> ndarray:
        """Returns the simulation conductance values of the link

        Args:
            time: Time values for which the conductances are
                returned. The values need ascending order. For given
                time points between simulation time steps, the
                resulting conductance values are linearly interpolated.
                If _no_ times are given, the solver times are used.

        Returns:
            Conductances on given times
        """

        if isinstance(self.link.elem0, LPPartSurf) or isinstance(
            self.link.elem1, LPPartSurf
        ):
            cond = self.data.net_clean_conds[:, self.idxs1, self.idxs0].T.sum(axis=0)
        else:
            cond = (
                self.data.net_conds.vix[:, self.idxs1, self.idxs0]
                .todense()
                .T.sum(axis=1)
            )  # could be summed as coo to save memory

        if time is None:
            result = cond
        else:
            result = interp1d_time(self.data.times, time, cond)
        return result

    def cond_frame(self, time: ndarray | list[float] = None) -> DataFrame:
        """Returns a data frame of the conductance values of the links

        Args:
            time: Time values for which the conductances are
                returned. The values need ascending order. For given
                time points between simulation time steps, the
                resulting conductance values are linearly interpolated.
                If _no_ times are given, the solver times are used.

        Returns:
            Data frame of times and conductances
        """
        return _create_data_frame(
            self.cond, time, None, 'Conductance', self.link_name, self.data
        )

    def film(self, time: ndarray | list[float] = None) -> ndarray:
        """Returns the simulation film coefficient values of the link

        Note:
            If unequal surfaces are connected, the film coefficients
            refer to the mean area.

        Args:
            time: Time values for which the film coefficients are
                returned. The values need ascending order. For given
                time points between simulation time steps, the
                resulting film coefficient values are linearly
                interpolated.
                If _no_ times are given, the solver times are used.

        Returns:
            Film coefficients on given times
        """
        area = self.data.network.link_areas(self.link)
        return self.cond(time) / area

    def film_frame(self, time: ndarray | list[float] = None) -> DataFrame:
        """Returns a data frame of the film coefficient values of the
        links

        Note:
            If unequal surfaces are connected, the film coefficients
            refer to the mean area.

        Args:
            time: Time values for which the film coefficients are
                returned. The values need ascending order. For given
                time points between simulation time steps, the
                resulting film coefficient values are linearly
                interpolated.
                If _no_ times are given, the solver times are used.

        Returns:
            Data frame of times and film coefficients
        """
        return _create_data_frame(
            self.film, time, None, 'Film coefficient', self.link_name, self.data
        )

    def heat(self, time: ndarray | list[float] = None) -> ndarray:
        """Returns the simulation heat flow values across the
        link

        Args:
            time: Time values for which the heat flow values are
                returned. The values need ascending order. For given
                time points between simulation time steps, the
                resulting heat flow values are linearly interpolated.
                If _no_ times are given, the solver times are used.

        Returns:
            Heat flow on given times
        """
        temps0, temps1 = (
            self.data.net_temps[:, self.idxs0],
            self.data.net_temps[:, self.idxs1],
        )
        conds = self.data.net_conds.vix[:, self.idxs1, self.idxs0].todense().T
        heat = ((temps0 - temps1) * conds).sum(axis=1)
        if time is None:
            result = heat
        else:
            result = interp1d_time(self.data.times, time, heat)
        return result

    def heat_frame(self, time: ndarray | list[float] = None) -> DataFrame:
        """Returns a data frame of the heat flow values across the
        links

        Args:
            time: Time values for which the heat flow values are
                returned. The values need ascending order. For given
                time points between simulation time steps, the
                resulting heat flow values are linearly interpolated.
                If _no_ times are given, the solver times are used.

        Returns:
            Data frame of times and heat flow values
        """
        return _create_data_frame(
            self.heat, time, None, 'Heat flow', self.link_name, self.data
        )

    def flux(self, time: ndarray | list[float] = None) -> ndarray:
        """Returns the simulation heat flux densities on the part surface

        Note:
            If unequal surfaces are connected, the heat flux
            densities refer to the mean area.

        Args:
            time: Time values for which the heat flux densities are
                returned. The values need ascending order. For given
                time points between simulation time steps, the
                resulting heat flux densities are linearly
                interpolated.
                If _no_ times are given, the solver times are used.

        Returns:
            Heat flux densities on given times
        """
        area = self.data.network.link_areas(self.link)
        return self.heat(time) / area

    def flux_frame(self, time: ndarray | list[float] = None) -> DataFrame:
        """Returns a data frame of the heat flux densities on the part
        surface

        Note:
            If unequal surfaces are connected, the heat flux
            densities refer to the mean area.

        Args:
            time: Time values for which the heat flux densities are
                returned. The values need ascending order. For given
                time points between simulation time steps, the
                resulting heat flux densities are linearly
                interpolated.
                If _no_ times are given, the solver times are used.

        Returns:
            Data frame of times and heat flux densities
        """
        return _create_data_frame(
            self.flux, time, None, 'Heat flux density', self.link_name, self.data
        )

    def _get_node_link_idxs(self, elem0, elem1):
        return (
            self.data.network.elem_idx[elem0],
            self.data.network.elem_idx[elem1],
        )


def interp1d_time(grid_time, time, grid_values):
    interp_func = interp1d(grid_time, grid_values, axis=0, bounds_error=True)
    try:
        result = interp_func(time)
    except ValueError:
        interp_func = interp1d(
            grid_time,
            grid_values,
            axis=0,
            bounds_error=False,
            fill_value='extrapolate',
        )
        result = interp_func(time)
        warnings.warn(
            "Some temperatures are extrapolated because the given time is outside"
            " the simulation time span! Because of finite float precision, this "
            "may occur during interpolation at the bounds of the time span."
        )
    return result


def _create_data_frame(res_func, time, lctn, phys_quantity_str, name, data):
    if not pandas_installed:
        raise Exception('Install pandas to get data frames!')

    if time is None:
        time = data.times
    series = []
    if lctn is not None:  # temp is 2d array shape(time, lctn)
        res = res_func(time, lctn)
        for temp, lctn in zip(res.T, atleast_2d(lctn)):
            s = pd.Series(temp, index=time)
            s.index.name = 'Time'
            col_name_lctn = name
            col_name_lctn += ' lctn=(' + float_to_compact_str(lctn[0])
            col_name_lctn += ', ' + float_to_compact_str(lctn[1])
            col_name_lctn += ', ' + float_to_compact_str(lctn[2]) + ')'
            s = s.rename(col_name_lctn)
            series.append(s)
        frame = pd.concat(series, axis=1)
    else:  # temp is 1d array
        res = res_func(time)
        s = pd.Series(res, index=time)
        s.index.name = 'Time'
        s = s.rename(name)
        frame = s.to_frame()
    frame.columns.name = phys_quantity_str
    return frame


class ResultData:
    """Data container for the results of a simulation run

    The purpose of the container is internal data storage.
    Prepared for access across result time steps.
    Convenient access to result data is given by the :class:`Result`
    class.

    Args:
        network: The network the results belong to
        times: Time stamps for each result time step
        net_temps: Network node temperatures
        net_capys: Network capacities
        net_heat_srcs: Network heat sources
        net_posns: Network node positions
        net_time_consts: Network node time constants
        net_conds: Network conductance matrices
        net_clean_conds: Network conductance matrices without leaking
            conductance from LPParts
        net_heats:Network heat flow matrices
        io_temps: Part temperatures
        io_temp_dt: Recarray datatype for 'io_temps' access
        lp_M_diagss: LPPart capacities
        lp_Lss: LPPart conductance matrices
    """

    def __init__(
        self,
        network: Network,
        times: ndarray,
        net_temps: ndarray,
        net_capys: ndarray,
        net_heat_srcs: ndarray,
        net_posns: ndarray,
        net_time_consts: ndarray,
        net_conds: DOK,
        net_clean_conds: ndarray,
        net_heats: DOK,
        io_temps: ndarray,
        io_temp_dt,
        lp_M_diagss: list,
        lp_Lss: list[csr_matrix],
    ):
        self.network = network
        self.net_posns = asarray(net_posns)  # shape=(time_slot_count, node_count, 3)
        self.times = asarray(times)  # shape=(time_slot_count,)
        self.net_temps = asarray(net_temps)  # shape=(time_slot_count, node_count)
        self.io_temps = asarray(io_temps)
        self.net_capys = asarray(net_capys)  # shape=(time_slot_count, node_count)
        self.net_heat_srcs = asarray(net_heat_srcs)  # shape=(time_slot_count, node_count))
        self.net_time_consts = asarray(net_time_consts)
        self.net_conds = net_conds  # sparse.DOK
        self.net_clean_conds = asarray(net_clean_conds)
        self.net_heats = net_heats  # sparse.DOK
        self.time_slot_count = len(times)
        self.num_net_nodes = len(net_temps[0])
        self.io_temp_dt = io_temp_dt
        self.lp_M_diagss = lp_M_diagss
        self.lp_Lss = lp_Lss

    @classmethod
    def from_collector(cls, network, collector):
        time_slot_count = len(collector.times)
        node_count = len(collector.net_temps[0])
        # TODO: think about to evaluate heats (and time_consts) lazy because of memory usage
        temps = array(collector.net_temps)
        capys = array(collector.net_capys)
        heats, time_consts = cls._calc_heat_and_time_const(
            collector.net_conds, temps, capys
        )
        shape = (time_slot_count, node_count, node_count)
        conds = cls._create_sparse_arrays_over_all_times(collector.net_conds, shape)
        heats = cls._create_sparse_arrays_over_all_times(heats, shape)
        result_data = ResultData(
            network=network,
            times=array(collector.times),  # shape=(time_slot_count,)
            net_temps=temps,  # shape=(time_slot_count, node_count),
            net_capys=capys,  # shape=(time_slot_count, node_count),
            net_heat_srcs=array(
                collector.net_heat_srcs
            ),  # shape=(time_slot_count, node_count)),
            net_posns=array(collector.net_posns),  # shape=(time_slot_count, node_count, 3),
            net_time_consts=time_consts,
            net_conds=conds,
            net_clean_conds=array(collector.net_clean_conds),
            net_heats=heats,
            io_temps=array(collector.io_temps),
            io_temp_dt=collector.io_temp_dt,
            lp_M_diagss=collector.lp_M_diagss,
            lp_Lss=collector.lp_Lss,
        )
        return result_data

    @staticmethod
    def _calc_heat_and_time_const(coll_conds, temps, capys):
        """Calculate heat flow and time constants from iteration result data

        transform res_cond to sparse.COO array
        calculate res_heat and res_capy
        combine res_cond and res_capy to sparse.COO arrays res_conds and res_capys over all times
        """
        time_slot_count = len(temps)
        node_count = len(temps[0])
        time_consts = empty((time_slot_count, node_count))
        heats = []
        for temp, res_capy, cond, time_const in zip(
            temps, capys, coll_conds, time_consts
        ):
            # make a copy
            # emulate DOK.copy() because it does not exist,
            # don't transfer data to DOK.__init__ because internal copy is expensive
            patched_cond = DOK(
                shape=cond.shape, dtype=cond.dtype, fill_value=cond.fill_value
            )
            patched_cond.data = cond.data.copy()
            cond_data = patched_cond.data
            # infs to 0.s
            for key, value in cond_data.items():
                if value == inf:
                    cond_data[key] = 0.0
            # temp = COO.from_numpy(res_temp)
            # dtemp = (temp[:, None] - temp)  # creates big amount of memory
            # res_heat = cond.to_coo()*dtemp
            patched_cond = patched_cond.to_coo()
            cond_idxs = patched_cond.nonzero()
            dtemp = COO(
                cond_idxs, temp[cond_idxs[0]] - temp[cond_idxs[1]], patched_cond.shape
            )
            res_heat = patched_cond * dtemp
            heats.append(DOK.from_coo(res_heat))
            csum = cond.to_coo().sum(axis=1).todense()
            divide(res_capy, csum, out=time_const, where=csum != 0)
        return heats, time_consts

    @staticmethod
    def _create_sparse_arrays_over_all_times(list_of_dok_arrays, shape):
        """Combines sparse arrays for each result point in time to a sparse
        array containing all points in time.
        """
        # sparse.COO has no vindex ability needed to extract element related results
        # -> use patched DOK for now
        res = DOK(shape=shape, dtype=float64)
        for i, c in enumerate(list_of_dok_arrays):
            for k, v in c.data.items():
                res.data[(i, *k)] = v
        return res

    def __len__(self):
        return self.time_slot_count


def float_to_compact_str(number):
    """Float to compact string without leading and trailing zeros"""
    str = ('%f' % number).rstrip('0')
    if number != 0:
        str.lstrip('0')
    return str
