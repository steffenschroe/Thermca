"""Classes for properties of solid and fluid materials."""

import warnings
import numbers
from functools import reduce
from typing import Union, Sequence

# fmt: off
from numpy import (
    array, interp, amin, amax, linspace, float64, union1d, full_like, clip, printoptions
)
# fmt: on
from numpy import sum as asum
from numpy.core.multiarray import ndarray


def func_to_table(temp, func):
    temp_arr = array(temp)
    return temp_arr, func(temp_arr)


class Solid:
    """Solid state materials

    Defines a solid state material. Inputs are thermal material
    properties, which optionally may be temperature dependent.

    The properties can be given as a scalar or as a two-dimensional
    table. The table contains temperatures in the first row and
    corresponding property values in the second row. The temperatures
    must be increasing. For performance, the temperature points of dens
    and spec_heat should be the same.
    Property units are given in SI standard and temperatures in °C.

    Args:
        condy: Thermal conductivity
        spec_heat: Specific heat capacity
        dens: Density
        limits: Temperatures that limit the valid range of the
            material properties. If set to None it is automatically
            determined from the temperature limits of the given
            property table data. Tables with one temperature entry are
            ignored thereby.
        name: name of the material.

    Examples::

        >>> from thermca import Solid
        >>>
        >>> # Create aluminium material with scalar properties:
        >>> al = Solid(
        ...     condy=238,
        ...     dens=2700,
        ...     spec_heat=945,
        ...     name='Aluminium')
        >>>
        >>> # Create aluminium material with material properties as table:
        >>>
        >>> from thermca import Solid, func_to_table
        >>>
        >>> al = Solid(
        ...     condy=[[-100, 0, 100, 300, 700],  # Temperatures
        ...            [241, 236, 240, 233, 92]], # Condy. values
        ...     # The temperature points of dens and spec_heat should be
        ...     # the same for maximum performance
        ...     dens=func_to_table(
        ...         [-196, -100, -0.,  100,  300,  500],
        ...         lambda temp: 2714 - temp*0.185),
        ...     spec_heat=[[-196, -100, -0.,  100,  300,  500],
        ...                [336., 743., 880., 937., 1021., 1130.]],
        ...     name='Aluminium')
        >>>
        >>> # Create aluminium material with material properties given as
        >>> # functions by using the helper function `func_to_table`:
        >>>
        >>> from thermca import Solid, func_to_table
        >>> from numpy import linspace
        >>>
        >>> al_alloy = Solid(
        ...     condy=func_to_table(
        ...         temp=linspace(0, 400, 11),
        ...         func=lambda temp: 164.257 + 0.082*temp - 1.071e-4),
        ...     dens=2700,
        ...     spec_heat=func_to_table(
        ...         linspace(0, 400, 11),
        ...         lambda temp: 1100.*(1. + .0005*(temp - 20.))),
        ...     name='Aluminium alloy')
    """

    def __init__(
        self,
        *,
        condy: Union[Sequence, float],  # 2d table, or scalar
        dens: Union[Sequence, float],
        spec_heat: Union[Sequence, float],
        limits: Sequence = None,
        name: str = None
    ):
        self._condy = self.to_2d_array(condy)
        self._dens = self.to_2d_array(dens)
        self._spec_heat = self.to_2d_array(spec_heat)

        if limits is None:
            self.limits = float('-inf'), float('inf')
            self.set_limits(self._condy)
            self.set_limits(self._dens)
            self.set_limits(self._spec_heat)
        else:
            self.limits = limits

        self.name = name

        # Precalculate volumetric heat capacity for performance
        vol_capy_temp = self._temp_union(self._dens[0], self._spec_heat[0])
        self._vol_capy_temp = vol_capy_temp
        dens = interp(vol_capy_temp, self._dens[0], self._dens[1])
        spec_heat = interp(vol_capy_temp, self._spec_heat[0], self._spec_heat[1])
        self._vol_capy = dens * spec_heat

    def condy_interp(self, temp):
        return interp(temp, self._condy[0], self._condy[1])

    def spec_heat_interp(self, temp):
        return interp(temp, self._spec_heat[0], self._spec_heat[1])

    def dens_interp(self, temp):
        return interp(temp, self._dens[0], self._dens[1])

    def vol_capy_interp(self, temp):
        return interp(temp, self._vol_capy_temp, self._vol_capy)

    @staticmethod
    def to_2d_array(prop):
        if isinstance(prop, numbers.Real):
            return array(([float('nan')], [prop]), dtype=float64)
        elif isinstance(prop, (list, tuple, ndarray)):
            prop = array(prop, dtype=float64)
            if not prop.ndim == 2:
                raise TypeError('2d sequence in regular array shape expected.')
            return prop
        else:
            TypeError(
                'Arguments must be a float, or a table of '
                'temperatures and values as a two dimensional array like '
                'sequence.'
            )

    def set_limits(self, prop):
        if prop.shape[1] > 1:
            self.limits = (
                max(self.limits[0], prop[0, 0]),
                min(self.limits[1], prop[0, -1]),
            )

    def check_limits(self, temp: ndarray):
        """Checks, if the temperatures are in the valid range for the
        material properties.

        Args:
            temp: Temperatures to be checked for validity
        """
        if (amin(temp) < self.limits[0]) or amax(temp) > self.limits[1]:
            warnings.warn(
                'Temperatures not valid for '
                + self.__class__.__name__
                + ' material '
                + (self.name or '')
                + '.'
            )

    @staticmethod
    def _temp_union(temp0, temp1):
        if len(temp0) == 1:
            return temp1
        elif len(temp1) == 1:
            return temp0
        else:
            # Temperatures outside the overlap range are clipped
            # because they are not considered valid in the mix.
            temp0c = clip(temp0, temp1[0], temp1[-1])
            temp1c = clip(temp1, temp0[0], temp0[-1])
            return union1d(temp0c, temp1c)

    @classmethod
    def mix(cls, matls, shares, name=None):
        """Homogenised mix of multiple existing materials

        Args:
            matls: Sequence of materials
            shares: Sequence of volume fractions of the materials.
            name: name of the material mix.

        Example::

            # Create aluminium foam material with lower relative density as
            # aluminium alloy with a 'void' volume fraction:

            >>> al_foam = Solid.mix(
            ...     matls=(solids.al_alloy, fluids.air),
            ...     shares=(.8, .2),
            ...     name='aluminium_foam')

        """
        nshares = [float(s) / sum(shares) for s in shares]
        condys = [m._condy for m in matls]
        denss = [m._dens for m in matls]
        spec_heats = [m._spec_heat for m in matls]
        # TODO: more elegant way to handle Fluids
        if issubclass(cls, Fluid):
            viscs = [m._visc for m in matls]

        condy_temp = reduce(cls._temp_union, (c[0] for c in condys))
        dens_temp = reduce(cls._temp_union, (d[0] for d in denss))
        spec_heat_temp = reduce(cls._temp_union, (h[0] for h in spec_heats))
        if issubclass(cls, Fluid):
            visc_temp = reduce(cls._temp_union, (v[0] for v in viscs))

        condy_mix = asum(
            [interp(condy_temp, c[0], c[1]) * s for c, s in zip(condys, nshares)],
            axis=0,
        )
        dens_mix = asum(
            [interp(dens_temp, d[0], d[1]) * s for d, s in zip(denss, nshares)], axis=0
        )
        denss_interp = [interp(spec_heat_temp, d[0], d[1]) for d in denss]
        spec_heats_interp = [interp(spec_heat_temp, sh[0], sh[1]) for sh in spec_heats]
        # Homogenised specific heat has to be mixed in with its mass fraction
        mass = array([d * s for d, s in zip(denss_interp, nshares)])
        sum_mass = asum(mass, axis=0)
        mass_frac = mass / sum_mass
        ci = [sh * mf for sh, mf in zip(spec_heats_interp, mass_frac)]
        spec_heat_mix = asum(ci, axis=0)
        if issubclass(cls, Fluid):
            visc_mix = asum(
                (interp(visc_temp, v[0], v[1]) * s for v, s in zip(viscs, nshares)),
                axis=0,
            )

        if name is None:
            names = [m.name for m in matls]
            if None not in names:
                names = [
                    l + ' {0:0.3g} %'.format(s * 100) for l, s in zip(names, nshares)
                ]
                name = ', '.join(names)

        if issubclass(cls, Fluid):
            return Fluid(
                condy=[condy_temp, condy_mix],
                dens=[dens_temp, dens_mix],
                spec_heat=[spec_heat_temp, spec_heat_mix],
                visc=[visc_temp, visc_mix],
                name=name,
            )
        else:
            return Solid(
                condy=[condy_temp, condy_mix],
                dens=[dens_temp, dens_mix],
                spec_heat=[spec_heat_temp, spec_heat_mix],
                name=name,
            )

    def _prop_str(
        self,
        prop_name_str,
    ):
        prop = self.__dict__['_' + prop_name_str]
        limits = (0.0, 100.0) if self.limits is None else self.limits
        if isinstance(prop, numbers.Real):
            return '    ' + prop_name_str + '=' + str('{0:0.3g}'.format(prop)) + ','
        else:
            temps = prop[0]
            vals = prop[1]
        with printoptions(formatter={'float': lambda x: "{0:0.3g}".format(x)}):
            alines = repr(array([temps, vals], dtype=float64)).splitlines()
        lines = []
        lines.append('    ' + prop_name_str + '=' + alines[0])
        for aline in alines[1:]:
            lines.append('    ' + ' ' * (len(prop_name_str) + 1) + aline)
        lines[-1] += ','
        return '\n'.join(lines)

    _prop_names = ['condy', 'dens', 'spec_heat']

    def __repr__(self):
        lines = []
        lines.append(self.__class__.__name__ + '(')
        if self.name is not None:
            lines.append("    name='" + self.name + "',")
        for pname in self._prop_names:
            lines.append(self._prop_str(pname))
        if self.limits is not None:
            lines.append('    limits=({0:0.3g}, {1:0.3g})'.format(*self.limits) + ',')

        lines.append(')')
        return '\n'.join(lines)

        # def __repr__(self):
        #    return str(self) + '  # at 0x{}'.format(id(self))


class Fluid(Solid):
    """Fluid material

    Defines a fluid material. Inputs are thermal material
    properties, which optionally may be temperature dependent.

    The properties can be given as a scalar or as a two-dimensional
    table. The table contains temperatures in the first row and
    corresponding property values in the second row. The temperatures
    must be increasing.
    Property units are given in SI standard and temperatures in °C.

    Args:
        condy: Thermal conductivity
        spec_heat: Specific heat capacity
        dens: Density
        visc: kinematic viscosity
        limits: Temperatures that limit the valid range of the
            material properties. If set to None it is automatically
            determined from the temperature limits of the given
            property table data. Tables with one temperature entry are
            ignored thereby.
        name: Name of the material.
    """

    def __init__(
        self,
        *,
        condy: Union[Sequence, float],  # 2d table, or scalar
        dens: Union[Sequence, float],
        spec_heat: Union[Sequence, float],
        visc: Union[Sequence, float],
        limits: Sequence = None,
        name: str = None
    ):
        super().__init__(
            condy=condy,  # 2d table, function or scalar
            dens=dens,
            spec_heat=spec_heat,
            limits=limits,
            name=name,
        )

        self._condy = self.to_2d_array(condy)
        self._dens = self.to_2d_array(dens)
        self._spec_heat = self.to_2d_array(spec_heat)
        self._visc = self.to_2d_array(visc)

        if limits is None:
            self.limits = float('-inf'), float('inf')
            self.set_limits(self._condy)
            self.set_limits(self._dens)
            self.set_limits(self._spec_heat)
            self.set_limits(self._visc)
        else:
            self.limits = limits

    def visc_interp(self, temp):
        return interp(temp, self._visc[0], self._visc[1])

    _prop_names = ['condy', 'dens', 'spec_heat', 'visc']


Material = Union[Solid, Fluid]


def plot(matl, fallback_limits=(0.0, 100.0), mar=0.2):
    """Generates plots of material parameters over temperature.

    Args:
        matl (Material): Solid or fluid material
        mar (float): margin to plot over the defined temperature range
            as proportion of the range
        fallback_limits(tuple): This temperature limits are used
            as plotting range if the material has no temperature
            limits.
    """
    import matplotlib.pyplot as plt

    if matl.limits[0] == float('-inf'):
        dt = fallback_limits[1] - fallback_limits[0]
        tmin = fallback_limits[0] - dt * mar
        tmax = fallback_limits[1] + dt * mar
    else:
        dt = matl.limits[1] - matl.limits[0]
        tmin = matl.limits[0] - dt * mar
        tmax = matl.limits[1] + dt * mar
    # Calculate datapoints as well as range borders and margins for temperature
    pnt_count = 100
    temp = linspace(tmin, tmax, pnt_count)
    # create the plots
    fig, (left_ax_left, right_ax_left) = plt.subplots(1, 2, figsize=(15, 4))
    left_ax_left.set_title(matl.name or '', fontsize=13)
    trans = left_ax_left.get_xaxis_transform()  # x in data untis, y in axes fraction
    if matl.limits[0] != float('-inf'):
        left_ax_left.axvline(matl.limits[0], color='lightgrey')
        left_ax_left.annotate(
            ' lower limit', xy=(matl.limits[0], 0.9), xycoords=trans, color='grey'
        )
    if matl.limits[1] != float('inf'):
        left_ax_left.axvline(matl.limits[1], color='lightgrey')
        left_ax_left.annotate(
            ' upper limit',
            xy=(matl.limits[1], 0.9),
            xycoords=trans,
            color='grey',
            horizontalalignment='left',
        )
    right_ax_left.set_title(matl.name, fontsize=13)
    trans = right_ax_left.get_xaxis_transform()  # x in data untis, y in axes fraction
    if matl.limits[0] != float('-inf'):
        right_ax_left.axvline(matl.limits[0], color='lightgrey')
        right_ax_left.annotate(
            ' lower limit', xy=(matl.limits[0], 0.9), xycoords=trans, color='grey'
        )
    if matl.limits[1] != float('inf'):
        right_ax_left.axvline(matl.limits[1], color='lightgrey')
        right_ax_left.annotate(
            ' upper limit',
            xy=(matl.limits[1], 0.9),
            xycoords=trans,
            color='grey',
            horizontalalignment='left',
        )
    left_ax_right = plt.twinx(left_ax_left)
    if matl._condy.shape[1] > 1:
        left_ax_left.plot(matl._condy[0], matl._condy[1], color='green')
    else:
        left_ax_left.plot(temp, full_like(temp, matl._condy[1]), color='green')
    left_ax_left.set_xlabel(u'Temperature [\u00b0C]')
    left_ax_left.set_ylabel(u'Conductivity [W/(m\u00B7K)]', color='green')
    left_ax_right.set_ylabel(u'Density [kg/m\u00B3]', color='red')
    if matl._dens.shape[1] > 1:
        left_ax_right.plot(matl._dens[0], matl._dens[1], color='red')
    else:
        left_ax_right.plot(temp, full_like(temp, matl._dens[1]), color='red')
    right_ax_left.set_xlabel(u'Temperature [\u00b0C]')
    right_ax_left.set_ylabel(u'Specific heat capacity [J/(kg·K)]', color='blue')
    if matl._spec_heat.shape[1] > 1:
        right_ax_left.plot(matl._spec_heat[0], matl._spec_heat[1], color='blue')
    else:
        right_ax_left.plot(temp, full_like(temp, matl._spec_heat[1]), color='blue')

    if isinstance(matl, Fluid):
        right_ax_right = plt.twinx(right_ax_left)
        if matl._visc.shape[1] > 1:
            right_ax_right.plot(matl._visc[0], matl._visc[1], color='magenta')
        else:
            right_ax_right.plot(temp, matl._visc[1], color='magenta')
        right_ax_right.set_ylabel(u'Kinematic viscosity [m\u00B3/s]', color='magenta')
        right_ax_right.set_ylim(ymin=0)

    left_ax_left.set_ylim(ymin=0)
    right_ax_left.set_ylim(ymin=0)
    left_ax_right.set_ylim(ymin=0)

    plt.tight_layout()  # rearrange space between plots

    plt.show()
