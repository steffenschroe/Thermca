"""One node elements with lumped parameters"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from dataclasses import dataclass, field

from numpy import array, isfinite

from thermca.baseelements import ModelElement, NetDat

if TYPE_CHECKING:
    from thermca.materials import Material

BoundTempFunc = Callable[[], float]


@dataclass
class Node(ModelElement):
    """Node element with a spatially concentrated capacity

    Args:
        capy: The thermal capacity of the node
        posn: Position with the x, y and z coordinates
        init_temp: Initial temperature of the node
        name: Name of the node
    """

    capy: float
    init_temp: float = 0.0
    posn: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ''
    _net_dat: NetDat = field(init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        self.init_temp = float(self.init_temp)
        if not (isfinite(self.capy) and self.capy > 0.0):
            ValueError("'capy' must be finite and greater than zero!")
        self._net_dat = NetDat(
            posn=array(self.posn),
            dock_lctn=array([0.0, 0.0, 0.0]),
        )

    def __hash__(self):
        return id(self)


@dataclass
class BoundNode(ModelElement):
    """Node element with bound temperature.

    Args:
        temp: Bound temperature; The temperature argument can be given
            as scalar or function. The function sets the temperature
            dependent on simulation time. Such a function can be
            created using the 'Input' element. The 'get_value' method
            of this element is a compatible function that returns a
            time dependent value.
        posn: Position with the x, y and z coordinates
        name: Name of the node

    Example:
        Using a bound temperature function on a node element;
        The function produces a temperature step after one second.

        >>> from thermca import *
        >>> with Model() as model:
        ...     point_body = Node(capy=1.)
        ...     temp = Input([[0, 1], [20, 25]])
        ...     BoundNode(point_body, temp=temp.get_value)
    """

    temp: float | BoundTempFunc
    posn: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ''
    init_temp: float = field(init=False, repr=False)
    capy: float = field(init=False, repr=False)
    _net_dat: NetDat = field(init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        if not callable(self.temp):
            self.init_temp = float(self.temp)
        self.capy = 0.0
        self._net_dat = NetDat(
            posn=array(self.posn),
            dock_lctn=array([0.0, 0.0, 0.0]),
        )

    def __hash__(self):
        return id(self)


@dataclass
class StatNode(ModelElement):
    """Stationary node element

    Its time constant is zero, and it stays always in steady state.

    This node can be used, for example, for model parts with very small
    time constants. As normal nodes they could decrease the solver time
    step size and thus increase simulation time.

    Args:
        posn: Position with the x, y and z coordinates
        name: Name of the node
    """

    posn: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ''
    init_temp: float = field(init=False, repr=False)
    capy: float = field(init=False, repr=False)
    _net_dat: NetDat = field(init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        self.capy = 0.0
        self.init_temp = 0.0
        self._net_dat = NetDat(
            posn=array(self.posn),
            dock_lctn=array([0.0, 0.0, 0.0]),
        )

    def __hash__(self):
        return id(self)


@dataclass
class MatlNode(ModelElement):
    """Node element with material based capacity

    It can be used to define a thermal capacity with material and
    volume. The temperature dependent thermal capacity of the node
    material is implicit considered if 'is_temp_dep' is True.
    Alternatively the node can represent a fluid for modeling
    convective heat exchange between solid bodies and fluids. In this
    case the temperature dependent material properties in the
    convection boundary layer ca be calculated in the film
    functions for film coefficients.

    Args:
        vol: Volume of the node
        matl: Material of the node
        posn: Position with the x, y and z coordinates
        init_temp: Initial temperature of the node
        temp_dependent: Switch for temperature dependent material behavior
        name: Name of the node
    """

    vol: float
    matl: Material
    init_temp: float = 0.0
    posn: tuple[float, float, float] = (0.0, 0.0, 0.0)
    temp_dependent = False
    name: str = ''
    _net_dat: NetDat = field(init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        if not (isfinite(self.vol) and self.vol > 0.0):
            raise ValueError("'vol' must be finite and greater than zero!")
        self._net_dat = NetDat(
            posn=array(self.posn),
            dock_lctn=array([0.0, 0.0, 0.0]),
        )

    def __hash__(self):
        return id(self)
