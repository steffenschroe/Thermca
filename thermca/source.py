"""Heat and flux sources"""

from __future__ import annotations

from collections.abc import Sequence as CSequence
from typing import Sequence, Union, Callable, Tuple

from numpy import ndarray

from thermca.baseelements import ModelElement

from thermca.pointnodes import StatNode, MatlNode, BoundNode, Node
from thermca.fem.fe_part import FEPartSurf
from thermca.lpm.lp_part import LPPartSurf

PointNode = Node | MatlNode | StatNode | BoundNode
PartSurf = FEPartSurf | LPPartSurf
NetElement = PointNode | PartSurf  # Linkable network nodes


HeatFunc = Callable[[float], float]


def sum_heat(heat_func, heat):
    """Sums heat of function and a scalar or of two heat functions"""

    def heat_sum_func_func(temp):
        return heat_func(temp) + heat(temp)

    def heat_sum_func_scalar(temp):
        return heat_func(temp) + heat

    if callable(heat):
        return heat_sum_func_func
    else:
        return heat_sum_func_scalar


class HeatSource(ModelElement):
    """Source that feds heat into an element with nodes.

    Args:
        net_elem: Node elements
        heat: Heat (power) that will be fed in to the element. If given
            as a function, it calculates heat that may depend on mean
            nodes temperature of the element nodes.
        name: Label

    If heat is a function, it must have the following signature:

    The signature of the heat source function::

        HeatFunc = Callable[[
                float  #  Mean nodes temperature
            ],
            float  # Heat fed into the element
        ]
    """

    def __init__(
        self, net_elem: NetElement, heat: Union[float, HeatFunc], name: str = None
    ):
        super(HeatSource, self).__init__()
        if not isinstance(net_elem, NetElement):
            raise TypeError("Heat can only be fed into point nodes and surfaces!")
        self.net_elem = net_elem
        self.heat = heat

    @classmethod
    def multi(
        cls,
        net_elems: Sequence[NetElement],
        heats: Union[float, HeatFunc, Sequence[float]],
        labels: Union[str, Sequence[str]] = None,
    ) -> Tuple['HeatSource', ...]:
        """Factory that creates multiple heat sources at once.

        Args:
            net_elems: Node elements
            heats: Heat (power) that will be fed in to the elements.
                See ``HeatSource`` argument
            labels: Labels

        Returns:
            Heat sources
        """
        # If heats is not iterable every element gets the heat
        heats = (
            (heats,) * len(net_elems) if not isinstance(heats, CSequence) else (heats)
        )
        labels = (
            (labels,) * len(net_elems)
            if (not isinstance(labels, CSequence) or isinstance(labels, str))
            else (labels)
        )
        return tuple(
            HeatSource(elem, heat, name=label)
            for (elem, heat, label) in zip(net_elems, heats, labels)
        )


FluxFunc = Callable[[ndarray], ndarray]


class FluxSource(ModelElement):
    """Source that feds a heat flux into a body surface.

    Args:
        surf: Body surface
        flux: Heat flux density on the surface. It is a heat rate
            (power) per unit area. It can be given as a function that
            calculates the heat flux density dependent on the function
            argument tha contains the nodes temperature of the surface.
        name: Name

    The signature of the heat source function::

        FluxFunc = Callable[[
                ndarray  #  Surface nodes temperatures
            ],
            ndarray  # Heat fed into the surface nodes
        ]
    """

    def __init__(
        self, surf: LPPartSurf | FEPartSurf, flux: float | FluxFunc, name: str = None
    ):
        if not isinstance(surf, (LPPartSurf, FEPartSurf)):
            raise TypeError("Heat flux can only be fed into surfaces of parts!")
        super().__init__()
        self.surf = surf
        self.flux = flux
