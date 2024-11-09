"""Links to connect node elements"""

from __future__ import annotations
from collections.abc import Sequence as CSequence
from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Callable, Sequence, Optional

from numpy.core.multiarray import ndarray  # for type hints
from numpy import isfinite

from thermca.materials import Material, Fluid
from thermca.baseelements import ModelElement
from thermca.pointnodes import StatNode, MatlNode, BoundNode, Node
from thermca.fem.fe_part import FEPartSurf
from thermca.lpm.lp_part import LPPartSurf

PointNode = Node | MatlNode | StatNode | BoundNode
PartSurf = FEPartSurf | LPPartSurf
NetElement = PointNode | PartSurf  # Linkable network nodes


class BaseLink(ModelElement):
    """Base class for Links.

    Linkable elements are surfaces of parts or point node elements.

    Args:
        elem0: First linkable element
        elem1: Second linkable element to connect to
    """

    def __init__(self, elem0: NetElement, elem1: NetElement):
        super().__init__()
        self.elem0 = elem0
        self.elem1 = elem1


LinkFunc = Callable[[ndarray, ndarray, Optional[Material]], Union[float, ndarray]]


@dataclass
class CondLink(BaseLink):
    """Connects two network elements with a thermal conductance

    The thermal conductance is measured in heat flow per surface area
    and temperature unit (e.g. W/(m²K).

    For specific heat transfer phenomena, conductances can be
    determined with the help of functions, found in the literature.
    Thermca provides some of these functions in its libraries.

    Args:
        elem0: Part surface or point node element
        elem1: part surface or point node element
        cond: Thermal conductance,  which can be given as a scalar or
            as a function with a specific signature (see below). The
            function is called in each simulation time step. It
            calculates the film coefficient dependent on the function
            arguments time and surface temperatures of both connected
            elements. Additionally, the temperature dependent
            behaviour of the materials of the surface film can be
            considered. In this case the material has to be provided
            to the 'matl' argument of the link. During simulation, this
            material is forwarded to the third parameter of the
            function.
        matl: Material provided to the link function
        name: Label of the link

    The signature of the film function::

        LinkFunc = Callable[[
                ndarray,  # Temperatures of the first element
                ndarray,  # Temperatures of the second element
                Optional[Material],  # Material or None
            ],
            ndarray  # Resulting film coefficient(s)
        ]

    The function has three default positional arguments for surface
    temperature, surroundings temperature and material of the film
    layer. There may be additional arguments in order to influence the
    film coefficient. These arguments are key word arguments with
    default values.

    Example::

        # A conductance function with temperature dependency
        def cond_func(temp0: float, temp1: float):
            return 4. + (temp0 + temp1)/2. * .001
    """

    elem0: NetElement
    elem1: NetElement
    cond: float | LinkFunc
    matl: Optional[Material] = None
    name: str = ''

    def __post_init__(self):
        super().__init__(elem0=self.elem0, elem1=self.elem1)
        if not (
            isinstance(self.elem0, NetElement) and isinstance(self.elem1, NetElement)
        ):
            raise TypeError("Only part surfaces and point node elements can be linked.")
        if isinstance(self.elem0, StatNode) and isinstance(self.elem1, StatNode):
            raise TypeError("Two 'StatNodes' can't be linked.")
        if not (callable(self.cond) or isfinite(float(self.cond))):
            raise TypeError(
                "The third argument conductance must be a function "
                "or a finite float."
            )

    @classmethod
    def multi(
        cls,
        elem: NetElement,
        elems: Sequence[NetElement],
        conds: Sequence[float | LinkFunc] | (float | LinkFunc),
        matls: Optional[Material | Sequence[Material]] = None,
        names: Optional[str | Sequence[str]] = '',
    ) -> tuple[CondLink, ...]:
        """Creates multiple 'CondLink' elements at once

        Docks is a sequence to make multiple links at once.

        Args:
            elem: Surface or point node element
            elems: Multiple surface or point node elements
            conds: Conductances that are set between the elements. If
                only a scalar value is given, it is set for each link
                created.
            matls: Material; If only one is given, it is set for each
                link created.
            names: Label or labels for the conductance links

        Returns:
            Created ``CondLinks``
        """
        conds = (conds,) * len(elems) if not isinstance(conds, CSequence) else (conds)
        matls = (matls,) * len(elems) if not isinstance(matls, CSequence) else matls
        if names is None or isinstance(names, str):
            names = (names,) * len(elems)
        return tuple(
            CondLink(elem, other_dock, cond, matl, name)
            for other_dock, cond, matl, name in zip(elems, conds, matls, names)
        )


def sum_films(link_func, film):
    """Sums film coefficients together

    The arguments can be either a link function and a scalar
    or two link functions.
    """

    def film_sum_func_func(surf_temp, fluid_temp, matl):
        return link_func(surf_temp, fluid_temp, matl) + film(
            surf_temp, fluid_temp, matl
        )

    def film_sum_func_scalar(surf_temp, fluid_temp, fluid):
        return link_func(surf_temp, fluid_temp, fluid) + film

    if callable(film):
        return film_sum_func_func
    else:
        return film_sum_func_scalar


@dataclass
class FilmLink(BaseLink):
    """Connects a part surface to any network element with a boundary
    film

    The boundary film models the heat transfer through a layer formed
    on top of a solid surface. The layer may consist of a fluid or
    solid. A 'film coefficient' or 'heat' transfer coefficient' is used
    as a simplified model of the heat transfer. The film coefficient is
    measured in heat flow per surface area and temperature unit,
    e.g. W/(m²K).

    For typical heat transfer phenomena like convection and heat
    radiation, the coefficients can be determined with the help of
    tables and functions found in the literature. Thermca provides
    functions in libraries to compute those coefficents.

    If surfaces with different surface areas are connected, the mean
    area is used to calculate the resulting conductance.

    Args:
        surf: Coupling surface of a part
        elem: Network element to connect to
        film: Film coefficient, which can be given as a scalar or as a
            function with a specific signature (see below). The
            function is called in each simulation time step. It
            calculates the film coefficient dependent on the function
            arguments time and surface temperatures of both connected
            elements. Additionally, the temperature dependent
            behaviour of the materials of the surface film can be
            considered. In this case the material has to be provided
            to the 'matl' argument of the link. During simulation, this
            material is forwarded to the third parameter of the
            function.
        matl: Material provided to the link function
        name: Name of the link

    The signature of the film function::

        LinkFunc = Callable[[
                ndarray,  # Temperatures of the surface
                ndarray,  # Temperatures of second surface or point node
                Optional[Material],  # Material or None
            ],
            Union[float, ndarray]  # Resulting film coefficient(s)
        ]

    The function has three default positional arguments for surface
    temperature, surroundings temperature and material of the film
    layer. There may be additional arguments in order to influence the
    film coefficient. These arguments are key word arguments with
    default values.

    Example::

        Creating and using a link function considering temperature
        dependency.

        >>> from numpy import ndarray
        >>> from thermca import *
        ...
        >>> def link_func(
        ...     solid_temp: ndarray,
        ...     fluid_temp: ndarray,
        ...     fluid: Fluid
        ... ):
        ...     return 4. + (solid_temp + fluid_temp)/2. * .001
        ...
        >>> with Model() as model:
        ...     block = lp_parts.block()
        ...     air_node = BoundNode(temp=0.)
        ...     FilmLink(block.surf.right, air_node, link_func)

    There is a convention in the libraries that the functions that are
    included will always return 'LinkFunction's'. The functions may have
    default key word arguments for model parameters. This arguments
    can be customized during model building by calling the library
    functions with changed keyword arguments. This is demonstrated
    below using a library film function for estimating the film
    coefficient for convection and radiation in rooms. The default
    emission coefficient of the function is changed to .5. This adjusts
    the coefficient for a property of a glass surface.

    Example::

        >>> from thermca import *
        ...
        >>> with Model() as model:
        ...     environment = BoundNode(temp=0.)
        ...     glass_brick = lp_parts.block(
        ...         matl=solids.glass,
        ...         name='glass_brick',
        ...    )
        ...    FilmLink(
        ...        glass_brick.surf.left,
        ...        environment,
        ...        film=combd_film.conv_radn_room(emis_coef=.5)
        ...   )
    """

    surf: PartSurf
    elem: NetElement
    film: float | LinkFunc
    matl: Optional[Material] = None
    name: str = ''

    def __post_init__(self):
        super().__init__(elem0=self.surf, elem1=self.elem)
        if not isinstance(self.surf, PartSurf):
            raise TypeError("The first argument must be a surface of a part!")
        if not isinstance(self.elem, (PartSurf, PointNode)):
            raise TypeError(
                "The second argument must be a surface or a point node element!"
            )
        if not (callable(self.film) or isfinite(float(self.film))):
            raise TypeError("The 'film' argument must be a function or a finite float!")

    @classmethod
    def multi(
        cls,
        surfs: PartSurf | Sequence[PartSurf],
        elems: NetElement | Sequence[NetElement],
        films: Sequence[float | LinkFunc] | (float | LinkFunc),
        matls: Optional[Material | Sequence[Material]] = None,
        tuple_names: Optional[str | Sequence[str]] = None,
        names: Optional[str | Sequence[str]] = '',
    ) -> tuple[FilmLink, ...]:
        """Factory that creates multiple ``FilmLinks`` at once.

        One of the arguments surfs and elems can be a sequence to make
        multiple links at once.

        Args:
            surfs: Surface(s) of a part
            elems: Surface(s) or point node element(s)
            films: Film (heat transfer) coefficients; If only one
                value is given, it is set for each link created.
            matls: Material; If only one is given, it is set for each
                link created.
            tuple_names: Names of the links for access over names of the
                returned namedtuple.
            names: Name or names for the conductance links

        Returns:
            The created 'FilmLink's
        """
        if isinstance(surfs, CSequence):
            link_count = len(surfs)
            elems = (elems,) * link_count
        elif isinstance(elems, CSequence):
            link_count = len(elems)
            surfs = (surfs,) * link_count
        else:
            raise TypeError("Only one of the first two arguments may be a sequence.")
        films = (films,) * link_count if not isinstance(films, CSequence) else films
        matls = (matls,) * link_count if not isinstance(matls, CSequence) else matls
        if names is None or isinstance(names, str):
            names = (names,) * len(elems)

        # names needed for the names of the namedtuple that will be returned
        if tuple_names is None:
            tuple_names = tuple('link' + str(i) for i in range(link_count))
        links = [
            FilmLink(surf, dock, film, matl, name)
            for surf, dock, film, matl, name in zip(surfs, elems, films, matls, names)
        ]
        return namedtuple('MultiLinks', tuple_names)(*links)

    @classmethod
    def pairs(
        cls,
        elem_pairs: Sequence,
        films: Sequence[float | LinkFunc] | (float | LinkFunc),
        matls: Optional[Material | Sequence[Material]] = None,
        tuple_names: Optional[str | Sequence[str]] = None,
        names: Optional[str | Sequence[str]] = '',
    ) -> tuple[FilmLink, ...]:
        """Creates multiple 'FilmLinks' at once.

        Args:
            elem_pairs: Sequence of element pairs that are to be linked
            films: Film (heat transfer) coefficients; If only one
                value is given, it is set for each link created.
            matls: Material; If only one is given, it is set for each
                link created.
            tuple_names: Names of the links for access over names of the
                returned namedtuple.
            names: Name or names for the conductance links

        Returns:
            Created 'FilmLinks'
        """
        elem_pairs = list(elem_pairs)
        link_count = len(elem_pairs)
        surfs, docks = zip(*elem_pairs)
        films = (films,) * link_count if not isinstance(films, CSequence) else films
        matls = (matls,) * link_count if not isinstance(matls, CSequence) else matls
        if names is None or isinstance(names, str):
            names = (names,) * link_count
        # names needed for the names of the namedtuple that will be returned
        if tuple_names is None:
            tuple_names = tuple('link' + str(i) for i in range(link_count))
        links = [
            FilmLink(surf, dock, film, matl, name)
            for surf, dock, film, matl, name in zip(surfs, docks, films, matls, names)
        ]
        return namedtuple('PairLinks', tuple_names)(*links)


VolFlowFunc = Callable[
    [
        ndarray,
        ndarray,
        Fluid,
    ],  # Temperatures of destination and source nodes, fluid of nodes
    Union[float, ndarray],  # Resulting volume flow(s)
]


@dataclass
class FlowLink(BaseLink):
    """Create heat transport between material point node elements.

    Args:
        dest_node: Destination node element
        src_node: Source node element
        vol_flow: Volume flow that is set between the two elements.
            Volume flow = cross-sectional area * flow velocity;
            It can be set as a positive finite float or a function
            returning such a value.
            The function can calculate the volume flow dependent on the
            function arguments time and the temperatures of the
            connected nodes. Additionally, the temperature dependent
            behaviour of fluids during heat transfer can be considered.
            The fluid of the material nodes is given in the volume
            flow function as the third parameter and can be used to
            calculate temperature dependent material behaviour.
        matl: Material provided to the link function
        name: Name of the link

    The signature of the volume flow function::

        VolFlowFunc = Callable[[
                ndarray,  # Temperatures of the destination nodes
                ndarray,  # Temperatures of the source nodes
                Fluid  # Fluid material of the 'dest_node'
            ],
            Union[float, ndarray]  # Resulting volume flow(s)
        ]
    """

    dest_node: MatlNode | BoundNode
    src_node: MatlNode | BoundNode
    vol_flow: float | VolFlowFunc
    matl: Optional[Material] = None
    name: str = ''

    def __post_init__(self):
        super().__init__(elem0=self.dest_node, elem1=self.src_node)
        if not (
            isinstance(self.dest_node, (MatlNode, BoundNode))
            and isinstance(self.src_node, (MatlNode, BoundNode))
        ):
            raise TypeError("Only 'MatlNodes' and 'BoundNode' can be linked with 'FlowLink'!")

        if not (
            callable(self.vol_flow)
            or (isfinite(float(self.vol_flow)) and self.vol_flow > 0)
        ):
            raise TypeError(
                "The 'vol_flow' argument must be a "
                "positive finite float or a function."
            )
