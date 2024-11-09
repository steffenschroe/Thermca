"""Container for model elements"""

from collections import deque, OrderedDict
from functools import wraps

from thermca.source import HeatSource, FluxSource
from thermca.pointnodes import Node, StatNode, MatlNode, BoundNode
from thermca.fem.fe_part import FEPart, FEPartSurf
from thermca.lpm.lp_part import LPPart, LPPartSurf
from thermca.links import CondLink, FilmLink, FlowLink
from thermca.input import Input
import thermca.plot.model as plot

# After having nearly the same idea to use a context manager to define
# the active model for model elements, some implementation ideas were
# borrowed from the nengo project.
# https://github.com/nengo/nengo/blob/master/nengo/network.py (17.08.2015)

# As a rough guide, a separate element class should be created,
# if a symbol is necessary to express its individual behaviour.


class Model:
    """Assembly of model elements

    The 'Model' is used to build the thermal model by adding and
    parameterizing elements. Available elements are:

    +---------------------------+-------------------------------------+
    | Point nodes for various   | 'Node', 'MatlNode', 'StatNode' and  |
    | purposes                  | 'BoundNode'                         |
    |                           |                                     |
    | Parts for describing      | 'LPPart' and 'FEPart'               |
    | solids                    |                                     |
    |                           |                                     |
    | Links to couple point     | 'CondLink', 'FilmLink' and          |
    | nodes and parts           | 'FlowLink'                          |
    |                           |                                     |
    | Sources to feed heat into | 'HeatSource' and 'FluxSource'       |
    | parts and point nodes     |                                     |
    |                           |                                     |
    | Time-varying model inputs | 'Input'                             |
    +---------------------------+-------------------------------------+

    'Model' is implemented as a context manager. Add elements by
    creating them inside a with block.

    Args:
        name: Name of the model
    """

    # Rule: expose element lists and use the dict logic only inside Network

    reference_stack = deque()  # model instances during model creation

    def __init__(self, name: str = None):
        self.name = name
        # objects to connect the elements with the model
        self._models = []
        # fmt: off
        self._elems = OrderedDict((
            (Node, []), (MatlNode, []), (StatNode, []), (BoundNode, []),
            (LPPart, []), (LPPartSurf, []),
            (FEPart, []), (FEPartSurf, []),
            (CondLink, []), (FilmLink, []), (FlowLink, []),
            (HeatSource, []), (FluxSource, []),
            (Input, []),
        ))
        # fmt: on

        if Model.reference_stack:  # If sub-model
            self._models.append(self)

    @staticmethod
    def _append(elem):
        """Adds an element to the model"""
        if not Model.reference_stack:
            raise RuntimeError(
                f"{elem} should be used in a 'with' code block using "
                "a 'Model' instance as the context."
            )
        current_model = Model.reference_stack[-1]
        for cls in elem.__class__.__mro__:
            if cls in current_model._elems:
                current_model._elems[cls].append(elem)
                break
        else:
            raise RuntimeError(f"{elem} could not be added to the model.")

    def __enter__(self):
        Model.reference_stack.append(self)
        return self

    def __exit__(self, _0, _1, _2):
        if Model.reference_stack.pop() is not self:
            raise RuntimeError(
                "Got the wrong model context during exit of the with code block."
            )

    def __contains__(self, elem):
        """Returns true, if an element is inside the model"""
        return type(elem) in self._elems and (elem in self._elems[type(elem)])

    def _get_elems_of_types(self, elem_types):
        """Returns a list of all elements of the specified types from
        this model only
        """
        elems = []
        try:  # Try if iterable of element types
            for elem_type in elem_types:
                elems.extend(self._elems[elem_type])
        except TypeError:  # Assume only one type given
            elems = self._elems[elem_types]
        return elems

    def _get_all_elems_of_type(self, elem_type):
        """Returns a list of all elements of the specified type from
        this model and its sub-models
        """
        elems = list(self._elems[elem_type])
        for submodel in self._models:
            elems.extend(submodel._elems[elem_type])
        return elems

    def _get_all_elems_of_types(self, elem_types):
        """Returns a list of all elements of the specified types in this
        model and its sub-models
        """
        elems = []
        try:  # Try if iterable of element types
            for elem_type in elem_types:
                elems.extend(self._get_all_elems_of_type(elem_type))
        except TypeError:  # Assume only one type given
            elems = self._get_all_elems_of_type(elem_types)
        return elems

    def __str__(self):
        return f"{self.__class__.__name__} '{self.name or ''}'"

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name or ''}'" f" at 0x{id(self)}>"

    @wraps(plot.elements)
    def plot(self, *args, **kwargs):
        return plot.elements(self, *args, **kwargs)
