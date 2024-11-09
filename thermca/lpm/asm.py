"""Assembly of block and cylinder geometry elements"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Union, TYPE_CHECKING
from functools import wraps

if TYPE_CHECKING:
    from thermca.lpm.cube import Cube
    from thermca.lpm.cyl import Cyl

from thermca.baseelements import check_name
from thermca.plot.asm import asm_elem as plot_asm


class AssemblyElement:
    """Base for body geometry elements."""

    def __init__(self):
        Asm._append(self)


@dataclass
class BodyFace:
    name: str
    body: Union[Cube, Cyl]


@dataclass
class Surf(AssemblyElement):
    """Coupling surface

    Group of body surfaces used as a coupling interface for body
    assemblies in the thermal model.

    Args:
        faces: Body surfaces
        name: Name of the coupling surface.
    """
    name: str
    faces: list | tuple

    def __post_init__(self):
        AssemblyElement.__init__(self)
        check_name(self.name)

    def area(self):
        return sum(face.area() for face in self.faces)


@dataclass
class ForceConts(AssemblyElement):
    """Collection of enforced body face contacts

    Allows the fusion of bodies whose surfaces are not in contact.

    Args:
        face_pairs: Pairs of body faces to be merged.
        name: Name of contact collection
    """
    face_pairs: list[tuple[BodyFace, BodyFace]]
    name: str = ''

    def __post_init__(self):
        AssemblyElement.__init__(self)


class Asm:
    """Assembly of cuboid and cylinder elements

    The 'Assembly' is used to build part geometries by adding and
    parameterizing elements. Available elements are:

    +---------------------+------------------+
    | Bodies for basic    | 'Cube' and 'Cyl' |
    | solid geometries    |                  |
    |                     |                  |
    | Surfaces to define  | 'Surf'           |
    | coupling interfaces |                  |
    |                     |                  |
    | Glue for bodies not | 'ForceConts'     |
    | in contact          |                  |
    +---------------------+------------------+

    The body assembly is implemented as a context manager. Add body
    assembly elements by creating them inside a with block.

    If faces of two bodies are in contact, then these bodies get merged
    on the faces. A special case are subdivided basic bodies. Their
    faces can only be merged if they have the same number of sub-faces
    or one non subdivided face is in contact with a subdivided face.

    If two bodies are supposed to be thermally coupled,
    but not touching each other: merge them with 'ForceConts'.


    Example:
        >>> from thermca import *
        >>> with Asm() as hammer:
        ...     handle = Cyl(
        ...         lgth=.3,
        ...         lgth_div=4,
        ...         outer_rad=.01,
        ...     )
        ...     head = Cube(
        ...         posn=(.3, -.03, -.0125),
        ...         width=.025,
        ...         hgt=.06,
        ...         depth=.025,
        ...     )
        ...     Surf('handle', [handle.face.base, handle.face.outer])
        ...     Surf('head', head.face)  # All cube faces
    """

    reference_stack = deque()  # Body assemblies instances during model creation

    def __init__(self, name: str = None):
        from thermca.lpm.cube import Cube
        from thermca.lpm.cyl import Cyl
        self.name = name
        self._assemblies = []
        self._elems = {
            Cube: [],
            Cyl: [],
            Surf: [],
            ForceConts: [],
        }

        if Asm.reference_stack:  # Sub-context
            self._assemblies.append(self)

    @staticmethod
    def _append(elem):
        """Adds an element to the body assembly"""
        if not Asm.reference_stack:
            raise RuntimeError(f"{type(elem)} '{elem.name}' should be used in a 'with' "
                               f"code block using a 'BodyASM' instance as the context.")
        current_asm = Asm.reference_stack[-1]
        for cls in elem.__class__.__mro__:
            if cls in current_asm._elems:
                current_asm._elems[cls].append(elem)
                break
        else:
            raise RuntimeError(f"{type(elem)} '{elem.name}' could not be added to the assembly.")

    def __enter__(self):
        Asm.reference_stack.append(self)
        return self

    def __exit__(self, _0, _1, _2):
        if len(Asm.reference_stack) == 1:  # Exit the root assembly
            pass

        if Asm.reference_stack.pop() is not self:
            raise RuntimeError("Got the wrong model context during exit of the "
                               "with code block.")

    def __contains__(self, elem):
        """Returns true, if an element is inside the assembly"""
        return type(elem) in self._elems and (
            elem in self._elems[type(elem)])

    def _get_all_elems_of_type(self, elem_type):
        """Returns a list of all elements of the specified type from
        this context, and it's nested sub-contexts
        """
        elems = list(self._elems[elem_type])
        for sub_asm in self._assemblies:
            elems.extend(self._elems[elem_type])
        return elems

    @wraps(plot_asm)
    def plot(self, *args, **kwargs):
        return plot_asm(self, *args, **kwargs)
