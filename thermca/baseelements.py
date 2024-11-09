"""Base model elements"""

from __future__ import annotations
from dataclasses import dataclass

from numpy import ndarray


class ModelElement:
    """Base for all simulation model elements."""

    def __init__(self):
        from thermca.model import Model  # Avoid circular dependency

        Model._append(self)
        self.model = Model.reference_stack[-1]  # still needed?


@dataclass
class NetDat:
    """Element network data"""

    posn: ndarray
    dock_lctn: ndarray
    areas: ndarray | None = None


def check_name(name):
    if name is not None and not name.isidentifier():
        raise Exception(
            "Names must only contain the alphanumeric letters"
            ' "a" to "z", "A" to "Z", "0" to "9" and "_". '
            "The name cannot start with a number."
        )
