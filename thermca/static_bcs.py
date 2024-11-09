from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Protocol

from numpy import array, int64


class BCSurf(Protocol):
    name: str


@dataclass
class TempBC:
    """Temperature boundary condition (Dirichlet)

    Args:
        surf: Coupling surface
        temp: Fixed surface temperature
    """

    surf: BCSurf
    temp: float


@dataclass
class HeatBC:
    """Heat flow boundary condition (Neumann)

    Args:
        surf: Coupling surface
        heat: Heat flow (power)
    """

    surf: BCSurf
    heat: float


@dataclass
class FluxBC:
    """Heat flux boundary condition (Neumann)

    Args:
        surf: Coupling surface
        flux: Heat flux as heat per area
    """

    surf: BCSurf
    flux: float  # Heat flux input (heat flow per area)


@dataclass
class FilmBC:  # Robin boundary condition
    """Film coefficient boundary condition (Robin)

    Args:
        surf: Coupling surface
        film: Film or heat transfer coefficient
        env_temp: Temperature of surface environment
    """

    surf: BCSurf
    film: float
    env_temp: float


BCs = Union[TempBC, HeatBC, FluxBC, FilmBC]


def flatten_bcs(bound_condns, surf_areas, surf_names):
    bound_condns = [
        HeatBC(bc.surf, bc.flux * surf_areas[surf_names.index(bc.surf.name)])
        if isinstance(bc, FluxBC)
        else bc
        for bc in bound_condns
    ]
    temp_bcs = [bc for bc in bound_condns if isinstance(bc, TempBC)]
    bc_temp_idxs = array(
        [surf_names.index(bc.surf.name) for bc in temp_bcs], dtype=int64
    )
    bc_temps = array([bc.temp for bc in temp_bcs])
    heat_bcs = [bc for bc in bound_condns if isinstance(bc, HeatBC)]
    bc_heat_idxs = array(
        [surf_names.index(bc.surf.name) for bc in heat_bcs], dtype=int64
    )
    bc_heats = array([bc.heat for bc in heat_bcs])
    film_bcs = [bc for bc in bound_condns if isinstance(bc, FilmBC)]
    bc_film_idxs = array(
        [surf_names.index(bc.surf.name) for bc in film_bcs], dtype=int64
    )
    bc_films = array([bc.film for bc in film_bcs])
    bc_film_temps = array([bc.env_temp for bc in film_bcs])

    return (
        bc_temp_idxs,
        bc_temps,
        bc_heat_idxs,
        bc_heats,
        bc_film_idxs,
        bc_films,
        bc_film_temps,
    )
