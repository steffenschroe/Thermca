from thermca.model import Model
from thermca.network import Network
from thermca.pointnodes import Node, StatNode, MatlNode, BoundNode
from thermca.links import CondLink, FilmLink, sum_films, FlowLink
from thermca.input import Input
from thermca.source import HeatSource, sum_heat, FluxSource
from thermca.resultdata import Result
from thermca.materials import Solid, Fluid, func_to_table
from thermca.plot.primitives import NAME, AXES, DEFAULT, BRIGHT
from thermca.plot.mesh import BODY_POINT_IDX, POINT_IDX
# fmt: off
from thermca.plot.model import (
    LINK, HEAT_SRC_ELEM, CAPY, HEAT, COND, CONN, TEMP, LUMP_NODE_IDX, COLOR_BAR,
    HEAT_SRC, VOL, AREA, SIM_SPAN, POINT_IN_TIME)
from thermca.lib import (
    free_conv, forced_conv, radiation, evaporation, combd_film, solids, fluids, bearing,
    contact, ball_screw, lp_parts
)
# fmt: on
from thermca.lib.combd_film import ASSISTING, OPPOSING, TRANSVERSE
from thermca.lib.ball_screw import SNGL_NUT, DBLE_NUT
from thermca._utils.func_tools import curry
from thermca.lpm.cube import Cube
from thermca.lpm.cyl import Cyl
from thermca.lpm.asm import Asm, Surf, ForceConts
from thermca.lpm.lp_part import LPPart
from thermca.fem.fe_part import FEPart, SFC_TO_CENTER_PLANE, SFC_TO_SFC
from thermca.mesh import Mesh, TRIANGLE, TETRA
from thermca.solver import RK23, RK45, LSODA


# https://stackoverflow.com/questions/15115514/how-do-i-document-classes-without-the-module-name/31594545#31594545
# fmt: off
__all__ = [
    'Model', 'Network', 'Node', 'StatNode', 'MatlNode', 'CondLink',
    'FilmLink', 'sum_films', 'FlowLink', 'Input', 'BoundNode', 'HeatSource', 'sum_heat',
    'FluxSource', 'Result', 'Solid', 'Fluid', 'func_to_table', 'solids',
    'fluids', 'bearing', 'contact', 'free_conv', 'forced_conv', 'radiation',
    'evaporation', 'combd_film', 'curry', 'ball_screw',
    'Cube', 'Cyl', 'Asm', 'Surf', 'ForceConts', 'LPPart',
    'lp_parts', 'FEPart', 'Mesh',
    'SFC_TO_CENTER_PLANE', 'SFC_TO_SFC', 'CAPY', 'HEAT', 'COND', 'LINK', 'TEMP',
    'LUMP_NODE_IDX', 'POINT_IDX', 'BODY_POINT_IDX', 'COLOR_BAR', 'HEAT_SRC',
    'HEAT_SRC_ELEM', 'NAME', 'AXES', 'CONN', 'VOL', 'AREA', 'DEFAULT', 'BRIGHT',
    'SIM_SPAN', 'POINT_IN_TIME', 'ASSISTING', 'OPPOSING', 'TRANSVERSE', 'SNGL_NUT',
    'DBLE_NUT', 'RK23', 'RK45', 'LSODA',
]
# fmt: on
