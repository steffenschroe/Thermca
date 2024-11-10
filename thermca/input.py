from bisect import bisect_right
from dataclasses import dataclass, field

from numpy import ndarray, float64, array, asarray, nan

from thermca.baseelements import ModelElement

@dataclass
class Input(ModelElement):
    """Time changing input variables

    The input variables are boundary conditions for the simulation. They
    can take any course over time. The course is specified by a series
    of timestamps and corresponding values.

    At the start of eachsimulation time step, the last available value
    for the given simulation time is searched and saved in the `value`
    property. It can then be used by multiple model elements or library
    functions. The expensive search is only carried out once in this
    way.

    During model building, time dependent properties can be parametrised
    with the `value` attribute of the `Input`. This `value` is a mutable
    numpy array scalar. Some attributes of library functions can receive
    such a time chanching varable. Further, the `BoundNode` element,
    whitch defines temperature boundary conditions, can receive a time
    dependent function. The `get_value` method of the `Input` can be
    passed to it.

    Args:
        time_table: The table contains time values in the first column
            and corresponding values of the input variable in the second
            column. The time values must be increasing.

    Attributes:
        value: Contains the last measured value for the current
            simulation time.

    Examples:

        Input for boundary temperatures:

        >>> from thermca import *
        >>> with Model() as model:
        >>>     # Table of time steps and corresponding temperatures
        >>>     # First column time, second colum temperatures
        >>>     env_temp = Input(
        >>>         [[0, 19],
        >>>          [20, 23],
        >>>          [40, 27]],
        >>>         name='environment temperatures'
        >>>     )
        >>>     environment = BoundNode(
        >>>         temp=env_temp.get_value,  # Given as function
        >>>         name='environment',
        >>>     )

        Input for convection film coefficient on a rotating disc:

        >>> with Asm() as disc_asm:
        >>>     cyl = Cyl(
        >>>         inner_rad=.011,
        >>>         outer_rad=.075,
        >>>         lgth=.0015,
        >>>         rad_div=16,
        >>>      )
        >>>      Surf(
        >>>         name='circ_faces',
        >>>         faces=[cyl.face.base, cyl.face.end],
        >>>      )
        >>>      Surf(
        >>>         name='outer',
        >>>         faces=[cyl.face.outer],
        >>>      )
        >>> with model:
        >>>     disc = LPPart(
        >>>         asm=disc_asm,
        >>>         matl=solids.steel,
        >>>         init_temp=20.,
        >>>         name='disc',
        >>>     )
        >>>     rpm = 2*3.1415/60
        >>>     # Table of time steps and corresponding speeds
        >>>     rot_freq = Input(
        >>>         [[0, 6000*rpm],
        >>>         [20, 3000*rpm]]
        >>>     )
        >>>     FilmLink(
        >>>         disc.surf.circ_faces,
        >>>         environment,
        >>>         film=forced_conv.rot_disc_in_air(  # Function from included library
        >>>             rot_freq=rot_freq.value,  # Given as time dependent mutable value
        >>>             rad=cyl.outer_rad,
        >>>         )
        >>>     )

        Input for heat source:

        >>> with model:
        >>>     # Table of time steps and corresponding heat
        >>>     cutting_heat = Input(
        >>>         [[0, 40],
        >>>          [20, 80]]
        >>>     )
        >>>     HeatSource(
        >>>         disc.surf.outer,
        >>>         heat=lambda temp: cutting_heat.value,
        >>>     )

        Further information in the cutting_disc example
    """

    time_table: ndarray | list
    name: str = ""
    _old_time: float = field(repr=False, default=float("-inf"))
    _old_idx: int = field(repr=False, default=0)
    _value: ndarray = field(repr=False, init=False)
    _time_table: ndarray = field(repr=False, init=False)

    def __post_init__(self):
        super().__init__()
        self._value = array(nan)
        self._time_table = asarray(self.time_table, dtype=float64)
        if not self._time_table.ndim == 2:
            raise TypeError("2d sequence in regular array shape expected.")

    @property
    def value(self):
        return self._value

    def get_value(self):
        """Method for use with temperature boundary conditions"""
        return self._value

    def _set_time(self, time):
        """Called by the simulation routine to set the time dependent
        value
        """
        if time > self._old_time:
            # Search rest of array only
            lo = self._old_idx
        else:
            lo = 0
        self._old_time = time
        self._old_idx = bisect_right(self._time_table[:, 0], time, lo=lo) - 1
        self.value.real = self._time_table[self._old_idx, 1]

