from functools import wraps
import pickle
from pathlib import Path
from typing import Callable

import joblib

from numpy import uint8, ndarray


# make curry as simple as possible for future optimisations with e.g. numba
def curry(func: Callable) -> Callable:
    """Returns a given function and overwrites its keyword arguments
    if necessary.

    It is intended to parametrise library functions for heat sources,
    conductance's and film coefficients. These functions have positional
    and keyword arguments. The latter are mainly used to specify some
    additional model parameters. The curry function is used to change
    the keywords default arguments to desired values.
    The benefit is that the original function can stay in a library
    and does not need to be changed. Also, this helps with equal behaviour
    of library functions. They should always return a modeling compatible
    function.

    Example:
        >>> from thermca import *
        ...
        ... # a function for convection film coefficients including free and forced convection
        >>> @curry
        >>> def conv_film(surf_temp, air_temp, fluid=None, air_vel=1.):
        >>>     return 2.2*abs(surf_temp - air_temp)**.33 + 10.45 - air_vel + 10*air_vel**.5
        ...
        >>> with Model() as model:
        ...     butter = lp_parts.block(
        ...         matl=solids.salted_butter,
        ...         wid=.1, hgt=.035, dep=.075,
        ...         init_temp=10.)
        ...     air = BoundNode(temp=20.)
        ...     # Generate the convection film function for air velocities of 5 m/s
        ...     # and 'overwrite' the default value for the argument air_vel.
        ...     # The generated function supports the calling convention
        ...     # of FilmLink model elements.
        ...     FilmLink.multi(
        ...         butter.surf,
        ...         air,
        ...         films=conv_film(air_vel=5)
        ...     )
        ...
        ... # The function can also be configured and called directly with the
        ... # positional arguments like this:
        ... conv_film(air_vel=5)(surf_temp=40., air_temp=20.)
    """
    @wraps(func)  # transfers the docstring and signature from original function
    def curried(**kwargs0):
        if not kwargs0:
            return lambda *args, **kwargs: func(*args, **kwargs)
        return lambda *args, **kwargs1: func(*args, **{**kwargs0, **kwargs1})
    '''
    # the following dont preserves the signature, which destroys the documentation 
    # of the curried functions by not recognising the function arguments anymore
    doc = getattr(func, '__doc__')  # __doc__ is None or the given docstring
    doc = '' if doc is None else doc
    curried_doc = ("    Returns the function described below\n"
                   "    and overwrites its keyword arguments if necessary.\n\n")
    curried.__doc__ = curried_doc + doc
    curried.__name__ = getattr(func, '__name__', '<curry>')
    curried.__module__ = getattr(func, '__module__', None)
    curried.__qualname__ = getattr(func, '__qualname__', None)
    '''
    return curried


def disk_cache(func):
    """Cash result of costly function on disk
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        hash_str = joblib.hash(args)  # Unique hash for function args
        file_name = Path(".thermca_cache/" + func.__name__ + "_" + hash_str + ".pickle")
        file_name.parent.mkdir(exist_ok=True, parents=True)

        try:
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            result = func(*args)
            with open(file_name, 'wb') as f:
                pickle.dump(result, f)
            return result

    return wrapper



