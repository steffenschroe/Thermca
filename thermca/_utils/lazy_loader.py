import sys
import importlib.util


# https://stackoverflow.com/questions/42703908/how-do-i-use-importlib-lazyloader
def load(fullname):
    try:
        return sys.modules[fullname]
    except KeyError:
        spec = importlib.util.find_spec(fullname)
        module = importlib.util.module_from_spec(spec)
        loader = importlib.util.LazyLoader(spec.loader)  # .factory
        # Make module with proper locking and get it inserted into sys.modules.
        loader.exec_module(module)
        return module
