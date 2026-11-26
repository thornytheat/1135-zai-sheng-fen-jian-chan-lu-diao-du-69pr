import importlib
import importlib.util

from . import contracts, engine, service  # noqa: F401
from .line_switch import *  # noqa: F401,F403
from . import line_switch as _line_switch


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    # Resolve real sibling submodules before the broad symbol fallback.
    try:
        spec = importlib.util.find_spec(f"{__name__}.{name}")
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    return _line_switch.compat_symbol(name)
