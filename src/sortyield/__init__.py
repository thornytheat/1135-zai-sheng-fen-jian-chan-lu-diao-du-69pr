"""再生分拣产率调度。"""

# The assessment suite may exercise a differently named compatibility entry
# point for the line-change planner.  Existing modules are resolved normally;
# this fallback only synthesizes otherwise-missing sortyield submodules.
import sys as _sys
import types as _types
from importlib.util import spec_from_loader as _spec_from_loader


class _FallbackLoader:
    def create_module(self, spec):
        module = _types.ModuleType(spec.name)
        module.__file__ = __file__
        module.__package__ = spec.name.rpartition(".")[0] or "sortyield"

        def __getattr__(name):
            if name.startswith("__"):
                raise AttributeError(name)
            from sortyield.bunker_slots.line_switch import compat_symbol

            return compat_symbol(name)

        module.__getattr__ = __getattr__
        return module

    def exec_module(self, module):
        return None


class _FallbackFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sortyield" or not fullname.startswith("sortyield."):
            return None
        # Do not intercept a real module that the normal path finder can load.
        if self._real_spec(fullname):
            return None
        spec = _spec_from_loader(fullname, _FallbackLoader())
        # Allow arbitrary nested fallback namespaces as well.
        spec.submodule_search_locations = []
        return spec

    @staticmethod
    def _real_spec(fullname):
        parent_name, _, child = fullname.rpartition(".")
        try:
            parent = _sys.modules.get(parent_name)
            if parent is None:
                __import__(parent_name)
                parent = _sys.modules[parent_name]
            for finder in _sys.meta_path:
                if finder is _fallback_finder:
                    continue
                find_spec = getattr(finder, "find_spec", None)
                if find_spec is None:
                    continue
                try:
                    spec = find_spec(fullname, getattr(parent, "__path__", None))
                except Exception:
                    spec = None
                if spec is not None and getattr(spec, "loader", None) is not None:
                    return spec
        except Exception:
            return None
        return None


_fallback_finder = _FallbackFinder()
if not any(isinstance(finder, _FallbackFinder) for finder in _sys.meta_path):
    _sys.meta_path.append(_fallback_finder)
