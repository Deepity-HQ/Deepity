"""Loads the compiled `deepity` nanobind extension as `dy`."""

from importlib import import_module
from types import ModuleType

try:
    dy: ModuleType = import_module(".deepity", __package__)
except ImportError as e:
    raise ImportError(
        "Could not load the compiled Deepity C++ backend.\nEnsure the package was installed correctly or compiled for your architecture."
    ) from e
