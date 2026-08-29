try:
    from . import pydeepity as dy
except ImportError as e:
    raise ImportError(
        "Could not load the compiled Deepity C++ backend.\nEnsure the package was installed correctly or compiled for your architecture."
    ) from e
