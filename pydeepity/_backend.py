try:
        # Absolute form, not `from . import pydeepity as dy`: pyright can't
        # resolve a relative import of a submodule that shares its name
        # with the enclosing package (no stub exists for the compiled
        # extension either way, but the relative form fails to resolve
        # at all, which cascades into "dy is unknown import symbol"
        # everywhere dy gets re-exported, outside this package's own
        # reportAttributeAccessIssue-suppressed executionEnvironment).
        import pydeepity.pydeepity as dy
except ImportError as e:
        raise ImportError(
                        "Could not load the compiled Deepity C++ backend.\nEnsure the package was installed correctly or compiled for your architecture."
                            ) from e
