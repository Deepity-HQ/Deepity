"""
Reporting package.

main() only ever talks to the Reporter interface, never to rich or
plain-text internals directly. Each concrete reporter imports rich symbols
locally, right where they're used, so importing this package never requires
rich to be installed, and static type checkers never see a "possibly
unbound" import.
"""

from __future__ import annotations

import importlib.util

from ..config import BuildConfig
from ..git_info import GitInfo
from .base import Reporter


def is_library_installed(library_name: str) -> bool:
    """Check if a python library is installed without importing it."""
    if not isinstance(library_name, str) or not library_name.strip():
        raise ValueError("Library name must be a non-empty string.")
    return importlib.util.find_spec(library_name) is not None


def rich_available() -> bool:
    try:
        return is_library_installed("rich")
    except ValueError:
        return False


def make_reporter(config: BuildConfig, git: GitInfo, generator: str) -> Reporter:
    """
    Construct the best available reporter and announce the choice.

    Kept here (rather than in cli.py) so the "which reporter, and why" story
    lives next to the reporters themselves.
    """
    if rich_available():
        from .rich_reporter import RichReporter

        print("✅ 'rich' is installed. Using the interactive dashboard.")
        return RichReporter(config, git, generator)

    from .plain_reporter import PlainReporter

    print("❌ 'rich' is NOT installed. Falling back to plain-text output.")
    return PlainReporter(config, git, generator)


__all__ = ["Reporter", "make_reporter", "is_library_installed", "rich_available"]