"""
Build configuration: everything that determines *what* CMake is asked to
build. No process handling and no reporting logic lives here — this module
should be safe to import and unit-test on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════
# Architecture profiles
#
# "native" is fast but only safe on the exact machine that built it — the
# binary can hit an illegal-instruction crash on any other CPU. "fast" and
# "distributed" are portable baselines you can hand to other people; the
# difference is how old a CPU they still cover.
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ArchProfile:
    name: str
    unix_flags: str
    msvc_flags: str
    description: str


ARCH_PROFILES: dict[str, ArchProfile] = {
    "native": ArchProfile(
        name="native",
        unix_flags="-march=native",
        msvc_flags="/arch:AVX2",  # MSVC has no true "native"; this is its closest knob
        description=(
            "Tuned for THIS machine's CPU. Fastest option, but the resulting "
            "binary can crash with an illegal instruction on any other machine. "
            "Local development only — never distribute a native build."
        ),
    ),
    "fast": ArchProfile(
        name="fast",
        unix_flags="-march=x86-64-v3 -mtune=generic",
        msvc_flags="/arch:AVX2",
        description=(
            "AVX2 + FMA baseline (x86-64-v3). Covers the large majority of "
            "desktops/servers from the last decade. Good default for local "
            "builds when you don't need maximum portability."
        ),
    ),
    "distributed": ArchProfile(
        name="distributed",
        unix_flags="-march=x86-64-v2 -mtune=generic",
        msvc_flags="",  # MSVC's default codegen is already an SSE2-era baseline
        description=(
            "SSE4.2 baseline (x86-64-v2), maximally portable. Use this for any "
            "build you intend to ship to other people (wheels, releases, CI "
            "artifacts)."
        ),
    ),
}

DEFAULT_ARCH_PROFILE = "fast"


@dataclass(frozen=True)
class BuildConfig:
    build_type: str
    jobs: int
    arch_profile: str
    cuda: bool
    blas: str
    build_tests: bool
    run_tests: bool
    python_bindings: bool
    clean: bool
    verbose: bool
    build_root: Path = Path("build")

    @property
    def profile(self) -> ArchProfile:
        return ARCH_PROFILES[self.arch_profile]

    @property
    def build_dir(self) -> Path:
        return self.build_root / self.build_type

    @property
    def deps_dir(self) -> Path:
        return self.build_dir / "_deps"

    @property
    def cache_file(self) -> Path:
        return self.build_dir / "CMakeCache.txt"
