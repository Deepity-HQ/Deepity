"""Turns a BuildConfig into the actual `cmake` command lines."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .config import BuildConfig


def find_generator() -> tuple[str | None, str]:
    """Returns (ninja_path_or_None, display_name)."""
    ninja = shutil.which("ninja")
    return ninja, "Ninja" if ninja else "CMake"


def configure_command(config: BuildConfig, ninja: str | None, pgo_phase: str | None = None) -> list[str]:
    """pgo_phase: None (no PGO), "GENERATE", or "USE" -- distinguishes
    which pass of the two-pass PGO workflow this configure call is for.
    A single config.pgo boolean can't express this on its own, since
    both passes share the same BuildConfig."""
    profile = config.profile

    cmd = [
        "cmake",
        "-B",
        str(config.build_dir),
        f"-DCMAKE_BUILD_TYPE={config.build_type}",
        f"-DDEEPITY_BUILD_TESTS={'ON' if config.build_tests else 'OFF'}",
        f"-DDEEPITY_ENABLE_CUDA={'ON' if config.cuda else 'OFF'}",
        f"-DDEEPITY_USE_MKL={'ON' if config.blas == 'MKL' else 'OFF'}",
        f"-DDEEPITY_BUILD_PYTHON_BINDINGS={'ON' if config.python_bindings else 'OFF'}",
        f"-DDEEPITY_ARCH_FLAGS={profile.unix_flags}",
    ]

    if pgo_phase in ("GENERATE", "USE"):
        cmd.append(f"-DDEEPITY_PGO_MODE={pgo_phase}")
        cmd.append(f"-DDEEPITY_PGO_DATA_DIR={config.pgo_data_dir}")

    if profile.msvc_flags:
        cmd.append(f"-DDEEPITY_MSVC_ARCH_FLAGS={profile.msvc_flags}")

    if sys.platform == "win32":
        if ninja:
            cmd.extend(["-G", "Ninja"])

            # CC being unset doesn't mean clang isn't in play -- CMake can
            # auto-detect and pick it up on its own (as this project's own
            # builds have shown), so check what's actually on PATH rather
            # than an environment variable that may never have been set.
            clang_path = shutil.which("clang")
            cc_env = os.environ.get("CC", "").lower()
            using_clang = clang_path is not None or "clang" in cc_env

            if using_clang:
                # Locate libomp.lib next to the clang.exe on PATH
                omp_lib = None
                if clang_path:
                    llvm_lib_dir = Path(clang_path).parent.parent / "lib"
                    candidate = llvm_lib_dir / "libomp.lib"
                    if candidate.is_file():
                        omp_lib = candidate.as_posix()

                cmd.extend([
                    f"-DCMAKE_C_COMPILER=clang",
                    f"-DCMAKE_CXX_COMPILER=clang++",
                    "-DOpenMP_C_FLAGS=-fopenmp",
                    "-DOpenMP_CXX_FLAGS=-fopenmp",
                    "-DOpenMP_C_LIB_NAMES=omp",
                    "-DOpenMP_CXX_LIB_NAMES=omp",
                ])
                if omp_lib:
                    cmd.append(f"-DOpenMP_omp_LIBRARY={omp_lib}")
    return cmd


def build_command(config: BuildConfig) -> list[str]:
    return [
        "cmake",
        "--build",
        str(config.build_dir),
        "-j",
        str(config.jobs),
        "--config",
        config.build_type,
    ]


def test_command(config: BuildConfig) -> list[str]:
    return [
        "ctest",
        "--test-dir",
        str(config.build_dir),
        "--build-config",
        config.build_type,
        "--output-on-failure",
        "-j",
        str(config.jobs),
    ]