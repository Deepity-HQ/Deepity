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


def configure_command(config: BuildConfig, ninja: str | None) -> list[str]:
    profile = config.profile

    cmd = [
        "cmake",
        "-B",
        str(config.build_dir),
        f"-DCMAKE_BUILD_TYPE={config.build_type}",
        f"-DDEEPITY_BUILD_TESTS={'ON' if config.build_tests else 'OFF'}",
        f"-DDEEPITY_ENABLE_CUDA={'ON' if config.cuda else 'OFF'}",
        f"-DDEEPITY_BUILD_PYTHON_BINDINGS={'ON' if config.python_bindings else 'OFF'}",
        f"-DDEEPITY_ARCH_FLAGS={profile.unix_flags}",
    ]

    if profile.msvc_flags:
        cmd.append(f"-DDEEPITY_MSVC_ARCH_FLAGS={profile.msvc_flags}")

    if sys.platform == "win32":
        cmd.extend(["-A", "x64"])
        vcpkg_root = os.environ.get("VCPKG_ROOT")
        if vcpkg_root:
            toolchain = Path(vcpkg_root) / "scripts/buildsystems/vcpkg.cmake"
            if toolchain.is_file():
                cmd.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain.as_posix()}")
    else:
        if ninja:
            cmd.extend(["-G", "Ninja"])

        vcpkg_root = os.environ.get("VCPKG_ROOT")
        if vcpkg_root:
            cmd.append(f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_root}/scripts/buildsystems/vcpkg.cmake")

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


def test_executable_name() -> str:
    return "DeepityTests.exe" if os.name == "nt" else "DeepityTests"


def test_executable_candidates(config: BuildConfig) -> list[Path]:
    exe_name = test_executable_name()

    return [
        config.build_dir / "bin" / exe_name,
        config.build_dir / "bin" / config.build_type / exe_name,
        config.build_dir / exe_name,
        config.build_dir / config.build_type / exe_name,
    ]


def find_test_executable(config: BuildConfig) -> Path | None:
    return next((path for path in test_executable_candidates(config) if path.is_file()), None)