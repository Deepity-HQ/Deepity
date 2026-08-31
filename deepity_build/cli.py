from __future__ import annotations

import argparse
import multiprocessing
import shutil
import sys
import os
import stat
import time
from pathlib import Path

from .cmake_runner import (
    build_command,
    configure_command,
    find_generator,
    test_command,
)
from .config import ARCH_PROFILES, DEFAULT_ARCH_PROFILE, BuildConfig
from .git_info import get_git_info
from .process import run_captured_command, run_streaming_command
from .reporting import make_reporter

def _rmtree_onexc(func, path, exc_info):
    """shutil.rmtree fails on Windows for read-only files (e.g. inside a
    FetchContent'd repo's .git/objects/pack/) because it never clears the
    read-only attribute before unlinking. Clear it and retry once."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deepity Cross-Platform Build & Test Runner"
    )
    parser.add_argument(
        "build_type",
        nargs="?",
        default="Release",
        choices=["Release", "Debug"],
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel build jobs",
    )
    parser.add_argument(
        "blas",
        nargs="?",
        default="OpenBLAS",
        choices=["OpenBLAS", "MKL"],
    ) # Considering BLIS, ROCm, etc.

    arch_group = parser.add_mutually_exclusive_group()
    arch_group.add_argument(
        "--native",
        action="store_const",
        dest="arch_profile",
        const="native",
        help=(
            "Compile with -march=native. Fastest, but the binary is only safe "
            "to run on THIS machine. Never use for anything you plan to hand "
            "to someone else."
        ),
    )
    arch_group.add_argument(
        "--fast",
        action="store_const",
        dest="arch_profile",
        const="fast",
        help=(
            "Compile for an AVX2/FMA baseline (x86-64-v3). Portable across "
            "most machines from the last ~10 years. Default for local builds."
        ),
    )
    arch_group.add_argument(
        "--distributed",
        action="store_const",
        dest="arch_profile",
        const="distributed",
        help=(
            "Compile for a maximally portable SSE4.2 baseline (x86-64-v2). "
            "Use this for anything you're going to package and ship, e.g. "
            "wheels built for PyPI."
        ),
    )
    parser.set_defaults(arch_profile=None)

    parser.add_argument(
        "--cuda",
        dest="cuda",
        action="store_true",
        default=None,
        help="Force CUDA support ON (requires CUDAToolkit to be found).",
    )
    parser.add_argument(
        "--no-cuda",
        dest="cuda",
        action="store_false",
        help="Force CUDA support OFF. Recommended together with --distributed.",
    )

    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip building and running the DeepityTests target",
    )
    parser.add_argument(
        "--no-python-bindings",
        action="store_true",
        help="Skip building the pydeepity extension module",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the build directory before building",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full build output",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print available --native/--fast/--distributed profiles and exit",
    )
    parser.add_argument(
        "--pgo",
        action="store_true",
        help=(
            "Profile-Guided Optimization: builds an instrumented binary, "
            "runs a short representative workload against it to collect "
            "real branch/call-frequency data, then rebuilds using that "
            "data. GCC/Clang only. Roughly doubles total build time (two "
            "full compiles) but the workload run itself is short -- a few "
            "dozen batches, not a full training run."
        ),
    )

    args = parser.parse_args(argv)

    if args.list_profiles:
        for profile in ARCH_PROFILES.values():
            marker = " (default)" if profile.name == DEFAULT_ARCH_PROFILE else ""
            print(f"{profile.name}{marker}")
            print(f"  {profile.description}")
            print(f"  unix:  {profile.unix_flags}")
            print(f"  msvc:  {profile.msvc_flags or '(compiler default)'}")
            print()
        sys.exit(0)

    if args.arch_profile is None:
        args.arch_profile = DEFAULT_ARCH_PROFILE

    if args.cuda is None:
        # Distributed builds default to CPU-only, since a CUDA-linked binary
        # isn't portable either. Everything else keeps the old default (ON).
        args.cuda = args.arch_profile != "distributed"

    return args


def build_config_from_args(args: argparse.Namespace) -> BuildConfig:
    return BuildConfig(
        build_type=args.build_type,
        jobs=args.jobs,
        arch_profile=args.arch_profile,
        cuda=args.cuda,
        blas=args.blas,
        build_tests=not args.no_tests,
        run_tests=not args.no_tests,
        python_bindings=not args.no_python_bindings,
        clean=args.clean,
        verbose=args.verbose,
        pgo=args.pgo,
    )


def _run_configure_build_test(
    config: BuildConfig,
    ninja: str | None,
    generator: str,
    log_output,
    pgo_phase: str | None = None,
    run_tests_override: bool | None = None,
) -> None:
    """One full configure+build[+test] pass, reported through its own
    reporter session. Used directly for the normal (non-PGO) path, and
    called twice (once per phase) for the PGO workflow -- each phase gets
    a clean reporter lifecycle rather than sharing one across two builds.
    """
    git = get_git_info()
    reporter = make_reporter(config, git, generator)
    run_tests = config.run_tests if run_tests_override is None else run_tests_override

    with reporter:
        if not config.deps_dir.is_dir():
            reporter.clean_reconfigure()
            if config.build_dir.exists():
                shutil.rmtree(config.build_dir, onexc=_rmtree_onexc)

        # ────────────────────────────────────────────────────────────
        # Configure
        # ────────────────────────────────────────────────────────────
        configure_time: float | None = None

        if pgo_phase is not None or not config.cache_file.is_file():
            # PGO phases always reconfigure -- GENERATE and USE need
            # genuinely different compiler flags, so a cached
            # configuration from a previous phase can't be reused.
            cmd = configure_command(config, ninja, pgo_phase=pgo_phase)

            reporter.configure_started()
            start = time.perf_counter()

            return_code, config_output = run_captured_command(cmd)
            log_output("CMake Configuration", config_output)

            configure_time = time.perf_counter() - start

            if return_code != 0:
                reporter.configure_failed(config_output)
                sys.exit(return_code)

            reporter.configure_complete(configure_time)
        else:
            reporter.configure_cached()

        # ────────────────────────────────────────────────────────────
        # Build
        # ────────────────────────────────────────────────────────────
        reporter.build_started()
        start = time.perf_counter()

        return_code, build_output = run_streaming_command(
            build_command(config),
            on_line=reporter.build_line,
        )
        log_output("Compilation", build_output)

        build_time = time.perf_counter() - start

        if return_code != 0:
            reporter.build_failed(build_output)
            sys.exit(return_code)

        reporter.build_complete(build_time)

        if not run_tests:
            reporter.success(configure_time, build_time, 0.0)
            return

        # ────────────────────────────────────────────────────────────
        # Run tests via CTest
        # ────────────────────────────────────────────────────────────
        reporter.tests_started()
        start = time.perf_counter()

        cmd = test_command(config)

        return_code, test_output = run_streaming_command(
            cmd,
            on_line=reporter.test_line,
        )
        log_output("Tests", test_output)

        test_time = time.perf_counter() - start

        if return_code != 0:
            reporter.tests_failed(test_output)
            sys.exit(return_code)

        reporter.tests_complete(test_time)
        reporter.success(configure_time, build_time, test_time)


def _run_pgo_workload(config: BuildConfig) -> None:
    """Runs the short, dedicated profile-collection workload against the
    just-built instrumented binary. Deliberately NOT the full mnist.py
    training run -- PGO only needs to see which code paths are hot, and
    the settling loop's structure repeats identically on batch 1 and
    batch 234, so a few dozen batches already captures the same
    branch/call-frequency information a full run would, at a fraction of
    the cost. This cost is paid on every --pgo build, not once."""
    import subprocess

    print("\n--- PGO: running representative workload to collect profile data ---")
    result = subprocess.run(
        [sys.executable, "pgo_workload.py"],
        cwd=Path(__file__).resolve().parent.parent,
    )
    if result.returncode != 0:
        print("PGO workload run failed -- aborting before the optimized rebuild.")
        sys.exit(result.returncode)
    print("--- PGO: profile data collected, proceeding to optimized rebuild ---\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config_from_args(args)

    if config.clean and config.build_dir.exists():
        shutil.rmtree(config.build_dir, onexc=_rmtree_onexc)

    ninja, generator = find_generator()

    # ────────────────────────────────────────────────────────────────────
    # Setup logging
    # ────────────────────────────────────────────────────────────────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "build.log"

    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"--- Deepity Build Log ({config.build_type}, arch={config.arch_profile}) ---\n\n")

    def log_output(phase: str, output: str) -> None:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"=== {phase} ===\n")
            f.write(output)
            f.write("\n\n")

    if not config.pgo:
        _run_configure_build_test(config, ninja, generator, log_output)
        return

    # ────────────────────────────────────────────────────────────────────
    # PGO: two full passes. GENERATE builds an instrumented binary (tests
    # skipped -- irrelevant for a throwaway instrumented build); the
    # workload run collects real profile data; USE rebuilds with it.
    # ────────────────────────────────────────────────────────────────────
    if config.pgo_data_dir.exists():
        shutil.rmtree(config.pgo_data_dir)
    config.pgo_data_dir.mkdir(parents=True, exist_ok=True)

    print("=== PGO pass 1/2: instrumented (GENERATE) build ===")
    _run_configure_build_test(config, ninja, generator, log_output, pgo_phase="GENERATE", run_tests_override=False)

    _run_pgo_workload(config)

    print("=== PGO pass 2/2: optimized (USE) build ===")
    _run_configure_build_test(config, ninja, generator, log_output, pgo_phase="USE")


if __name__ == "__main__":
    main()
