import argparse
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
import importlib.util


def is_library_installed(library_name: str) -> bool:
    """
    Check if a python library is installed without importing it.

    Args:
        library_name (str): The name of the library to check
    Returns:
        bool: True if installed, False otherwise
    """
    if not isinstance(library_name, str) or not library_name.strip():
        raise ValueError("Library name must be a non-empty string.")
    return importlib.util.find_spec(library_name) is not None


try:
    RICH_AVAILABLE = is_library_installed("rich")
except ValueError as e:
    print(f"Error: {e}")
    RICH_AVAILABLE = False

if RICH_AVAILABLE:
    print("✅ 'rich' is installed. Using the interactive dashboard.")
else:
    print("❌ 'rich' is NOT installed. Falling back to plain-text output.")


BUILD_PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(.*)")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m {seconds:.0f}s"

    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m"


def command_string(cmd: list[str]) -> str:
    return " ".join(cmd)


# ══════════════════════════════════════════════════════════════════════════
# Reporter interface
#
# main() only ever talks to this interface, never to rich or plain-text
# internals directly. That keeps main() free of "if RICH_AVAILABLE" checks
# (aside from picking which implementation to construct) and means neither
# implementation needs module-level conditional imports of rich symbols -
# each one imports what it needs locally, right where it's used, which is
# also what keeps static type checkers (Pylance/pyright) happy: nothing is
# "possibly unbound" because every rich import lives inside the exact
# function that uses it, in a class that is only ever instantiated when
# rich is actually installed.
# ══════════════════════════════════════════════════════════════════════════
class Reporter(ABC):
    @abstractmethod
    def __enter__(self) -> "Reporter": ...

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None: ...

    @abstractmethod
    def clean_reconfigure(self) -> None: ...

    @abstractmethod
    def configure_started(self) -> None: ...

    @abstractmethod
    def configure_cached(self) -> None: ...

    @abstractmethod
    def configure_complete(self, duration: float) -> None: ...

    @abstractmethod
    def configure_failed(self, output: str) -> None: ...

    @abstractmethod
    def build_started(self) -> None: ...

    @abstractmethod
    def build_line(self, line: str) -> None: ...

    @abstractmethod
    def build_complete(self, duration: float) -> None: ...

    @abstractmethod
    def build_failed(self, output: str) -> None: ...

    @abstractmethod
    def tests_missing(self, exe_name: str, paths: list[Path]) -> None: ...

    @abstractmethod
    def tests_started(self) -> None: ...

    @abstractmethod
    def test_line(self, line: str) -> None: ...

    @abstractmethod
    def tests_complete(self, duration: float) -> None: ...

    @abstractmethod
    def tests_failed(self, output: str) -> None: ...

    @abstractmethod
    def success(
        self,
        configure_time: float | None,
        build_time: float,
        test_time: float,
    ) -> None: ...


# ══════════════════════════════════════════════════════════════════════════
# Rich-backed reporter (used only when 'rich' is available)
# ══════════════════════════════════════════════════════════════════════════
class RichReporter(Reporter):
    """Single mutable Rich Live dashboard for the entire build."""

    def __init__(self, build_type: str, generator: str, jobs: int) -> None:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self.build_type = build_type
        self.generator = generator
        self.jobs = jobs

        self.phase = "Starting..."
        self.configure_status = "[dim]waiting[/dim]"
        self.build_status = "[dim]waiting[/dim]"
        self.test_status = "[dim]waiting[/dim]"

        self.build_message = ""
        self.test_lines: deque[str] = deque(maxlen=8)

        self._configure_time_display = "—"
        self._build_time_display = "—"
        self._test_time_display = "—"
        self._total_time_display = "—"

        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        self.build_task = self.progress.add_task("Waiting", total=1, completed=0)

        self._live = None

    # -- context manager: owns the Live session --------------------------
    def __enter__(self) -> "RichReporter":
        from rich.live import Live

        self.console.clear()
        self.console.print()

        self._live = Live(
            self._render(),
            refresh_per_second=12,
            console=self.console,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)

    # -- rendering ---------------------------------------------------------
    def _header(self):
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(padding=(0, 2))
        table.add_column()
        table.add_column()
        table.add_column()
        table.add_column()

        table.add_row(
            "[bold cyan]Deepity Engine[/bold cyan]",
            f"[dim]Build[/dim] [bold]{self.build_type}[/bold]",
            f"[dim]Generator[/dim] [bold]{self.generator}[/bold]",
            f"[dim]Jobs[/dim] [bold]{self.jobs}[/bold]",
        )

        return Panel(table, border_style="cyan", padding=(0, 1))

    def _status(self):
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", width=12)
        table.add_column()

        table.add_row("Configure", self.configure_status)
        table.add_row("Build", self.build_status)
        table.add_row("Tests", self.test_status)

        return Panel(table, title="[bold]Build Status[/bold]", border_style="blue")

    def _activity(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        if self.phase == "Configuring":
            body = Group(
                Text("Configuring CMake...", style="bold yellow"),
                Text(""),
                Text.from_markup(
                    "[dim]CMake output is hidden while the build is running "
                    "and shown in full if configuration fails.[/dim]"
                ),
            )
            return Panel(
                body,
                title="[bold yellow]⚙  Configuration[/bold yellow]",
                border_style="yellow",
            )

        if self.phase == "Building":
            current = self.build_message or "Working..."
            return Panel(
                Group(self.progress, Text.from_markup(f"[dim]{current}[/dim]")),
                title="[bold blue]⚙ Compilation[/bold blue]",
                border_style="blue",
            )

        if self.phase == "Testing":
            body = (
                "\n".join(self.test_lines)
                if self.test_lines
                else "[dim]Waiting for test output...[/dim]"
            )
            return Panel(
                body,
                title="[bold magenta]▶ Tests[/bold magenta]",
                border_style="magenta",
            )

        if self.phase == "Success":
            body = Table.grid(padding=(0, 2))
            body.add_column(style="dim")
            body.add_column(style="bold")

            body.add_row("Build type", self.build_type)
            body.add_row("Generator", self.generator)
            body.add_row("Parallel jobs", str(self.jobs))
            body.add_row("Configuration", self._configure_time_display)
            body.add_row("Compilation", self._build_time_display)
            body.add_row("Tests", self._test_time_display)
            body.add_row("Total", self._total_time_display)

            return Panel(
                Group(
                    Text("✓  BUILD SUCCESSFUL", style="bold green", justify="center"),
                    Text(""),
                    body,
                ),
                title="[bold green]Deepity[/bold green]",
                border_style="green",
                padding=(1, 2),
            )

        return Panel("[dim]Waiting...[/dim]", title="[bold]Activity[/bold]", border_style="dim")

    def _render(self):
        from rich.console import Group

        return Group(self._header(), self._status(), self._activity())

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render(), refresh=True)

    def _print_failure(self, title: str, body: str) -> None:
        from rich.panel import Panel

        body = body.rstrip() or "No output"
        self.console.print()
        self.console.print(Panel(body, title=f"[bold red]{title}[/bold red]", border_style="red"))

    # -- Reporter interface -------------------------------------------------
    def clean_reconfigure(self) -> None:
        self.phase = "Configuring"
        self.configure_status = "[yellow]● clean reconfigure[/yellow]"
        self._refresh()

    def configure_started(self) -> None:
        self.phase = "Configuring"
        self.configure_status = "[yellow]● configuring[/yellow]"
        self._refresh()

    def configure_cached(self) -> None:
        self.configure_status = "[green]✓ cached[/green]"
        self._refresh()

    def configure_complete(self, duration: float) -> None:
        self.configure_status = (
            f"[bold green]✓ complete[/bold green] [dim]({format_duration(duration)})[/dim]"
        )
        self._refresh()

    def configure_failed(self, output: str) -> None:
        self._print_failure("✗ CMake Configuration Failed", output)

    def build_started(self) -> None:
        self.phase = "Building"
        self.build_status = "[yellow]● compiling[/yellow]"
        self.progress.update(self.build_task, total=1, completed=0, description="Compiling")
        self._refresh()

    def build_line(self, line: str) -> None:
        match = BUILD_PROGRESS_RE.search(line)

        if not match:
            stripped = line.strip()
            if stripped:
                self.build_message = stripped[-160:]
            self._refresh()
            return

        current = int(match.group(1))
        total = int(match.group(2))
        message = match.group(3).strip()

        self.build_message = message
        self.progress.update(self.build_task, total=total, completed=current)
        self._refresh()

    def build_complete(self, duration: float) -> None:
        self.build_status = (
            f"[bold green]✓ complete[/bold green] [dim]({format_duration(duration)})[/dim]"
        )
        self._refresh()

    def build_failed(self, output: str) -> None:
        self.build_status = "[bold red]✗ failed[/bold red]"
        self.phase = "Build failed"
        self._refresh()
        self._print_failure("✗ Build Failed", output)

    def tests_missing(self, exe_name: str, paths: list[Path]) -> None:
        self.test_status = "[bold red]✗ binary missing[/bold red]"
        self.phase = "Tests unavailable"
        self._refresh()
        self._print_failure(f"✗ Could Not Find {exe_name}", "\n".join(str(p) for p in paths))

    def tests_started(self) -> None:
        self.phase = "Testing"
        self.test_status = "[yellow]● running[/yellow]"
        self.test_lines.clear()
        self._refresh()

    def test_line(self, line: str) -> None:
        line = line.rstrip()
        if line:
            self.test_lines.append(line)
        self._refresh()

    def tests_complete(self, duration: float) -> None:
        self.test_status = (
            f"[bold green]✓ complete[/bold green] [dim]({format_duration(duration)})[/dim]"
        )
        self._refresh()

    def tests_failed(self, output: str) -> None:
        self.test_status = "[bold red]✗ failed[/bold red]"
        self.phase = "Tests failed"
        self._refresh()
        self._print_failure("✗ Test Suite Failed", output)

    def success(
        self,
        configure_time: float | None,
        build_time: float,
        test_time: float,
    ) -> None:
        self._configure_time_display = (
            format_duration(configure_time) if configure_time else "cached"
        )
        self._build_time_display = format_duration(build_time)
        self._test_time_display = format_duration(test_time)

        total = sum(value or 0 for value in (configure_time, build_time, test_time))
        self._total_time_display = format_duration(total)

        self.phase = "Success"
        self._refresh()


# ══════════════════════════════════════════════════════════════════════════
# Plain-text reporter (used when 'rich' is NOT available)
# ══════════════════════════════════════════════════════════════════════════
class PlainReporter(Reporter):
    """
    stdout-only mirror of RichReporter. Same public interface, no rich
    dependency at all: statuses are printed as they change, build progress
    is shown as an in-place percentage line, and test output streams
    straight through.
    """

    def __init__(self, build_type: str, generator: str, jobs: int) -> None:
        self.build_type = build_type
        self.generator = generator
        self.jobs = jobs

        self.phase = "Starting..."
        self.configure_status = "waiting"
        self.build_status = "waiting"
        self.test_status = "waiting"

        self._last_build_pct = -1
        self._build_line_open = False

    def __enter__(self) -> "PlainReporter":
        print(
            f"Deepity Engine — build={self.build_type} "
            f"generator={self.generator} jobs={self.jobs}"
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def _status_line(self) -> None:
        print(
            f"[{self.phase}] configure={self.configure_status} "
            f"build={self.build_status} tests={self.test_status}"
        )

    def _print_failure(self, title: str, body: str) -> None:
        body = body.rstrip() or "No output"
        print()
        print(f"---- {title} " + "-" * max(0, 60 - len(title)))
        print(body)
        print("-" * 70)

    def _close_build_line(self) -> None:
        if self._build_line_open:
            print()
            self._build_line_open = False

    # -- Reporter interface -------------------------------------------------
    def clean_reconfigure(self) -> None:
        self.phase = "Configuring"
        self.configure_status = "clean reconfigure"
        self._status_line()

    def configure_started(self) -> None:
        self.phase = "Configuring"
        self.configure_status = "configuring..."
        self._status_line()

    def configure_cached(self) -> None:
        self.configure_status = "cached"
        self._status_line()

    def configure_complete(self, duration: float) -> None:
        self.configure_status = f"complete ({format_duration(duration)})"
        self._status_line()

    def configure_failed(self, output: str) -> None:
        self._print_failure("✗ CMake Configuration Failed", output)

    def build_started(self) -> None:
        self.phase = "Building"
        self.build_status = "compiling..."
        self._status_line()

    def build_line(self, line: str) -> None:
        match = BUILD_PROGRESS_RE.search(line)

        if not match:
            return

        current = int(match.group(1))
        total = int(match.group(2))
        message = match.group(3).strip()

        pct = int((current / total) * 100) if total else 0

        # Avoid flooding the terminal: only print when percentage moves.
        if pct != self._last_build_pct:
            self._last_build_pct = pct
            print(f"\r  [{current}/{total}] {pct:3d}%  {message[:80]}", end="", flush=True)
            self._build_line_open = True

    def build_complete(self, duration: float) -> None:
        self._close_build_line()
        self.build_status = f"complete ({format_duration(duration)})"
        self._status_line()

    def build_failed(self, output: str) -> None:
        self._close_build_line()
        self.build_status = "failed"
        self.phase = "Build failed"
        self._status_line()
        self._print_failure("✗ Build Failed", output)

    def tests_missing(self, exe_name: str, paths: list[Path]) -> None:
        self.test_status = "binary missing"
        self.phase = "Tests unavailable"
        self._status_line()
        self._print_failure(f"✗ Could Not Find {exe_name}", "\n".join(str(p) for p in paths))

    def tests_started(self) -> None:
        self.phase = "Testing"
        self.test_status = "running..."
        self._status_line()

    def test_line(self, line: str) -> None:
        line = line.rstrip()
        if line:
            print(f"  {line}")

    def tests_complete(self, duration: float) -> None:
        self.test_status = f"complete ({format_duration(duration)})"
        self._status_line()

    def tests_failed(self, output: str) -> None:
        self.test_status = "failed"
        self.phase = "Tests failed"
        self._status_line()
        self._print_failure("✗ Test Suite Failed", output)

    def success(
        self,
        configure_time: float | None,
        build_time: float,
        test_time: float,
    ) -> None:
        total = sum(value or 0 for value in (configure_time, build_time, test_time))

        print()
        print("=" * 60)
        print("BUILD SUCCESSFUL")
        print("=" * 60)
        print(f"Build type     : {self.build_type}")
        print(f"Generator      : {self.generator}")
        print(f"Parallel jobs  : {self.jobs}")
        print(
            "Configuration  : "
            + (format_duration(configure_time) if configure_time else "cached")
        )
        print(f"Compilation    : {format_duration(build_time)}")
        print(f"Tests          : {format_duration(test_time)}")
        print(f"Total          : {format_duration(total)}")


def run_streaming_command(
    cmd: list[str],
    *,
    on_line,
) -> tuple[int, str]:
    """
    Run a command while streaming its combined stdout/stderr to on_line.

    The complete output is also retained so that failures can show the
    original diagnostic output in full.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    output: list[str] = []

    for line in process.stdout:
        output.append(line)
        on_line(line)

    return process.wait(), "".join(output)


def run_captured_command(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return result.returncode, result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deepity Cross-Platform Build & Test Runner"
    )
    parser.add_argument(
        "build_type",
        nargs="?",
        default="Release",
        choices=["Release", "Debug"],
    )
    args = parser.parse_args()

    build_type = args.build_type
    jobs = multiprocessing.cpu_count()

    build_dir = Path("build") / build_type
    deps_dir = build_dir / "_deps"
    cache_file = build_dir / "CMakeCache.txt"

    ninja = shutil.which("ninja")
    generator = "Ninja" if ninja else "CMake"

    # ────────────────────────────────────────────────────────────────────
    # Setup Logging
    # ────────────────────────────────────────────────────────────────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "build.log"

    # Start a fresh log file for this run
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"--- Deepity Build Log ({build_type}) ---\n\n")

    def log_output(phase: str, output: str) -> None:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"=== {phase} ===\n")
            f.write(output)
            f.write("\n\n")

    reporter: Reporter = (
        RichReporter(build_type, generator, jobs)
        if RICH_AVAILABLE
        else PlainReporter(build_type, generator, jobs)
    )

    with reporter:
        # ────────────────────────────────────────────────────────────────
        # Dependencies
        # ────────────────────────────────────────────────────────────────
        if not deps_dir.is_dir():
            reporter.clean_reconfigure()

            if build_dir.exists():
                shutil.rmtree(build_dir)

        # ────────────────────────────────────────────────────────────────
        # Configure
        # ────────────────────────────────────────────────────────────────
        configure_time: float | None = None

        if not cache_file.is_file():
            config_cmd = [
                "cmake",
                "-B",
                str(build_dir),
                f"-DCMAKE_BUILD_TYPE={build_type}",
                "-DDEEPITY_BUILD_TESTS=ON",
                "-DDEEPITY_ENABLE_CUDA=ON",
            ]

            if sys.platform == "win32":
                config_cmd.extend(["-A", "x64"])
                vcpkg_path = os.environ.get("VCPKG_ROOT", "C:/Users/Jack/vcpkg")
                toolchain = Path(vcpkg_path) / "scripts/buildsystems/vcpkg.cmake"
                if toolchain.is_file():
                    config_cmd.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain.as_posix()}")
            else:
                if ninja:
                    config_cmd.extend(["-G", "Ninja"])

                vcpkg_root = os.environ.get("VCPKG_ROOT")
                if vcpkg_root:
                    config_cmd.append(f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_root}/scripts/buildsystems/vcpkg.cmake")

            reporter.configure_started()

            start = time.perf_counter()

            return_code, config_output = run_captured_command(config_cmd)
            log_output("CMake Configuration", config_output)

            configure_time = time.perf_counter() - start

            if return_code != 0:
                reporter.configure_failed(config_output)
                sys.exit(return_code)

            reporter.configure_complete(configure_time)

        else:
            reporter.configure_cached()

        # ────────────────────────────────────────────────────────────────
        # Build
        # ────────────────────────────────────────────────────────────────
        build_cmd = [
            "cmake",
            "--build",
            str(build_dir),
            "-j",
            str(jobs),
            "--config",
            build_type,
        ]

        reporter.build_started()

        start = time.perf_counter()

        return_code, build_output = run_streaming_command(
            build_cmd,
            on_line=reporter.build_line,
        )
        log_output("Compilation", build_output)

        build_time = time.perf_counter() - start

        if return_code != 0:
            reporter.build_failed(build_output)
            sys.exit(return_code)

        reporter.build_complete(build_time)

        # ────────────────────────────────────────────────────────────────
        # Find test executable
        # ────────────────────────────────────────────────────────────────
        exe_name = "DeepityTests.exe" if os.name == "nt" else "DeepityTests"

        test_paths = [
            build_dir / "bin" / exe_name,
            build_dir / "bin" / build_type / exe_name,
        ]

        test_exe = next((path for path in test_paths if path.is_file()), None)

        if test_exe is None:
            reporter.tests_missing(exe_name, test_paths)
            sys.exit(1)

        # ────────────────────────────────────────────────────────────────
        # Tests
        # ────────────────────────────────────────────────────────────────
        reporter.tests_started()

        test_cmd = [str(test_exe)]

        start = time.perf_counter()

        return_code, test_output = run_streaming_command(
            test_cmd,
            on_line=reporter.test_line,
        )
        log_output("Tests", test_output)

        test_time = time.perf_counter() - start

        if return_code != 0:
            reporter.tests_failed(test_output)
            sys.exit(return_code)

        reporter.tests_complete(test_time)

        # ────────────────────────────────────────────────────────────────
        # Success
        # ────────────────────────────────────────────────────────────────
        reporter.success(configure_time, build_time, test_time)


if __name__ == "__main__":
    main()