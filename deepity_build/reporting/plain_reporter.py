from __future__ import annotations

from pathlib import Path

from ..config import BuildConfig
from ..git_info import GitInfo
from ..process import BUILD_PROGRESS_RE, format_duration
from .base import Reporter


class PlainReporter(Reporter):
    """
    stdout-only mirror of RichReporter. Same public interface, no rich
    dependency at all: statuses are printed as they change, build progress
    is shown as an in-place percentage line, and test output streams
    straight through.
    """

    def __init__(self, config: BuildConfig, git: GitInfo, generator: str) -> None:
        self.config = config
        self.git = git
        self.generator = generator

        self.phase = "Starting..."
        self.configure_status = "waiting"
        self.build_status = "waiting"
        self.test_status = "waiting"

        self.targets_built = 0
        self._last_build_pct = -1
        self._build_line_open = False

    def __enter__(self) -> "PlainReporter":
        profile = self.config.profile
        print(
            f"Deepity Engine — build={self.config.build_type} "
            f"generator={self.generator} jobs={self.config.jobs} "
            f"arch={profile.name} cuda={'on' if self.config.cuda else 'off'}"
        )
        if profile.name == "native":
            print("  ⚠ arch=native is not portable — do not distribute this binary.")
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

    def build_summary(self, targets: int) -> None:
        print(f"Targets built: {targets}")

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
        print(f"Build type     : {self.config.build_type}")
        print(f"Generator      : {self.generator}")
        print(f"Arch profile   : {self.config.profile.name}")
        print(f"CUDA           : {'ON' if self.config.cuda else 'off'}")
        print(f"Parallel jobs  : {self.config.jobs}")
        print(f"Targets build  : {self.targets_built}")
        print(
            "Configuration  : "
            + (format_duration(configure_time) if configure_time else "cached")
        )
        print(f"Compilation    : {format_duration(build_time)}")
        print(f"Tests          : {format_duration(test_time)}")
        print(f"Total          : {format_duration(total)}")