from __future__ import annotations

from collections import deque
from pathlib import Path

from ..config import BuildConfig
from ..git_info import GitInfo
from ..process import BUILD_PROGRESS_RE, format_duration
from .base import Reporter


class RichReporter(Reporter):
    """Single mutable Rich Live dashboard for the entire build."""

    def __init__(self, config: BuildConfig, git: GitInfo, generator: str) -> None:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self.config = config
        self.git = git
        self.generator = generator

        self.targets_built = 0
        self.num_passed = 0
        self.num_failed = 0

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

        profile = self.config.profile
        arch_style = "green" if profile.name == "distributed" else (
            "yellow" if profile.name == "fast" else "red"
        )

        table = Table.grid(padding=(0, 2))
        table.add_column()
        table.add_column()
        table.add_column()
        table.add_column()

        table.add_row(
            "[bold cyan]Deepity Engine[/bold cyan]",
            f"[dim]Build[/dim] [bold]{self.config.build_type}[/bold]",
            f"[dim]Generator[/dim] [bold]{self.generator}[/bold]",
            f"[dim]Jobs[/dim] [bold]{self.config.jobs}[/bold]",
        )

        table.add_row(
            "[dim]Arch profile[/dim]",
            f"[bold {arch_style}]{profile.name}[/bold {arch_style}]"
            + (
                "  [dim](not portable — local dev only)[/dim]"
                if profile.name == "native"
                else ""
            ),
            "[dim]CUDA[/dim]",
            "[bold green]ON[/bold green]" if self.config.cuda else "[dim]off[/dim]",
        )

        table.add_row(
            "[dim]Git[/dim]",
            f"[bold]{self.git.branch}[/bold] @ {self.git.commit}"
            + (" [yellow]● modified[/yellow]" if self.git.dirty else " [green]✓ clean[/green]"),
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

            body.add_row("Build type", self.config.build_type)
            body.add_row("Arch profile", self.config.profile.name)
            body.add_row("CUDA", "ON" if self.config.cuda else "off")
            body.add_row("Parallel jobs", str(self.config.jobs))
            body.add_row("Targets", str(self.targets_built))
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

    def build_summary(self, targets: int) -> None:
        self.targets_built = targets
        self._refresh()

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
        self.targets_built = max(self.targets_built, total)
        self.build_message = message
        self.progress.update(self.build_task, total=total, completed=current)

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