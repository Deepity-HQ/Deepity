"""Small process/formatting utilities shared by the CMake runner and reporters."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable

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


def run_streaming_command(
    cmd: list[str],
    *,
    on_line: Callable[[str], None],
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