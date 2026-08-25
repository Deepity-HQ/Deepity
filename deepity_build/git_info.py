"""Best-effort git metadata for the reporter header. Never raises."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GitInfo:
    branch: str
    commit: str
    dirty: bool


def get_git_info() -> GitInfo:
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        status = subprocess.call(
            ["git", "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        dirty = status != 0
        return GitInfo(branch or "detached", commit, dirty)

    except (subprocess.CalledProcessError, FileNotFoundError):
        return GitInfo("unknown", "unknown", False)