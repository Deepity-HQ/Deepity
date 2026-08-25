from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Reporter(ABC):
    @abstractmethod
    def __enter__(self) -> "Reporter": ...

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None: ...

    @abstractmethod
    def build_summary(self, targets: int) -> None: ...

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