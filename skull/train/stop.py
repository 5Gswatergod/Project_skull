from __future__ import annotations

from pathlib import Path


class StopRequested(KeyboardInterrupt):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__(f"Stop requested via {self.path}")
