"""CSV output for OCR results."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

CSV_HEADER = ("id", "filename", "text", "processing_ms")


@dataclass(frozen=True)
class Result:
    """One completed image."""

    filename: str
    text: str
    processing_ms: float
    error: str | None = None


class ResultWriter:
    """Write results to a CSV with sequential IDs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(CSV_HEADER)
        self._next_id = 1

    def write(self, result: Result) -> int:
        row_id = self._next_id
        self._next_id += 1
        self._writer.writerow(
            [row_id, result.filename, result.text, f"{result.processing_ms:.2f}"]
        )
        self._handle.flush()
        return row_id

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
