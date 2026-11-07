"""Compare OCR output against the dataset's ground-truth labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

LABELS_NAME = "labels.txt"


@dataclass(frozen=True)
class Accuracy:
    labeled: int
    exact: int
    truth_chars: int
    edit_distance: int
    unlabeled: int

    @property
    def exact_rate(self) -> float:
        return self.exact / self.labeled if self.labeled else 0.0

    @property
    def char_rate(self) -> float:
        """Share of ground-truth characters recognized (1 - edit distance / characters)."""
        if not self.truth_chars:
            return 0.0
        return max(0.0, 1.0 - self.edit_distance / self.truth_chars)


def load_labels(path: str | Path) -> dict[str, str]:
    """Read ``filename,word`` rows into ``{filename: word}``."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return {row["filename"]: row["word"] for row in csv.DictReader(handle)}


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def score(results_csv: str | Path, labels: dict[str, str]) -> Accuracy:
    """Compare the ``text`` column of the results CSV with the labels."""
    labeled = exact = truth_chars = distance = unlabeled = 0
    with Path(results_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            truth = labels.get(row["filename"])
            if truth is None:
                unlabeled += 1
                continue
            truth = truth.strip().lower()
            text = row["text"].strip().lower()
            labeled += 1
            exact += text == truth
            truth_chars += len(truth)
            distance += edit_distance(text, truth)
    return Accuracy(labeled, exact, truth_chars, distance, unlabeled)


def format_accuracy(result: Accuracy) -> str:
    lines = [
        f"exact match: {result.exact}/{result.labeled} ({result.exact_rate:.1%})",
        f"char accuracy: {result.char_rate:.1%} "
        f"({result.edit_distance} edits / {result.truth_chars} chars)",
    ]
    if result.unlabeled:
        lines.append(f"unlabeled:   {result.unlabeled} image(s) skipped")
    return "\n".join(lines) + "\n"
