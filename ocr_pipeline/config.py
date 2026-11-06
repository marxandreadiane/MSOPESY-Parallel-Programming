"""Read ``image_dir`` and ``n_workers`` from ``config.txt``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    image_dir: Path
    n_workers: int


def load_config(path: str | Path = "config.txt") -> Config:
    source = Path(path).expanduser().resolve()
    values: dict[str, str] = {}
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    image_dir = Path(values["image_dir"]).expanduser()
    if not image_dir.is_absolute():
        image_dir = (source.parent / image_dir).resolve()

    n_workers = int(values["n_workers"])
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")

    return Config(image_dir=image_dir, n_workers=n_workers)
