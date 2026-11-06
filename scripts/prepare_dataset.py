"""Extract the dataset into ``data/images``."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LABELS_NAME = "labels.txt"  # ground truth 


def extract_dataset(zip_path: Path, destination: Path) -> int:
    """Write supported images from the zip into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    count = 0

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            parts = Path(info.filename).parts
            if "__MACOSX" in parts:
                continue

            name = Path(info.filename).name
            if name.startswith("."):
                continue

            target = destination / name
            if name == LABELS_NAME:
                target.write_bytes(archive.read(info))
                continue
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                continue

            target.write_bytes(archive.read(info))

            with Image.open(target) as image:
                image.verify()

            count += 1

    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        default=str(ROOT / "data" / "raw" / "dataset.zip"),
        help="path to dataset.zip",
    )
    parser.add_argument(
        "--dest",
        default=str(ROOT / "data" / "images"),
        help="destination directory",
    )
    args = parser.parse_args(argv)

    zip_path = Path(args.zip).expanduser().resolve()
    destination = Path(args.dest).expanduser().resolve()

    try:
        count = extract_dataset(zip_path, destination)
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"extracted {count} image(s) to {destination}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
