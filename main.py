"""Parallel OCR pipeline entry point."""

from __future__ import annotations

import argparse

from ocr_pipeline.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parallel OCR pipeline")
    parser.add_argument("--config", default="config.txt")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"image_dir: {config.image_dir}")
    print(f"n_workers: {config.n_workers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
