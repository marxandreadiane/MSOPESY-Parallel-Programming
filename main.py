"""Parallel OCR pipeline entry point."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from ocr_pipeline.config import load_config
from ocr_pipeline.ocr import OcrError
from ocr_pipeline.pipeline import PipelineError, format_stats, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel OCR pipeline")
    parser.add_argument("--config", default="config.txt")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.workers is not None and args.workers < 1:
        print("error: --workers must be at least 1", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
        if args.workers is not None:
            config = replace(config, n_workers=args.workers)
        stats = run_pipeline(
            config,
            output_path=args.output,
            limit=args.limit,
            progress=not args.quiet,
        )
    except (PipelineError, OcrError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    print()
    print(format_stats(stats))
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
