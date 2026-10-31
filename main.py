"""Parallel OCR pipeline entry point."""

import sys

from ocr_pipeline.config import load_config
from ocr_pipeline.ocr import OcrError
from ocr_pipeline.pipeline import PipelineError, format_stats, run_pipeline


def main() -> int:
    try:
        stats = run_pipeline(load_config("config.txt"))
    except (PipelineError, OcrError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_stats(stats))
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
