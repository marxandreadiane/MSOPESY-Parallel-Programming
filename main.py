"""Parallel OCR pipeline entry point."""

import multiprocessing as mp
import sys
from pathlib import Path

from ocr_pipeline.config import load_config
from ocr_pipeline.ocr import OcrError
from ocr_pipeline.pipeline import PipelineError, format_stats, run_pipeline


def main() -> int:
    print("=== Parallel OCR Pipeline ===")

    try:
        config = load_config("config.txt")
        print("config:    config.txt")
        print(f"images:    {config.image_dir}")
        print(f"workers:   {config.n_workers}")
        print(f"output:    {Path.cwd() / 'outputs' / 'ocr_results.csv'}")
        print()
        print("pipeline started")

        stats = run_pipeline(config)
    except (PipelineError, OcrError, FileNotFoundError, ValueError) as exc:
        print("pipeline failed", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("pipeline finished")
    print(format_stats(stats))
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    mp.set_start_method("spawn")
    raise SystemExit(main())
