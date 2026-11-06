"""Benchmark the parallel OCR pipeline for 1-16 workers."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_pipeline.config import load_config
from ocr_pipeline.pipeline import run_pipeline


def summarize(runs: list[dict]) -> list[dict]:
    grouped: dict[int, list[float]] = {}
    images = 0
    for run in runs:
        grouped.setdefault(run["workers"], []).append(run["elapsed_s"])
        images = run["images"]

    medians = {workers: statistics.median(times) for workers, times in grouped.items()}
    baseline = medians[1]

    rows = []
    for workers in sorted(medians):
        median_s = medians[workers]
        speedup = baseline / median_s if median_s else 0.0
        rows.append(
            {
                "workers": workers,
                "runs": len(grouped[workers]),
                "median_s": round(median_s, 3),
                "speedup": round(speedup, 2),
                "efficiency": round(speedup / workers, 2),
                "throughput": round(images / median_s, 2) if median_s else 0.0,
            }
        )
    return rows


def quality(path: Path) -> tuple[int, int, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    empty = sum(1 for row in rows if not row["text"])
    junk = sum(1 for row in rows if row["text"] and not row["text"][0].isalnum())
    mean_length = sum(len(row["text"]) for row in rows) / len(rows) if rows else 0.0
    return empty, junk, mean_length


def write_reports(rows: list[dict], quality_row: tuple[int, int, float], folder: Path) -> tuple[Path, Path]:
    csv_path = folder / "benchmark_results.csv"
    md_path = folder / "benchmark_results.md"
    fields = ["workers", "runs", "median_s", "speedup", "efficiency", "throughput"]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# OCR pipeline benchmark",
        "",
        "| Workers | Runs | Median time (s) | Speedup | Efficiency | Throughput (img/s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['workers']} | {row['runs']} | {row['median_s']:.3f} | "
            f"{row['speedup']:.2f}x | {row['efficiency']:.2f} | {row['throughput']:.2f} |"
        )

    empty, junk, mean_length = quality_row
    lines.extend(
        [
            "",
            "## Output quality",
            "",
            "| Empty rows | Leading-junk rows | Mean text length |",
            "| ---: | ---: | ---: |",
            f"| {empty} | {junk} | {mean_length:.1f} |",
            "",
            "Speedup uses the 1-worker median as the baseline.",
            "Efficiency = speedup / workers.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.txt"))
    parser.add_argument("--workers", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-dir", default=str(ROOT / "benchmarks"))
    args = parser.parse_args(argv)

    if 1 not in args.workers:
        print("error: --workers must include 1 for the baseline", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("error: --repeat must be at least 1", file=sys.stderr)
        return 2

    config = load_config(args.config)
    folder = Path(args.output_dir).resolve()
    runs_folder = folder / "runs"
    runs_folder.mkdir(parents=True, exist_ok=True)

    runs = []
    for workers in args.workers:
        for repeat in range(1, args.repeat + 1):
            print(f"workers={workers} repeat={repeat}")
            destination = runs_folder / f"run_w{workers}_r{repeat}.csv"
            stats = run_pipeline(
                replace(config, n_workers=workers),
                output_path=destination,
                progress=False,
            )
            print(
                f"  {stats.elapsed_s:.3f} s, "
                f"{stats.processed}/{stats.images} processed, {stats.failed} failed"
            )
            runs.append(
                {
                    "workers": workers,
                    "repeat": repeat,
                    "elapsed_s": stats.elapsed_s,
                    "images": stats.images,
                    "failed": stats.failed,
                }
            )

    rows = summarize(runs)
    sample = runs_folder / "run_w4_r1.csv"
    if not sample.is_file():
        sample = next(runs_folder.glob("run_w*_r1.csv"))
    csv_path, md_path = write_reports(rows, quality(sample), folder)

    print()
    print("| Workers | Median s | Speedup | Efficiency | Throughput |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['workers']} | {row['median_s']:.3f} | "
            f"{row['speedup']:.2f}x | {row['efficiency']:.2f} | "
            f"{row['throughput']:.2f} img/s |"
        )
    print()
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
