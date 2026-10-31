"""Parallel OCR pipeline"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config
from .ocr import check_tesseract, extract_text
from .preprocess import load_image, preprocess_image
from .results import Result, ResultWriter

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
OcrFn = Callable[[object], str]


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot start."""


@dataclass(frozen=True)
class PipelineStats:
    images: int
    processed: int
    failed: int
    elapsed_s: float
    sum_processing_ms: float
    output_csv: Path

    @property
    def throughput(self) -> float:
        return self.images / self.elapsed_s if self.elapsed_s else 0.0

    @property
    def average_processing_ms(self) -> float:
        return self.sum_processing_ms / self.processed if self.processed else 0.0


def list_images(directory: str | Path) -> list[Path]:
    """Return supported images sorted by filename."""
    folder = Path(directory)
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )


def run_pipeline(
    config: Config,
    *,
    ocr_fn: OcrFn | None = None,
    output_path: str | Path | None = None,
    limit: int | None = None,
    progress: bool = True,
) -> PipelineStats:
    """Process every image and write one CSV."""
    images = list_images(config.image_dir)
    if limit is not None:
        if limit < 1:
            raise PipelineError("limit must be at least 1")
        images = images[:limit]
    if not images:
        raise PipelineError(f"No supported images found in {config.image_dir}")

    destination = (
        Path(output_path).resolve()
        if output_path
        else (Path.cwd() / "outputs" / "ocr_results.csv")
    )

    if ocr_fn is None:
        check_tesseract()
        ocr_fn = extract_text

    n_workers = config.n_workers
    work_queue: queue.Queue[Path | None] = queue.Queue(maxsize=max(2, n_workers * 2))
    result_queue: queue.Queue[Result] = queue.Queue()

    def loader() -> None:
        for path in images:
            work_queue.put(path)
        for _ in range(n_workers):
            work_queue.put(None)

    def worker() -> None:
        while True:
            path = work_queue.get()
            if path is None:
                return

            started = time.perf_counter()
            try:
                image = load_image(path)
                prepared = preprocess_image(image)
                text = ocr_fn(prepared)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                result_queue.put(Result(path.name, text, elapsed_ms))
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                result_queue.put(
                    Result(
                        path.name,
                        "",
                        elapsed_ms,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    started_at = time.perf_counter()
    loader_thread = threading.Thread(target=loader, name="loader")
    workers = [
        threading.Thread(target=worker, name=f"ocr-{index}")
        for index in range(1, n_workers + 1)
    ]

    processed = 0
    failed = 0
    sum_processing_ms = 0.0
    total = len(images)

    loader_thread.start()
    for thread in workers:
        thread.start()

    try:
        with ResultWriter(destination) as writer:
            while processed < total:
                result = result_queue.get()
                writer.write(result)
                processed += 1
                sum_processing_ms += result.processing_ms

                if result.error is not None:
                    failed += 1
                    print(
                        f"[warn] {result.filename}: {result.error}",
                        file=sys.stderr,
                    )

                if progress and (processed % 10 == 0 or processed == total):
                    print(f"[{processed}/{total}] images written")
    finally:
        for thread in workers:
            thread.join()
        loader_thread.join()

    elapsed_s = time.perf_counter() - started_at
    return PipelineStats(
        images=total,
        processed=processed,
        failed=failed,
        elapsed_s=elapsed_s,
        sum_processing_ms=sum_processing_ms,
        output_csv=destination,
    )


def format_stats(stats: PipelineStats) -> str:
    return (
        f"images:      {stats.images}\n"
        f"processed:   {stats.processed}\n"
        f"failed:      {stats.failed}\n"
        f"wall time:   {stats.elapsed_s:.3f} s\n"
        f"throughput:  {stats.throughput:.2f} images/s\n"
        f"avg worker:  {stats.average_processing_ms:.1f} ms/image\n"
        f"output:      {stats.output_csv}"
    )
