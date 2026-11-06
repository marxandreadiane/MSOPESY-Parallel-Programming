"""Parallel OCR pipeline (multiprocessing version)"""

from __future__ import annotations
 
import difflib
import multiprocessing as mp
import queue
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
 
from .config import Config
from .ocr import check_tesseract, extract_text
from .preprocess import load_image, preprocess_image
from .results import Result, ResultWriter
 
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
 
 
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
 
 
def clean_text(text: str) -> str:
    """Remove symbols at the start and end of the text."""
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text)
  
def loader_process(images: list[Path], work_queue, n_workers: int) -> None:
    """Put every image path on the work queue, then one stop signal (None) per worker."""
    try:
        for path in images:
            work_queue.put(path)           
    finally:
        for _ in range(n_workers):
            work_queue.put(None)
 
 
def worker_process(work_queue, result_queue) -> None:
    """Take paths from the work queue, preprocess + OCR each, send Results back."""
    try:
        check_tesseract()   
        while True:
            path = work_queue.get()       
            if path is None:              
                break
 
            started = time.perf_counter()
            try:
                image = load_image(path)
                prepared = preprocess_image(image)
                text = clean_text(extract_text(prepared))
                error = None
            except Exception as exc:       
                text = ""
                error = f"{type(exc).__name__}: {exc}"
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result_queue.put(Result(path.name, text, elapsed_ms, error=error))
    finally:
        result_queue.put(None)             
 
 
def run_pipeline(
    config: Config,
    *,
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
 
    check_tesseract() 
 
    n_workers = config.n_workers
    work_queue = mp.Queue(maxsize=max(2, n_workers * 2))
    result_queue = mp.Queue()
 
    loader = mp.Process(
        target=loader_process, args=(images, work_queue, n_workers), name="loader"
    )
    workers = [
        mp.Process(target=worker_process, args=(work_queue, result_queue), name=f"ocr-{i}")
        for i in range(1, n_workers + 1)
    ]
    processes = [loader, *workers]
 
    processed = 0
    failed = 0
    exact_matches = 0
    char_similarity_sum = 0.0
    sum_processing_ms = 0.0
    total = len(images)
    workers_finished = 0
 
    # The clock starts before the processes launch, so startup cost is included.
    started_at = time.perf_counter()
    for process in processes:
        process.start()
 
    try:
        with ResultWriter(destination) as writer:  
            while workers_finished < n_workers:
                try:
                    result = result_queue.get(timeout=1)
                except queue.Empty:
                    if not any(worker.is_alive() for worker in workers):
                        break
                    continue
 
                if result is None:                  
                    workers_finished += 1
                    continue
 
                writer.write(result)
                processed += 1
                sum_processing_ms += result.processing_ms
 
                if result.error is not None:
                    failed += 1
                    print(f"[warn] {result.filename}: {result.error}", file=sys.stderr)
 
                if progress and (processed % 10 == 0 or processed == total):
                    print(f"[{processed}/{total}] images written")
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
 
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
        f"output:      {stats.output_csv}\n"
    )
 