from __future__ import annotations

import csv
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from ocr_pipeline.config import Config
from ocr_pipeline.pipeline import PipelineError, list_images, run_pipeline


def make_image(path: Path) -> None:
    image = Image.new("RGB", (80, 30), "white")
    image.save(path)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for index in range(1, 6):
            make_image(self.images / f"img{index:04d}.png")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, workers: int = 3) -> Config:
        return Config(image_dir=self.images, n_workers=workers)

    def read_rows(self, path: Path) -> list[list[str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))

    def test_processes_every_image_with_sequential_ids(self) -> None:
        def fake_ocr(image) -> str:
            time.sleep(0.005)
            return "recognized"

        output = self.root / "out" / "ocr_results.csv"
        stats = run_pipeline(
            self.config(workers=3),
            ocr_fn=fake_ocr,
            output_path=output,
            progress=False,
        )

        self.assertEqual(stats.images, 5)
        self.assertEqual(stats.processed, 5)
        self.assertEqual(stats.failed, 0)
        self.assertGreater(stats.elapsed_s, 0)

        rows = self.read_rows(output)
        self.assertEqual(rows[0], ["id", "filename", "text", "processing_ms"])
        body = rows[1:]
        self.assertEqual([row[0] for row in body], ["1", "2", "3", "4", "5"])
        self.assertEqual(
            {row[1] for row in body},
            {f"img{index:04d}.png" for index in range(1, 6)},
        )
        self.assertTrue(all(row[2] == "recognized" for row in body))

    def test_one_worker_finishes_cleanly(self) -> None:
        output = self.root / "out.csv"
        stats = run_pipeline(
            self.config(workers=1),
            ocr_fn=lambda image: "ok",
            output_path=output,
            progress=False,
        )

        self.assertEqual(stats.processed, 5)
        self.assertEqual(len(self.read_rows(output)), 6)

    def test_bad_image_is_reported_without_stopping_others(self) -> None:
        (self.images / "img0003.png").write_bytes(b"not an image")
        output = self.root / "out.csv"

        stats = run_pipeline(
            self.config(workers=2),
            ocr_fn=lambda image: "ok",
            output_path=output,
            progress=False,
        )

        self.assertEqual(stats.processed, 5)
        self.assertEqual(stats.failed, 1)
        rows = self.read_rows(output)
        bad = [row for row in rows if row[1] == "img0003.png"]
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0][2], "")

    def test_limit_processes_only_first_images(self) -> None:
        output = self.root / "out.csv"
        stats = run_pipeline(
            self.config(workers=2),
            ocr_fn=lambda image: "ok",
            output_path=output,
            limit=2,
            progress=False,
        )

        self.assertEqual(stats.images, 2)
        self.assertEqual(len(self.read_rows(output)), 3)

    def test_empty_directory_raises(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()

        with self.assertRaisesRegex(PipelineError, "No supported images"):
            run_pipeline(
                Config(image_dir=empty, n_workers=2),
                ocr_fn=lambda image: "ok",
                output_path=self.root / "out.csv",
                progress=False,
            )

    def test_list_images_is_sorted_and_filtered(self) -> None:
        (self.images / "notes.txt").write_text("ignore me", encoding="utf-8")
        names = [path.name for path in list_images(self.images)]
        self.assertEqual(names, [f"img{index:04d}.png" for index in range(1, 6)])

    def test_output_path_is_recorded(self) -> None:
        output = self.root / "custom" / "results.csv"
        stats = run_pipeline(
            self.config(workers=2),
            ocr_fn=lambda image: "ok",
            output_path=output,
            progress=False,
        )

        self.assertEqual(stats.output_csv, output.resolve())
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
