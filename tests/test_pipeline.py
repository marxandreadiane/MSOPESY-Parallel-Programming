from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from ocr_pipeline import ocr
from ocr_pipeline.pipeline import list_images, loader_process

ROOT = Path(__file__).resolve().parents[1]


class ListImagesTests(unittest.TestCase):
    def test_only_png_sorted_ignores_labels_txt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("b.png", "A.PNG", "labels.txt"):
                (root / name).write_bytes(b"")

            self.assertEqual([p.name for p in list_images(root)], ["A.PNG", "b.png"])


class LoaderTests(unittest.TestCase):
    def test_loader_queues_images_then_one_sentinel_per_worker(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            good = Path(folder) / "good.png"
            Image.new("RGB", (4, 3), "white").save(good)
            bad = Path(folder) / "bad.png"
            bad.write_bytes(b"not an image")

            queue = mp.Queue()
            loader_process([good, bad], queue, 2)
            items = [queue.get(timeout=5) for _ in range(4)]

            name, image, error = items[0]
            self.assertEqual((name, image.size, error), ("good.png", (4, 3), None))
            self.assertEqual(items[1][0], "bad.png")
            self.assertIsNone(items[1][1])
            self.assertIsNotNone(items[1][2])
            self.assertEqual(items[2:], [None, None])


class TesseractDiscoveryTests(unittest.TestCase):
    def test_env_var_wins(self) -> None:
        with mock.patch.dict(os.environ, {"TESSERACT_CMD": r"C:\x\tesseract.exe"}):
            self.assertEqual(ocr.find_tesseract(), r"C:\x\tesseract.exe")

    def test_returns_none_when_nothing_found(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TESSERACT_CMD"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch(
            "shutil.which", return_value=None
        ), mock.patch.object(ocr, "WINDOWS_DEFAULT", Path("nope/tesseract.exe")):
            self.assertIsNone(ocr.find_tesseract())

    @unittest.skipUnless(ocr.find_tesseract(), "Tesseract not installed")
    def test_tessdata_dir_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "with space"
            shutil.copytree(ROOT / "models" / "tessdata", data)
            image = Image.new("L", (200, 80), 255)
            self.assertIsInstance(ocr.extract_text(image, tessdata_dir=data), str)


if __name__ == "__main__":
    unittest.main()
