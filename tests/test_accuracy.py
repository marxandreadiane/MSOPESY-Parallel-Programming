from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from ocr_pipeline.accuracy import edit_distance, load_labels, score

ROOT = Path(__file__).resolve().parents[1]


def write_results(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = ["id,filename,text,processing_ms"]
    lines += [f"{i},{name},{text},1.00" for i, (name, text) in enumerate(rows, 1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class AccuracyTests(unittest.TestCase):
    def test_edit_distance(self) -> None:
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("same", "same"), 0)

    def test_score_exact_and_char_rates(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "r.csv"
            write_results(csv_path, [("a.png", "Grape"), ("b.png", "grap"), ("c.png", "")])
            labels = {"a.png": "grape", "b.png": "grape", "c.png": "ab"}

            result = score(csv_path, labels)

        self.assertEqual((result.labeled, result.exact), (3, 1))  # case-insensitive
        self.assertEqual(result.truth_chars, 12)
        self.assertEqual(result.edit_distance, 3)  # 0 + 1 + 2
        self.assertAlmostEqual(result.char_rate, 0.75)

    def test_unlabeled_images_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "r.csv"
            write_results(csv_path, [("a.png", "x"), ("z.png", "y")])

            result = score(csv_path, {"a.png": "x"})

        self.assertEqual((result.labeled, result.unlabeled), (1, 1))

    def test_load_labels(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.txt"
            path.write_text("filename,word\nimg1.png,apple\n", encoding="utf-8")

            self.assertEqual(load_labels(path), {"img1.png": "apple"})


class PrepareDatasetTests(unittest.TestCase):
    def test_extracts_images_and_labels_but_skips_macosx(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "prepare_dataset", ROOT / "scripts" / "prepare_dataset.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (2, 2)).save(root / "img.png")
            archive = root / "d.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.write(root / "img.png", "dataset/img.png")
                z.writestr("dataset/labels.txt", "filename,word\nimg.png,hi\n")
                z.writestr("__MACOSX/dataset/labels.txt", "junk")
                z.writestr("dataset/notes.txt", "ignore me")

            out = root / "out"
            count = module.extract_dataset(archive, out)

            self.assertEqual(count, 1)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["img.png", "labels.txt"])
            self.assertIn("img.png,hi", (out / "labels.txt").read_text())


if __name__ == "__main__":
    unittest.main()
