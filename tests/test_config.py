from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ocr_pipeline.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_loads_repository_config(self) -> None:
        config = load_config(ROOT / "config.txt")

        self.assertEqual(config.image_dir, (ROOT / "data" / "images").resolve())
        self.assertEqual(config.n_workers, 2)

    def test_resolves_relative_image_dir_against_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            (root / "config.txt").write_text(
                "image_dir=images\nn_workers=3\n", encoding="utf-8"
            )

            config = load_config(root / "config.txt")

            self.assertEqual(config.image_dir, (root / "images").resolve())
            self.assertEqual(config.n_workers, 3)

    def test_keeps_absolute_image_dir(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            images.mkdir()
            (root / "config.txt").write_text(
                f"image_dir={images}\nn_workers=1\n", encoding="utf-8"
            )

            config = load_config(root / "config.txt")

            self.assertEqual(config.image_dir, images.resolve())

    def test_reads_n_workers_as_integer(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.txt").write_text(
                "image_dir=.\nn_workers=8\n", encoding="utf-8"
            )

            self.assertEqual(load_config(root / "config.txt").n_workers, 8)
            self.assertIsInstance(load_config(root / "config.txt").n_workers, int)

    def test_missing_config_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_config(Path(folder) / "absent.txt")

    def test_skips_blank_lines_and_comments(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.txt").write_text(
                "# settings\n\nimage_dir=.\n\nn_workers=4\n", encoding="utf-8"
            )

            self.assertEqual(load_config(root / "config.txt").n_workers, 4)

    def test_missing_key_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.txt").write_text("image_dir=.\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "n_workers"):
                load_config(root / "config.txt")

    def test_rejects_zero_workers(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.txt").write_text(
                "image_dir=.\nn_workers=0\n", encoding="utf-8"
            )

            with self.assertRaises(ValueError):
                load_config(root / "config.txt")

    def test_returns_config_dataclass(self) -> None:
        self.assertIsInstance(load_config(ROOT / "config.txt"), Config)


if __name__ == "__main__":
    unittest.main()
