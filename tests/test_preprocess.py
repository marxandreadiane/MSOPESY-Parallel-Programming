from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from ocr_pipeline.preprocess import BORDER, load_image, preprocess_image


def make_image(path: Path, width: int = 80, height: int = 30) -> None:
    """White image with a black rectangle in the middle."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 8, 59, 21), fill="black")
    image.save(path)


class PreprocessTests(unittest.TestCase):
    def test_load_image_returns_grayscale(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            make_image(path)

            image = load_image(path)

            self.assertEqual(image.mode, "L")
            self.assertEqual(image.size, (80, 30))

    def test_preprocess_returns_grayscale(self) -> None:
        image = preprocess_image(Image.new("RGB", (400, 400), "white"))
        self.assertEqual(image.mode, "L")

    def test_preprocess_binarizes_pixels(self) -> None:
        image = preprocess_image(Image.new("RGB", (400, 400), "white"))
        # Only the 0 and 255 histogram bins may contain pixels.
        self.assertEqual(sum(image.histogram()[1:255]), 0)

    def test_small_image_is_upscaled(self) -> None:
        image = preprocess_image(Image.new("RGB", (80, 30), "white"))
        # 80x30 -> factor 4 -> 320x120, plus a 20 px border on every side.
        self.assertEqual(image.size, (80 * 4 + 2 * BORDER, 30 * 4 + 2 * BORDER))

    def test_large_image_is_not_upscaled(self) -> None:
        image = preprocess_image(Image.new("RGB", (400, 300), "white"))
        self.assertEqual(image.size, (400 + 2 * BORDER, 300 + 2 * BORDER))

    def test_border_is_white(self) -> None:
        image = preprocess_image(Image.new("RGB", (400, 400), "white"))
        self.assertEqual(image.getpixel((0, 0)), 255)
        self.assertEqual(image.getpixel((image.width - 1, image.height - 1)), 255)

    def test_dark_text_stays_black(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            make_image(path)

            image = preprocess_image(load_image(path))

            self.assertEqual(image.getpixel((BORDER + 100, BORDER + 40)), 0)


if __name__ == "__main__":
    unittest.main()
