from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytesseract
from PIL import Image

from ocr_pipeline.ocr import OcrError, check_tesseract, extract_text, find_tesseract


class OcrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("L", (10, 10), "white")
        self.original_cmd = pytesseract.pytesseract.tesseract_cmd

    def tearDown(self) -> None:
        pytesseract.pytesseract.tesseract_cmd = self.original_cmd

    def test_extract_text_strips_whitespace_and_form_feed(self) -> None:
        with patch(
            "ocr_pipeline.ocr.pytesseract.image_to_string",
            return_value="  hello\x0c\n",
        ):
            self.assertEqual(extract_text(self.image), "hello")

    def test_extract_text_passes_language_and_psm(self) -> None:
        with patch(
            "ocr_pipeline.ocr.pytesseract.image_to_string", return_value="text"
        ) as mocked:
            extract_text(self.image, lang="eng", psm=7)

        arguments = mocked.call_args.kwargs
        self.assertEqual(arguments["lang"], "eng")
        self.assertIn("--oem 3", arguments["config"])
        self.assertIn("--psm 7", arguments["config"])

    def test_tessdata_dir_is_passed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tessdata = Path(folder)
            with patch(
                "ocr_pipeline.ocr.pytesseract.image_to_string", return_value="text"
            ) as mocked:
                extract_text(self.image, tessdata_dir=tessdata)

        self.assertIn(f'--tessdata-dir "{tessdata}"', mocked.call_args.kwargs["config"])

    def test_tesseract_cmd_is_applied(self) -> None:
        command = r"C:\Tesseract-OCR\tesseract.exe"
        with patch("ocr_pipeline.ocr.pytesseract.image_to_string", return_value="text"):
            extract_text(self.image, tesseract_cmd=command)

        self.assertEqual(pytesseract.pytesseract.tesseract_cmd, command)

    def test_extract_text_wraps_tesseract_failure(self) -> None:
        with patch(
            "ocr_pipeline.ocr.pytesseract.image_to_string",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaisesRegex(OcrError, "Tesseract failed: boom"):
                extract_text(self.image)

    def test_check_tesseract_returns_version(self) -> None:
        with patch(
            "ocr_pipeline.ocr.pytesseract.get_tesseract_version",
            return_value="5.3.0",
        ):
            self.assertEqual(check_tesseract(), "5.3.0")

    def test_check_tesseract_raises_when_missing(self) -> None:
        with patch(
            "ocr_pipeline.ocr.pytesseract.get_tesseract_version",
            side_effect=Exception("not found"),
        ):
            with self.assertRaisesRegex(OcrError, "Tesseract is not available"):
                check_tesseract()

    def test_find_tesseract_returns_none_when_on_path(self) -> None:
        with patch(
            "ocr_pipeline.ocr.shutil.which",
            return_value=r"C:\fake\tesseract.exe",
        ):
            self.assertIsNone(find_tesseract())

    def test_find_tesseract_uses_known_location(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            candidate = Path(folder) / "tesseract.exe"
            candidate.write_bytes(b"")

            with patch("ocr_pipeline.ocr.shutil.which", return_value=None), patch(
                "ocr_pipeline.ocr.TESSERACT_CANDIDATES", (candidate,)
            ):
                self.assertEqual(find_tesseract(), str(candidate))

    def test_check_tesseract_uses_known_location(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            candidate = Path(folder) / "tesseract.exe"
            candidate.write_bytes(b"")

            with patch("ocr_pipeline.ocr.shutil.which", return_value=None), patch(
                "ocr_pipeline.ocr.TESSERACT_CANDIDATES", (candidate,)
            ), patch(
                "ocr_pipeline.ocr.pytesseract.get_tesseract_version",
                return_value="5.4.0",
            ):
                self.assertEqual(check_tesseract(), "5.4.0")

            self.assertEqual(pytesseract.pytesseract.tesseract_cmd, str(candidate))

    def test_extract_text_uses_known_location(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            candidate = Path(folder) / "tesseract.exe"
            candidate.write_bytes(b"")

            with patch("ocr_pipeline.ocr.shutil.which", return_value=None), patch(
                "ocr_pipeline.ocr.TESSERACT_CANDIDATES", (candidate,)
            ), patch(
                "ocr_pipeline.ocr.pytesseract.image_to_string",
                return_value="text",
            ):
                self.assertEqual(extract_text(self.image), "text")

            self.assertEqual(pytesseract.pytesseract.tesseract_cmd, str(candidate))


if __name__ == "__main__":
    unittest.main()
