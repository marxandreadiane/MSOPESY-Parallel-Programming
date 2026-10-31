"""Tesseract OCR wrapper."""

from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image

DEFAULT_LANG = "eng"
DEFAULT_PSM = 8


class OcrError(RuntimeError):
    """Raised when Tesseract is missing or fails."""


def check_tesseract() -> str:
    """Return the Tesseract version, or raise ``OcrError``."""
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception as exc:
        raise OcrError(
            "Tesseract is not available. Install it and make sure 'tesseract' "
            "is on PATH, or pass the full path to the executable."
        ) from exc


def extract_text(
    image: Image.Image,
    *,
    lang: str = DEFAULT_LANG,
    psm: int = DEFAULT_PSM,
    tessdata_dir: str | Path | None = None,
    tesseract_cmd: str | None = None,
) -> str:
    """Run Tesseract on one image and return the stripped text."""
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    config = f"--oem 3 --psm {psm}"
    if tessdata_dir:
        config += f' --tessdata-dir "{tessdata_dir}"'

    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config)
    except Exception as exc:
        raise OcrError(f"Tesseract failed: {exc}") from exc

    return text.replace("\x0c", "").strip()
