"""Tesseract OCR wrapper."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytesseract
from PIL import Image

DEFAULT_LANG = "eng"
DEFAULT_PSM = 8
TESSERACT_CANDIDATES = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)


class OcrError(RuntimeError):
    """Raised when Tesseract fails."""


def find_tesseract() -> str | None:
    """Return a Tesseract path when it is not on PATH."""
    if shutil.which("tesseract"):
        return None
    for candidate in TESSERACT_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


def _configure(tesseract_cmd: str | None) -> None:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    else:
        found = find_tesseract()
        if found:
            pytesseract.pytesseract.tesseract_cmd = found


def check_tesseract() -> str:
    _configure(None)
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception as exc:
        raise OcrError(
            "Tesseract is not available."
        ) from exc


def extract_text(
    image: Image.Image,
    *,
    lang: str = DEFAULT_LANG,
    psm: int = DEFAULT_PSM,
    tessdata_dir: str | Path | None = None,
    tesseract_cmd: str | None = None,
) -> str:
    """Run Tesseract on one image and return the extracted text."""
    _configure(tesseract_cmd)

    config = f"--oem 3 --psm {psm}"
    if tessdata_dir:
        config += f' --tessdata-dir "{tessdata_dir}"'

    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config)
    except Exception as exc:
        raise OcrError(f"Tesseract failed: {exc}") from exc

    return text.replace("\x0c", "").strip()
