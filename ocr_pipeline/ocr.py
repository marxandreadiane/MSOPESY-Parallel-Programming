"""Tesseract OCR wrapper."""

from pathlib import Path

import pytesseract
from PIL import Image

DEFAULT_LANG = "eng"
DEFAULT_PSM = 8
REPO_TESSDATA = Path(__file__).resolve().parents[1] / "models" / "tessdata"


class OcrError(RuntimeError):
    """Raised when Tesseract fails."""


def check_tesseract() -> str:
    """Return the Tesseract version, or raise OcrError."""
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception as exc:
        raise OcrError("Tesseract is not available.") from exc


def extract_text(
    image: Image.Image,
    *,
    lang: str = DEFAULT_LANG,
    psm: int = DEFAULT_PSM,
    tessdata_dir: str | Path | None = None,
) -> str:
    """Run Tesseract on one image and return the extracted text."""
    if tessdata_dir is None and REPO_TESSDATA.is_dir():
        tessdata_dir = REPO_TESSDATA

    config = f"--oem 3 --psm {psm}"
    if tessdata_dir:
        config += f" --tessdata-dir {tessdata_dir}"

    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config)
    except Exception as exc:
        raise OcrError(f"Tesseract failed: {exc}") from exc

    return text.replace("\x0c", "").strip()
