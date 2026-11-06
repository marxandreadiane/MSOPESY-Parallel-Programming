"""Tesseract OCR wrapper."""

import os
import shutil
from pathlib import Path

import pytesseract
from PIL import Image

DEFAULT_LANG = "eng"
DEFAULT_PSM = 8
REPO_TESSDATA = Path(__file__).resolve().parents[1] / "models" / "tessdata"
WINDOWS_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


class OcrError(RuntimeError):
    """Raised when Tesseract fails."""


def find_tesseract() -> str | None:
    """Find the Tesseract binary: TESSERACT_CMD, then PATH, then the Windows default."""
    explicit = os.environ.get("TESSERACT_CMD")
    if explicit:
        return explicit
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    if WINDOWS_DEFAULT.is_file():
        return str(WINDOWS_DEFAULT)
    return None


def configure_tesseract(tessdata_dir: str | Path | None = None) -> None:
    """Point pytesseract at the binary and the language data."""
    command = find_tesseract()
    if command:
        pytesseract.pytesseract.tesseract_cmd = command

    if tessdata_dir is None and REPO_TESSDATA.is_dir():
        tessdata_dir = REPO_TESSDATA
    if tessdata_dir:
        os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)


def check_tesseract() -> str:
    """Return the Tesseract version, or raise OcrError."""
    configure_tesseract()
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
) -> str:
    """Run Tesseract on one image and return the extracted text."""
    configure_tesseract(tessdata_dir)

    try:
        text = pytesseract.image_to_string(
            image, lang=lang, config=f"--oem 3 --psm {psm}"
        )
    except Exception as exc:
        raise OcrError(f"Tesseract failed: {exc}") from exc

    return text.replace("\x0c", "").strip()
