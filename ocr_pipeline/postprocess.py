"""Text cleanup after OCR."""

import re


def clean_text(text: str) -> str:
    """Remove non-alphanumeric characters from both ends."""
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text)
