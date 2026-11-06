"""Image loading and preprocessing."""

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

UPSCALE = 5
THRESHOLD = 128
DARK_CUTOFF = 128
CROP_PAD = 10


def load_image(path: str | Path) -> Image.Image:
    """Open an image and detach it from the file."""
    with Image.open(path) as image:
        return image.copy()


def preprocess_image(image: Image.Image) -> Image.Image:
    """Grayscale, upscale 5x, denoise, binarize, and crop to the text."""
    image = ImageOps.autocontrast(image.convert("L"))

    width, height = image.size
    image = image.resize(
        (width * UPSCALE, height * UPSCALE),
        Image.Resampling.LANCZOS,
    )

    image = image.filter(ImageFilter.MedianFilter(3))
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda pixel: 255 if pixel >= THRESHOLD else 0)

    mask = image.point(lambda pixel: 255 if pixel < DARK_CUTOFF else 0)
    bbox = mask.getbbox()
    if bbox:
        left, top, right, bottom = bbox
        image = image.crop(
            (
                max(0, left - CROP_PAD),
                max(0, top - CROP_PAD),
                min(image.width, right + CROP_PAD),
                min(image.height, bottom + CROP_PAD),
            )
        )

    return image
