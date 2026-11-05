"""Image loading and preprocessing."""

from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps

MIN_SHORT_SIDE = 300
MAX_UPSCALE = 4
BORDER = 20
THRESHOLD = 128
DARK_CUTOFF = 128 

def load_image(path: str | Path) -> Image.Image:
    """Open an image as grayscale."""
    with Image.open(path) as image:
        return image.convert("L")


def preprocess_image(image: Image.Image) -> Image.Image:
    """Grayscale, upscale small images, denoise, binarize, crop to text, sharpen, add a white border."""
    image = ImageOps.autocontrast(image.convert("L"))

    width, height = image.size
    short_side = min(width, height)
    if short_side and short_side < MIN_SHORT_SIDE:
        factor = min(MAX_UPSCALE, max(1, round(MIN_SHORT_SIDE / short_side)))
        image = image.resize(
            (width * factor, height * factor),
            Image.Resampling.LANCZOS,
        )

    image = image.filter(ImageFilter.MedianFilter(3))
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda pixel: 255 if pixel >= THRESHOLD else 0)

    mask = image.point(lambda p: 255 if p < DARK_CUTOFF else 0)
    bbox = mask.getbbox()
    if bbox:
        pad = 10
        left, top, right, bottom = bbox
        image = image.crop((
            max(0, left - pad),
            max(0, top - pad),
            min(image.width, right + pad),
            min(image.height, bottom + pad),
        ))

    return ImageOps.expand(image, border=BORDER, fill=255)
