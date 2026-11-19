from pathlib import Path

import cv2
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(path: str | Path) -> list[Path]:
    path = Path(path)
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if path.is_dir() else [path]


def read_image(path: str | Path, max_pixels: int = 16777216):
    path = Path(path)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image format. Use JPEG, PNG, WebP or BMP.")
    try:
        with Image.open(path) as header:
            if header.width * header.height > max_pixels:
                raise ValueError(f"Image exceeds the {max_pixels:,} pixel limit.")
    except (UnidentifiedImageError, Image.DecompressionBombError):
        raise ValueError("Cannot decode image.") from None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image.")
    return image
