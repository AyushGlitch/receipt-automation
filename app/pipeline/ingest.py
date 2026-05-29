from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from app.models.schemas import ReceiptInput

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def discover_inputs(path: Path) -> list[ReceiptInput]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return [ReceiptInput(source_path=file) for file in files]


def load_image(receipt_input: ReceiptInput) -> Image.Image:
    image = Image.open(receipt_input.source_path)
    image = ImageOps.exif_transpose(image)
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    return image
