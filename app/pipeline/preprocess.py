from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.config.settings import settings


def resize_for_processing(image: Image.Image, long_edge: int | None = None) -> Image.Image:
    limit = long_edge or settings.max_image_long_edge
    width, height = image.size
    scale = limit / max(width, height)
    if scale >= 1:
        return image
    return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)


def preprocess_printed(image: Image.Image) -> Image.Image:
    image = resize_for_processing(image)
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return Image.fromarray(enhanced)


def preprocess_handwritten(image: Image.Image) -> Image.Image:
    image = resize_for_processing(image)
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.convertScaleAbs(blurred, alpha=1.18, beta=8)
    return Image.fromarray(enhanced)


def preprocess_default(image: Image.Image) -> Image.Image:
    return resize_for_processing(image)
