from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_image(source, grayscale=True) -> Image.Image:
    """Load a path, PIL image, or byte stream as a detached PIL image."""
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        elif isinstance(source, (str, Path)):
            image = Image.open(source)
        else:
            image = Image.open(source)
        image.load()
        image = ImageOps.exif_transpose(image)
        return image.convert("L" if grayscale else "RGB")
    except Exception as exc:
        raise ValueError(f"Unable to load image: {exc}") from exc


def image_to_float01(image) -> np.ndarray:
    """Return a finite grayscale/RGB float32 array in [0, 1]."""
    arr = np.asarray(image)
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (1, 3, 4)):
        raise ValueError(f"Expected 2D grayscale or RGB image, got shape {arr.shape}")
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("Image is empty or contains non-finite values")
    arr = arr.astype(np.float32)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        scale = float(np.iinfo(np.asarray(image).dtype).max)
        arr /= scale
    elif arr.min() < 0 or arr.max() > 1:
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return np.clip(arr, 0.0, 1.0)


def preprocess_image(
    source,
    size=(224, 224),
    grayscale=True,
    clahe=False,
    denoise=None,
):
    """Load, optionally enhance, resize, and return a PIL image plus float array."""
    if len(size) != 2 or min(size) < 1:
        raise ValueError("size must be a pair of positive integers")
    image = load_image(source, grayscale=grayscale)
    if clahe:
        if not grayscale:
            raise ValueError("CLAHE is supported only for grayscale inputs")
        gray = np.asarray(image)
        image = Image.fromarray(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray))
    if denoise not in (None, "gaussian", "median"):
        raise ValueError("denoise must be None, 'gaussian', or 'median'")
    if denoise == "gaussian":
        image = Image.fromarray(cv2.GaussianBlur(np.asarray(image), (3, 3), 0))
    elif denoise == "median":
        image = Image.fromarray(cv2.medianBlur(np.asarray(image), 3))
    resized = image.resize((int(size[1]), int(size[0])), Image.Resampling.BILINEAR)
    return resized, image_to_float01(resized)


def image_to_tensor(image, size=(224, 224), grayscale=True) -> torch.Tensor:
    """Convert input image to a normalized CHW tensor in [0, 1]."""
    _, arr = preprocess_image(image, size=size, grayscale=grayscale)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def save_preprocessed(source, destination, size=(224, 224), clahe=False, denoise=None):
    image, _ = preprocess_image(source, size=size, clahe=clahe, denoise=denoise)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination
