from dataclasses import dataclass

import cv2
import numpy as np

from src.preprocessing import image_to_float01


@dataclass
class RegistrationResult:
    aligned: np.ndarray
    estimated_shift_xy: tuple
    applied_shift_xy: tuple
    response: float
    success: bool


def _gray_float(image):
    arr = image_to_float01(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(arr.astype(np.float32))


def fourier_shift(image, shift_x, shift_y):
    """Translate a 2D array using the Fourier shift theorem (wraparound edges)."""
    array = np.asarray(image)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("fourier_shift expects a finite 2D image")
    height, width = array.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    phase = np.exp(-2j * np.pi * (fx * float(shift_x) + fy * float(shift_y)))
    shifted = np.fft.ifft2(np.fft.fft2(array) * phase).real
    return shifted.astype(np.float32)


def register_translation(reference, moving, min_response=0.01):
    """Estimate moving-to-reference translation; no rotation/deformation is modeled."""
    ref = _gray_float(reference)
    mov = _gray_float(moving)
    if ref.shape != mov.shape:
        raise ValueError(f"Registration requires equal image sizes; got {ref.shape} and {mov.shape}")
    if min(ref.shape) < 8:
        raise ValueError("Registration requires images at least 8x8")
    (dx, dy), response = cv2.phaseCorrelate(ref, mov)
    correction = (-float(dx), -float(dy))
    channels = np.asarray(moving)
    if channels.ndim == 2:
        aligned = fourier_shift(image_to_float01(channels), *correction)
    elif channels.ndim == 3:
        aligned = np.stack(
            [fourier_shift(image_to_float01(channels[..., c]), *correction) for c in range(channels.shape[2])],
            axis=-1,
        )
    else:
        raise ValueError("Moving image must be a 2D image or a 3D color image")
    response = float(response)
    return RegistrationResult(
        aligned=np.clip(aligned, 0, 1),
        estimated_shift_xy=(float(dx), float(dy)),
        applied_shift_xy=correction,
        response=response,
        success=bool(np.isfinite(response) and response >= min_response),
    )
