import numpy as np

from src.preprocessing import image_to_float01
from src.registration import register_translation


def weighted_fusion(image_a, image_b, alpha=0.5, register=False):
    """Fuse aligned, intensity-normalized images; this is a display baseline only."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    first = image_to_float01(image_a)
    second = image_to_float01(image_b)
    if first.shape != second.shape:
        raise ValueError(f"Fusion requires matching shapes; got {first.shape} and {second.shape}")
    registration = register_translation(first, second) if register else None
    if registration is not None:
        second = registration.aligned
    fused = alpha * first + (1.0 - alpha) * second
    return np.clip(fused, 0, 1).astype(np.float32), registration
