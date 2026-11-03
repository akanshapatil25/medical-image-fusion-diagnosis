import numpy as np

from src.registration import register_translation


def test_register_translation_on_shifted_image():
    base = np.zeros((32, 32), dtype=np.float32)
    base[8:24, 8:24] = 1.0
    shifted = np.zeros_like(base)
    shifted[10:26, 10:26] = 1.0
    result = register_translation(base, shifted, min_response=0.01)
    assert result.success
    assert result.aligned.shape == base.shape
    assert np.isfinite(result.aligned).all()
