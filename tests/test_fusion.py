import numpy as np

from src.fusion import weighted_fusion


def test_weighted_fusion_valid_range():
    a = np.ones((16, 16), dtype=np.float32)
    b = np.zeros((16, 16), dtype=np.float32)
    fused, registration = weighted_fusion(a, b, alpha=0.3, register=False)
    assert fused.shape == a.shape
    assert fused.min() >= 0.0
    assert fused.max() <= 1.0
    assert registration is None
