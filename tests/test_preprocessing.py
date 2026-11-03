from PIL import Image
import numpy as np

from src.preprocessing import image_to_float01, preprocess_image


def test_image_to_float01_range():
    array = np.array([[0, 128], [255, 255]], dtype=np.uint8)
    result = image_to_float01(Image.fromarray(array))
    assert result.shape == array.shape
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_image_resizes_and_validates():
    image = Image.new("L", (32, 16), color=128)
    resized, arr = preprocess_image(image, size=(16, 16), grayscale=True)
    assert resized.size == (16, 16)
    assert arr.shape == (16, 16)
    assert np.isfinite(arr).all()
