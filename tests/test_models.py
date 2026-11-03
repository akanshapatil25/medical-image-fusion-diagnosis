import torch

from models import LiteBrainNet


def test_litebrainnet_output_shape():
    model = LiteBrainNet(num_classes=4, input_channels=1, dropout=0.1, channels=(8, 16, 24, 32))
    tensor = torch.randn(2, 1, 224, 224)
    logits = model(tensor)
    assert logits.shape == (2, 4)
