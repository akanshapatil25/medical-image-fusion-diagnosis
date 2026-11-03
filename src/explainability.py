from __future__ import annotations

import numpy as np
import torch


def gradcam_heatmap(model, image_tensor, target_class=None, layer_name=None):
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a torch.Tensor")
    model.eval()
    image_tensor = image_tensor.clone().detach().requires_grad_(True)

    feature_maps = []
    grad_maps = []

    def _forward_hook(module, inputs, output):
        feature_maps.append(output)

    def _backward_hook(module, grad_inputs, grad_outputs):
        grad_maps.append(grad_outputs[0])

    layer = getattr(model, "gradcam_layer", None)
    if layer is None:
        raise AttributeError("model does not expose a gradcam_layer for Grad-CAM")

    handle_f = layer.register_forward_hook(_forward_hook)
    handle_b = layer.register_full_backward_hook(_backward_hook)
    try:
        logits = model(image_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        model.zero_grad()
        logits[:, target_class].backward(retain_graph=False)
    finally:
        handle_f.remove()
        handle_b.remove()

    if not feature_maps or not grad_maps:
        raise RuntimeError("Grad-CAM failed to capture feature and gradient maps")

    feature = feature_maps[0]
    grad = grad_maps[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    return cam.squeeze(1).detach().cpu().numpy()


def normalize_heatmap(array):
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return array
    array = array - array.min()
    denom = array.max() - array.min()
    if denom > 0:
        array /= denom
    return np.clip(array, 0.0, 1.0)
