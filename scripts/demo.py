from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.fusion import weighted_fusion
from src.preprocessing import image_to_tensor


def main():
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = 224, 224
    base = np.zeros((height, width), dtype=np.float32)
    base[60:160, 60:160] = 1.0
    second = np.zeros_like(base)
    second[100:200, 100:200] = 1.0
    fused, _ = weighted_fusion(base, second, alpha=0.5, register=False)
    output = out_dir / "fusion_demo.png"
    Image.fromarray((fused * 255).astype(np.uint8)).save(output)
    print(f"Saved illustrative fusion demo to {output}")


if __name__ == "__main__":
    main()
