from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def load_image(path: Path) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size == (250, 250):
            left = (250 - 160) // 2
            top = (250 - 160) // 2
            image = image.crop((left, top, left + 160, top + 160))
        image = image.resize((112, 112), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def save_png(pixels: Tensor, path: Path) -> None:
    display = pixels[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(display * 255).astype(np.uint8)).save(path)
