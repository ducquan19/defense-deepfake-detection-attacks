from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
from PIL import Image

from core.base import BaseDefense


def jpeg_smoothing_batch(images: torch.Tensor, quality: int = 75) -> torch.Tensor:
    smoothed = [_jpeg_roundtrip(image.detach().cpu(), quality) for image in images]
    return torch.stack(smoothed).to(images.device)


def _jpeg_roundtrip(image: torch.Tensor, quality: int) -> torch.Tensor:
    array = image.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    pil_image = Image.fromarray(array, mode="RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    decoded_array = np.asarray(decoded, dtype=np.float32) / 255.0
    return torch.from_numpy(decoded_array).permute(2, 0, 1)


class JPEGSmoothingDefense(BaseDefense):
    name = "jpeg_smoothing"

    def __init__(self, quality: int = 75) -> None:
        self.quality = quality

    def apply(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        quality = int(kwargs.get("quality", self.quality))
        return jpeg_smoothing_batch(images, quality=quality)
