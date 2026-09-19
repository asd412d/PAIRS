from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import Tensor


BASIS_SHA256 = "df3dc57bb8acdfabe7cd93e03f709a2dc66d3be54ede06efabb50755e8502f49"
IMAGE_SHAPE = (3, 112, 112)
BASE_RGB_RMS = 10.0 ** 0.1


def load_basis(path: Path) -> Tensor:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != BASIS_SHA256:
        raise ValueError("The IRSE-100 basis file does not match the published artifact")
    basis = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(basis, Tensor) or tuple(basis.shape) != (512, *IMAGE_SHAPE):
        raise ValueError("Expected an IRSE-100 basis of shape [512,3,112,112]")
    if not basis.is_floating_point() or not torch.isfinite(basis).all():
        raise ValueError("The basis must contain finite floating-point values")
    return basis.float().contiguous()


def protect(pixels: Tensor, basis: Tensor, *, alpha: float, seed: int) -> Tensor:
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != IMAGE_SHAPE:
        raise ValueError("Expected RGB input with shape [N,3,112,112]")
    if basis.ndim != 4 or tuple(basis.shape[1:]) != IMAGE_SHAPE or len(basis) < 1:
        raise ValueError("Expected a basis with shape [K,3,112,112]")
    if not pixels.is_floating_point() or not torch.isfinite(pixels).all():
        raise ValueError("Input pixels must be finite floating-point values")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if alpha == 0:
        return pixels.clone()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    coefficients = torch.randn(
        len(pixels), len(basis), generator=generator, dtype=torch.float32
    ).to(pixels.device)
    basis = basis.to(device=pixels.device, dtype=torch.float32)
    native_delta = torch.einsum("bk,kchw->bchw", coefficients, basis)
    rgb_delta = native_delta / 2.0
    rms = rgb_delta.flatten(1).square().mean(1).sqrt().clamp_min(1e-12)
    target_rms = alpha * BASE_RGB_RMS
    rgb_delta = rgb_delta * (target_rms / rms)[:, None, None, None]
    return pixels + rgb_delta
