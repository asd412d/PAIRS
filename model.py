from __future__ import annotations

import contextlib
from pathlib import Path

import frbench
import torch
import torch.nn.functional as F
import yaml
from frbench.backbones import build_backbone
from torch import Tensor, nn


AUTHORIZED = "irse-100_arcface_ms1m"
UNAUTHORIZED = "swinmlp-b_arcface_ms1m"


class Recognizer(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)

    def forward(self, rgb_pixels: Tensor) -> Tensor:
        model_input = rgb_pixels * 2.0 - 1.0
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if rgb_pixels.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with context:
            embedding = self.backbone(model_input)
        return F.normalize(embedding.float(), p=2, dim=1)


def load_recognizer(name: str, weights_root: Path, device: torch.device) -> Recognizer:
    if name not in {AUTHORIZED, UNAUTHORIZED}:
        raise ValueError(f"Unsupported recognizer: {name}")
    with frbench.configure_scoped(
        cache=str(weights_root),
        repo="HKU-TASR/FRBench",
        release="weights-v1.0.0",
        update_check=False,
    ):
        asset = frbench.get_asset(name)
    model_dir = Path(asset["path"])
    config_path = model_dir / asset["contents"]["config"]
    checkpoint_path = model_dir / asset["contents"]["weight"]
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    settings = config["input"]
    divisor = settings.get("var", settings.get("std"))
    if (
        tuple(settings["size"]) != (112, 112)
        or settings.get("channel", "rgb") != "rgb"
        or tuple(settings["mean"]) != (0.5, 0.5, 0.5)
        or tuple(divisor) != (0.5, 0.5, 0.5)
    ):
        raise ValueError(f"Unexpected FRBench preprocessing: {config_path}")
    backbone_spec = config["backbone"]
    backbone = build_backbone(
        backbone_spec["name"], backbone_spec.get("kwargs", {}), device
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    return Recognizer(backbone).eval().to(device)
