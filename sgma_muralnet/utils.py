"""Small helpers: seeding, checkpointing, image saving."""
from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, iteration: int, generator: torch.nn.Module,
                    discriminator: torch.nn.Module,
                    gen_optimizer, disc_optimizer, extra: Dict | None = None) -> None:
    create_dir(os.path.dirname(path) or ".")
    payload = {
        "iteration": iteration,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "gen_optimizer": gen_optimizer.state_dict() if gen_optimizer else None,
        "disc_optimizer": disc_optimizer.state_dict() if disc_optimizer else None,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, generator: torch.nn.Module,
                    discriminator: torch.nn.Module,
                    gen_optimizer=None, disc_optimizer=None,
                    device: str = "cpu") -> int:
    data = torch.load(path, map_location=device)
    generator.load_state_dict(_strip_module(data["generator"]))
    if (discriminator is not None and "discriminator" in data
            and data["discriminator"] is not None):
        discriminator.load_state_dict(_strip_module(data["discriminator"]))
    if gen_optimizer is not None and data.get("gen_optimizer"):
        gen_optimizer.load_state_dict(data["gen_optimizer"])
    if disc_optimizer is not None and data.get("disc_optimizer"):
        disc_optimizer.load_state_dict(data["disc_optimizer"])
    return data.get("iteration", 0)


def _strip_module(state: Dict) -> Dict:
    """Remove a leading ``module.`` prefix added by DataParallel."""
    out = {}
    for k, v in state.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """(3, H, W) or (1, H, W) tensor in [0, 1] -> PIL RGB image."""
    t = t.detach().cpu().clamp(0, 1)
    if t.ndim == 3 and t.size(0) == 1:
        t = t.repeat(3, 1, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def save_image(t: torch.Tensor, path: str) -> None:
    create_dir(os.path.dirname(path) or ".")
    tensor_to_image(t).save(path)
