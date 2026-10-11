"""Sanity checks: forward shapes, parameter breakdown, and A0-A8 variants."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from sgma_muralnet.config import Config
from sgma_muralnet.model import build_discriminator, build_generator


def count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fmt(n: int) -> str:
    return f"{n:,} ({n / 1e6:.3f} M)"


def check_forward(cfg, variant, in_channels):
    cfg.update({"VARIANT": variant})
    gen = build_generator(cfg).eval()
    b, c, h, w = 1, 3, 256, 256
    io = torch.rand(b, c, h, w)
    line = torch.rand(b, 1, h, w)
    mask = (torch.rand(b, 1, h, w) > 0.5).float()
    with torch.no_grad():
        js, b_is, r, jf, b_i = gen(io, line, mask, return_all=True)
    assert tuple(js.shape) == (b, 3, h, w)
    assert tuple(b_i.shape) == (b, 3, h, w)
    assert -1.0 <= r.min() and r.max() <= 1.0, "R must be in [-1, 1]"
    known = 1.0 - mask
    assert torch.allclose(b_i * known, io * known, atol=1e-6), "observed pixels unchanged"
    assert gen.srn.stem.block[1].in_channels == in_channels, \
        f"variant {variant}: expected {in_channels} input channels"
    return count(gen)


def main():
    cfg = Config.from_defaults()
    gen = build_generator(cfg)
    disc = build_discriminator(cfg)

    line_encoder = count(gen.srn.line_encoder)
    heads = count(gen.srn.stage2.mod) + count(gen.srn.stage3.mod)
    mfa = count(gen.srn.adapter)
    se = count(gen.srn.adapter.se)
    branches = sum(count(b) for b in gen.srn.adapter.branches)
    proj = count(gen.srn.adapter.proj)
    added = line_encoder + heads + mfa

    print("Parameter accounting (A3 full model)")
    print(f"  line encoder            : {fmt(line_encoder)}")
    print(f"  conditioning heads      : {fmt(heads)}")
    print(f"  MFA dilated branches    : {fmt(branches)}")
    print(f"  MFA SE gate             : {fmt(se)}")
    print(f"  MFA residual projection : {fmt(proj)}")
    print(f"  added total             : {fmt(added)}")
    print(f"  generator (SRN + CCN)   : {fmt(count(gen))}")
    print(f"  discriminator           : {fmt(count(disc))}")
    print()

    assert line_encoder == 690176
    assert heads == 451520
    assert mfa == 664336
    assert added == 1806032

    # forward pass
    print("Forward-pass shapes (256x256)")
    b, c, h, w = 2, 3, 256, 256
    io = torch.rand(b, c, h, w)
    line = torch.rand(b, 1, h, w)
    mask = (torch.rand(b, 1, h, w) > 0.5).float()
    gen.eval()
    with torch.no_grad():
        js, b_is, r, jf, b_i = gen(io, line, mask, return_all=True)
    print(f"  Js : {tuple(js.shape)}  bIs : {tuple(b_is.shape)}  R : {tuple(r.shape)}  bI : {tuple(b_i.shape)}")
    print(f"  D(bI): {tuple(disc(b_i).shape)}")
    print()

    # ablation variants
    print("Ablation-variant generator param counts (SRN + CCN):")
    expected_input = {"A8": 3}
    for v in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"):
        ic = expected_input.get(v, 4)
        n = check_forward(cfg, v, ic)
        print(f"  {v}: {fmt(n)}")
    print()

    print("All checks passed.")


if __name__ == "__main__":
    main()
