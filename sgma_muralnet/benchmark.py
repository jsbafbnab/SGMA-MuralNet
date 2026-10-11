"""Efficiency measurement: MACs, latency, and peak GPU memory (batch 1, FP32, 256x256)."""
from __future__ import annotations

import time

import torch

from .model import build_generator


def count_macs(model: torch.nn.Module, size: int = 256) -> int:
    """Total multiply-accumulate operations for SRN + CCN (thop)."""
    try:
        from thop import profile
    except Exception:
        raise RuntimeError("thop is required for MAC counting; pip install thop")
    model.eval()
    device = next(model.parameters()).device
    io = torch.rand(1, 3, size, size, device=device)
    line = torch.rand(1, 1, size, size, device=device)
    mask = torch.zeros(1, 1, size, size, device=device)
    macs, _ = profile(model, inputs=(io, line, mask), verbose=False)
    return int(macs)


def measure_latency(model: torch.nn.Module, size: int = 256, warmup: int = 50,
                    repeats: int = 100) -> float:
    """Mean latency in milliseconds (batch size 1, FP32)."""
    device = next(model.parameters()).device
    model.eval()
    io = torch.rand(1, 3, size, size, device=device)
    line = torch.rand(1, 1, size, size, device=device)
    mask = torch.zeros(1, 1, size, size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(io, line, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(io, line, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / repeats * 1000.0


def measure_peak_memory(model: torch.nn.Module, size: int = 256) -> int:
    """Peak allocated GPU memory in MiB (isolates the algorithmic footprint)."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return 0
    model.eval()
    io = torch.rand(1, 3, size, size, device=device)
    line = torch.rand(1, 1, size, size, device=device)
    mask = torch.zeros(1, 1, size, size, device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(io, line, mask)
    return int(torch.cuda.max_memory_allocated() / (1024 * 1024))


def benchmark(config, device: torch.device) -> dict:
    generator = build_generator(config).to(device)
    macs = count_macs(generator, config.INPUT_SIZE)
    result = {
        "generator_params": sum(p.numel() for p in generator.parameters()),
        "macs": macs,                       # thop MAC count
        "flops": 2 * macs,                  # FLOPs (2x MACs)
        "latency_ms": measure_latency(generator, config.INPUT_SIZE),
        "peak_memory_mib": measure_peak_memory(generator, config.INPUT_SIZE),
    }
    return result


if __name__ == "__main__":
    import argparse
    from .config import Config
    p = argparse.ArgumentParser(description="Measure SGMA-MuralNet efficiency (Table 15)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--variant", type=str, default=None)
    a = p.parse_args()
    cfg = Config.from_yaml(a.config) if a.config else Config.from_defaults()
    if a.variant:
        cfg.update({"VARIANT": a.variant})
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    r = benchmark(cfg, dev)
    print(f"generator params : {r['generator_params']:,} ({r['generator_params']/1e6:.3f} M)")
    print(f"MACs (thop)      : {r['macs']:,} ({r['macs']/1e9:.2f} G)")
    print(f"FLOPs            : {r['flops']:,} ({r['flops']/1e9:.2f} G)")
    print(f"latency          : {r['latency_ms']:.2f} ms")
    print(f"peak GPU memory  : {r['peak_memory_mib']} MiB")
