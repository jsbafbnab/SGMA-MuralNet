"""Two-phase training for SGMA-MuralNet: phase 1 trains the SRN, phase 2 trains
the full SRN + CCN. Checkpoints are selected by the lowest validation hole MSE.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import DhMurals1714
from .losses import Criterion
from .model import build_discriminator, build_generator
from .utils import create_dir, load_checkpoint, save_checkpoint, set_seed


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SGMA-MuralNet")
    p.add_argument("--config", type=str, default=None, help="path to a YAML config")
    p.add_argument("--phase", type=int, default=1, choices=[1, 2])
    p.add_argument("--variant", type=str, default=None,
                   help="ablation variant A0-A8 (overrides config.VARIANT)")
    p.add_argument("--seed", type=int, default=None, help="random seed (overrides config.SEED)")
    p.add_argument("--resume", type=str, default=None, help="checkpoint to resume")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def _make_optimizer(params, lr, betas):
    return torch.optim.Adam(params, lr=lr, betas=betas)


@torch.no_grad()
def validate(generator, discriminator, loader, device, phase: int) -> float:
    """Return mean hole MSE over the validation set."""
    generator.eval()
    total_mse = 0.0
    count = 0
    for img, line, mask in loader:
        img = img.to(device); line = line.to(device); mask = mask.to(device)
        io = img * (1.0 - mask)
        js, b_is, _, jf, b_i = generator(io, line, mask, return_all=True)
        pred = b_is if phase == 1 else b_i
        err = ((pred - img) ** 2).mean(dim=1, keepdim=True)
        hole_err = (err * mask).sum() / mask.sum().clamp(min=1.0)
        total_mse += hole_err.item()
        count += 1
    generator.train()
    return total_mse / max(count, 1)


def run_phase(phase: int, config: Config, args: argparse.Namespace, device: torch.device):
    set_seed(config.SEED)
    out_dir = config.OUTPUT_DIR
    create_dir(out_dir)

    micro = config.PHASE1_MICRO if phase == 1 else config.PHASE2_MICRO
    effective = config.PHASE1_BATCH if phase == 1 else config.PHASE2_BATCH
    n_steps = config.PHASE1_STEPS if phase == 1 else config.PHASE2_STEPS
    assert effective % micro == 0, "effective batch must be a multiple of micro batch"
    acc_steps = effective // micro

    train_loader = DataLoader(DhMurals1714(config, "train"), batch_size=micro,
                              shuffle=True, num_workers=config.NUM_WORKERS,
                              drop_last=True)
    val_loader = DataLoader(DhMurals1714(config, "val"), batch_size=4,
                            shuffle=False, num_workers=config.NUM_WORKERS)

    generator = build_generator(config).to(device)
    discriminator = build_discriminator(config).to(device)
    criterion = Criterion(config).to(device)

    if phase == 1:
        # Phase 1 trains only the SRN subnetwork.
        gen_params = generator.srn.parameters()
    else:
        gen_params = generator.parameters()

    gen_opt = _make_optimizer(gen_params, config.GEN_LR, (config.BETA1, config.BETA2))
    disc_opt = _make_optimizer(discriminator.parameters(), config.DISC_LR,
                               (config.BETA1, config.BETA2))

    start_step = 0
    best_val = float("inf")
    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(args.resume, generator, discriminator,
                                     gen_opt, disc_opt, str(device))
        print(f"[train] resumed from {args.resume} at step {start_step}")

    # Phase 2 initializes from the best Phase-1 checkpoint.
    if phase == 2:
        ph1_best = os.path.join(out_dir, "phase1_best.pt")
        if os.path.exists(ph1_best):
            load_checkpoint(ph1_best, generator, discriminator, device=str(device))
            print(f"[train] phase 2 initialized from {ph1_best}")

    step = start_step
    print(f"[train] phase {phase}: {n_steps} steps, effective batch {effective} "
          f"(micro {micro} x acc {acc_steps}), gen_lr {config.GEN_LR}, disc_lr {config.DISC_LR}")

    data_iter = iter(train_loader)
    t0 = time.time()
    while step < n_steps:
        gen_opt.zero_grad()
        disc_opt.zero_grad()
        g_acc, d_acc = 0.0, 0.0

        for _ in range(acc_steps):
            try:
                img, line, mask = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                img, line, mask = next(data_iter)

            img = img.to(device); line = line.to(device); mask = mask.to(device)
            io = img * (1.0 - mask)                          # damaged observation

            js, b_is, r, jf, b_i = generator(io, line, mask, return_all=True)

            # discriminator step (detach generated images)
            d_real = discriminator(img)
            d_js = discriminator(js.detach())
            d_fakes = [d_js]
            if phase == 2:
                d_fakes.append(discriminator(b_i.detach()))
            d_loss = criterion.discriminator_loss(d_real, d_fakes)
            (d_loss / acc_steps).backward()
            d_acc += d_loss.item()

            # generator step (discriminator frozen)
            d_real_g = discriminator(img).detach()
            d_js_g = discriminator(js)
            d_fakes_g = [d_js_g]
            if phase == 2:
                d_fakes_g.append(discriminator(b_i))
            preds = [js] if phase == 1 else [js, b_i]
            g_logs = criterion.generator_loss(phase, d_real_g, d_fakes_g, preds,
                                              img, mask if phase == 2 else None)
            (g_logs["total"] / acc_steps).backward()
            g_acc += g_logs["total"].item()

        disc_opt.step()
        gen_opt.step()
        step += 1

        if step % config.LOG_INTERVAL == 0 or step == 1:
            dt = time.time() - t0
            print(f"[phase {phase}] step {step}/{n_steps}  "
                  f"g_loss {g_acc:.4f}  d_loss {d_acc:.4f}  "
                  f"({dt:.1f}s / {config.LOG_INTERVAL} steps)")
            t0 = time.time()

        if step % config.EVAL_INTERVAL == 0:
            val_mse = validate(generator, discriminator, val_loader, device, phase)
            print(f"[phase {phase}] step {step}: val hole MSE = {val_mse:.6f}")
            if val_mse < best_val:
                best_val = val_mse
                save_checkpoint(os.path.join(out_dir, f"phase{phase}_best.pt"),
                                step, generator, discriminator, gen_opt, disc_opt,
                                extra={"phase": phase, "val_hole_mse": val_mse})
                print(f"[phase {phase}] new best checkpoint saved "
                      f"(hole MSE {val_mse:.6f})")

        if step % config.SAVE_INTERVAL == 0:
            save_checkpoint(os.path.join(out_dir, f"phase{phase}_step{step}.pt"),
                            step, generator, discriminator, gen_opt, disc_opt,
                            extra={"phase": phase})

    save_checkpoint(os.path.join(out_dir, f"phase{phase}_final.pt"),
                    step, generator, discriminator, gen_opt, disc_opt,
                    extra={"phase": phase, "val_hole_mse": best_val})
    print(f"[train] phase {phase} finished. best val hole MSE = {best_val:.6f}")


def main():
    args = build_args()
    config = Config.from_yaml(args.config) if args.config else Config.from_defaults()
    if args.data_root:
        config.update({"DATA_ROOT": args.data_root})
    if args.output_dir:
        config.update({"OUTPUT_DIR": args.output_dir})
    if args.device:
        config.update({"DEVICE": args.device})
    if args.variant:
        config.update({"VARIANT": args.variant})
    if args.seed is not None:
        config.update({"SEED": args.seed})

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    run_phase(args.phase, config, args, device)


if __name__ == "__main__":
    main()
