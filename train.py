import os
import sys
from pathlib import Path
import argparse
import time
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from airis.model import AIRISNet
from airis.losses import AIRISLoss
from data.dataset import create_dataloader, prepare_dataset_splits, prepare_real_paired_splits
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import calculate_psnr, calculate_ssim


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0
) -> dict:
    model.train()
    total_loss_accum = 0.0
    loss_components_accum = {
        "char": 0.0, "edge": 0.0, "ssim": 0.0, "freq": 0.0,
        "identity": 0.0, "mask": 0.0, "reliability": 0.0
    }
    num_batches = len(dataloader)
    if num_batches == 0:
        return {"total": 0.0, **loss_components_accum}

    for step, batch in enumerate(dataloader):
        degraded = batch["degraded"].to(device)
        clean = batch["clean"].to(device)

        optimizer.zero_grad()
        outputs = model(degraded)
        loss, loss_dict = loss_fn(outputs, clean, degraded)

        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        optimizer.step()

        total_loss_accum += loss_dict["total"]
        for k in loss_components_accum:
            loss_components_accum[k] += loss_dict.get(k, 0.0)

        if (step + 1) % 25 == 0 or (step + 1) == num_batches:
            print(
                f"  Step [{step+1:3d}/{num_batches:3d}] | "
                f"Loss: {loss_dict['total']:.4f} "
                f"(Char: {loss_dict.get('char', 0.0):.4f}, SSIM: {loss_dict.get('ssim', 0.0):.4f})",
                flush=True
            )

    avg_loss = total_loss_accum / max(1, num_batches)
    avg_components = {k: v / max(1, num_batches) for k, v in loss_components_accum.items()}
    avg_components["total"] = avg_loss
    return avg_components


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_val_batches: int = 50
) -> dict:
    model.eval()
    total_loss_accum = 0.0
    psnr_accum = 0.0
    ssim_accum = 0.0
    num_batches = min(len(dataloader), max_val_batches)
    total_evaluated_samples = 0

    if num_batches == 0:
        return {"val_loss": 0.0, "psnr": 0.0, "ssim": 0.0}

    for idx, batch in enumerate(dataloader):
        if idx >= num_batches:
            break
        degraded = batch["degraded"].to(device)
        clean = batch["clean"].to(device)

        outputs = model(degraded)
        loss, _ = loss_fn(outputs, clean, degraded)
        total_loss_accum += loss.item()

        restored = outputs["restored"].float()
        # Compute metrics per sample
        for b in range(restored.size(0)):
            res_np = restored[b].cpu().clamp(0.0, 1.0).numpy()
            clean_np = clean[b].cpu().numpy()
            psnr_accum += calculate_psnr(res_np, clean_np)
            ssim_accum += calculate_ssim(res_np, clean_np)
            total_evaluated_samples += 1

    avg_loss = total_loss_accum / max(1, num_batches)
    avg_psnr = psnr_accum / max(1, total_evaluated_samples)
    avg_ssim = ssim_accum / max(1, total_evaluated_samples)

    return {
        "val_loss": avg_loss,
        "psnr": avg_psnr,
        "ssim": avg_ssim
    }


def main():
    if hasattr(os, "cpu_count") and os.cpu_count():
        torch.set_num_threads(os.cpu_count())

    parser = argparse.ArgumentParser(description="Train AIRIS-Net for Semiconductor Image Restoration")
    parser.add_argument("--config", type=str, default="configs/real_train.yaml", help="Path to config yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--scale", type=int, default=None, help="Scale factor (1 for denoising, 2 for 2x super-resolution)")
    parser.add_argument("--degradation_mode", type=str, default=None, help="Override degradation mode")
    parser.add_argument("--paired", action="store_true", help="Force real paired dataset training")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic random seed")
    parser.add_argument("--max_train_samples", "--max-train-samples", dest="max_train_samples", type=int, default=None, help="Limit train samples for fast training")
    parser.add_argument("--max_val_samples", "--max-val-samples", dest="max_val_samples", type=int, default=None, help="Limit val samples for fast validation")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Set seed
    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    # Hardware detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Override parameters if provided via CLI
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    scale_factor = args.scale if args.scale is not None else cfg["model"].get("scale_factor", 2)
    degradation_mode = args.degradation_mode or cfg["data"].get("degradation_mode", "random")

    lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    patch_size = cfg["data"].get("patch_size", None)
    num_workers = cfg["training"]["num_workers"]
    save_every = cfg["training"]["save_every"]
    validate_every = cfg["training"]["validate_every"]
    gradient_clip = float(cfg["training"]["gradient_clip"])

    is_paired_mode = args.paired or cfg["data"].get("mode") == "paired" or "noisy_train_dir" in cfg["data"]

    if is_paired_mode:
        print("[AIRIS-Net] Using Real Paired (NoisyLR, GT) Dataset Pipeline.")
        # Ensure paired dataset directories exist
        prepare_real_paired_splits(
            noisy_source_dir="train/train/NoisyLR",
            gt_source_dir="train/train/GT",
            base_data_dir="data",
            seed=seed
        )

        train_loader = create_dataloader(
            noisy_dir=cfg["data"].get("noisy_train_dir", "data/real_train/NoisyLR"),
            gt_dir=cfg["data"].get("gt_train_dir", "data/real_train/GT"),
            batch_size=batch_size,
            patch_size=patch_size,
            scale_factor=scale_factor,
            is_train=True,
            grayscale=cfg["data"].get("grayscale", True),
            num_workers=num_workers,
            max_samples=args.max_train_samples,
            seed=seed,
            is_paired=True
        )

        val_loader = create_dataloader(
            noisy_dir=cfg["data"].get("noisy_val_dir", "data/real_val/NoisyLR"),
            gt_dir=cfg["data"].get("gt_val_dir", "data/real_val/GT"),
            batch_size=batch_size,
            patch_size=patch_size,
            scale_factor=scale_factor,
            is_train=False,
            grayscale=cfg["data"].get("grayscale", True),
            num_workers=num_workers,
            max_samples=args.max_val_samples,
            seed=seed + 1,
            is_paired=True
        )
    else:
        print("[AIRIS-Net] Using Synthetic Degradation Pipeline.")
        prepare_dataset_splits(source_dir="train/train/GT", base_data_dir="data", seed=seed)

        train_loader = create_dataloader(
            clean_dir=cfg["data"]["train_dir"],
            batch_size=batch_size,
            patch_size=patch_size or 128,
            scale_factor=scale_factor,
            is_train=True,
            grayscale=cfg["data"]["grayscale"],
            num_workers=num_workers,
            degradation_mode=degradation_mode,
            max_samples=args.max_train_samples,
            seed=seed
        )

        val_loader = create_dataloader(
            clean_dir=cfg["data"]["val_dir"],
            batch_size=batch_size,
            patch_size=patch_size or 128,
            scale_factor=scale_factor,
            is_train=False,
            grayscale=cfg["data"]["grayscale"],
            num_workers=num_workers,
            degradation_mode=degradation_mode,
            max_samples=args.max_val_samples,
            seed=seed + 1
        )

    # Initialize AIRIS-Net
    model_cfg = cfg["model"]
    model = AIRISNet(
        in_channels=model_cfg.get("in_channels", 1),
        base_channels=model_cfg.get("base_channels", 48),
        degradation_dim=model_cfg.get("degradation_dim", 64),
        scale_factor=scale_factor,
        use_local_expert=model_cfg.get("use_local_expert", True),
        use_global_expert=model_cfg.get("use_global_expert", True),
        use_frequency_expert=model_cfg.get("use_frequency_expert", True),
        use_adaptive_routing=model_cfg.get("use_adaptive_routing", True),
        use_integrity_mask=model_cfg.get("use_integrity_mask", True)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[AIRIS-Net] Initialized model with {total_params:,} trainable parameters (scale={scale_factor}x).")

    # Initialize Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    # Initialize Loss
    loss_cfg = cfg["loss"]
    loss_fn = AIRISLoss(
        w_char=loss_cfg.get("w_char", 1.5),
        w_edge=loss_cfg.get("w_edge", 0.4),
        w_ssim=loss_cfg.get("w_ssim", 0.8),
        w_freq=loss_cfg.get("w_freq", 0.15),
        w_identity=loss_cfg.get("w_identity", 0.01),
        w_mask=loss_cfg.get("w_mask", 0.05),
        w_reliability=loss_cfg.get("w_reliability", 0.05),
        k_reliability=loss_cfg.get("k_reliability", 10.0)
    ).to(device)

    start_epoch = 1
    best_psnr = 0.0
    best_ssim = 0.0
    best_score = 0.0

    # Resume if requested
    if args.resume:
        chk = load_checkpoint(args.resume, model, optimizer, scheduler, device=device)
        start_epoch = chk.get("epoch", 0) + 1
        best_psnr = chk.get("best_psnr", 0.0)
        best_ssim = chk.get("best_ssim", chk.get("val_ssim", 0.0))
        best_score = best_ssim * 20.0 + best_psnr

    chk_dir = cfg["paths"]["checkpoint_dir"]
    Path(chk_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Starting AIRIS-Net Training: Epochs {start_epoch} to {epochs}")
    print(f"Dataset Mode: {'Real Paired' if is_paired_mode else 'Synthetic'} | Scale: {scale_factor}x | Batch Size: {batch_size}")
    print("=" * 60)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            gradient_clip=gradient_clip
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"\n--- Epoch {epoch}/{epochs} ({elapsed:.1f}s) ---")
        print(
            f"Train Loss: {train_metrics['total']:.4f} | "
            f"Char: {train_metrics['char']:.4f} | "
            f"Edge: {train_metrics['edge']:.4f} | "
            f"SSIM: {train_metrics['ssim']:.4f} | "
            f"Freq: {train_metrics['freq']:.4f} | "
            f"Id: {train_metrics['identity']:.4f} | "
            f"Mask: {train_metrics['mask']:.4f} | "
            f"Rel: {train_metrics['reliability']:.4f}"
        )
        print(f"Learning Rate: {current_lr:.6f}")

        # Validate
        if epoch % validate_every == 0 or epoch == epochs:
            val_metrics = validate(model, val_loader, loss_fn, device)
            val_loss = val_metrics["val_loss"]
            psnr = val_metrics["psnr"]
            ssim = val_metrics["ssim"]
            current_score = ssim * 20.0 + psnr

            print(f"Val Loss: {val_loss:.4f} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

            # Check for best checkpoint
            is_best = current_score > best_score or ssim > best_ssim or psnr > best_psnr
            if is_best:
                best_score = max(best_score, current_score)
                best_psnr = max(best_psnr, psnr)
                best_ssim = max(best_ssim, ssim)

            # Save state
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "val_psnr": psnr,
                "val_ssim": ssim,
                "scale_factor": scale_factor,
                "config": cfg
            }

            save_checkpoint(state, is_best=is_best, checkpoint_dir=chk_dir, filename=f"airis_epoch_{epoch:03d}.pth")
            if is_best:
                # Also save a scale-specific best checkpoint for clarity
                save_checkpoint(state, is_best=False, checkpoint_dir=chk_dir, filename=f"airis_x{scale_factor}_best.pth")

    print("\n" + "=" * 60)
    print(f"Training Complete! Best Validation SSIM: {best_ssim:.4f}, Best PSNR: {best_psnr:.2f} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()
