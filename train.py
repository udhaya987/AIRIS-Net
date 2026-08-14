import os
import sys
from pathlib import Path
import argparse
import time
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from airis.model import AIRISNet
from airis.losses import AIRISLoss
from data.dataset import create_dataloader, prepare_dataset_splits
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import calculate_psnr, calculate_ssim


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

    avg_loss = total_loss_accum / max(1, num_batches)
    avg_components = {k: v / max(1, num_batches) for k, v in loss_components_accum.items()}
    avg_components["total"] = avg_loss
    return avg_components


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> dict:
    model.eval()
    total_loss_accum = 0.0
    psnr_accum = 0.0
    ssim_accum = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        degraded = batch["degraded"].to(device)
        clean = batch["clean"].to(device)

        outputs = model(degraded)
        loss, _ = loss_fn(outputs, clean, degraded)
        total_loss_accum += loss.item()

        restored = outputs["restored"]
        # Compute metrics per sample
        for b in range(restored.size(0)):
            res_np = restored[b].cpu().clamp(0.0, 1.0).numpy()
            clean_np = clean[b].cpu().numpy()
            psnr_accum += calculate_psnr(res_np, clean_np)
            ssim_accum += calculate_ssim(res_np, clean_np)

    total_samples = num_batches * dataloader.batch_size
    avg_loss = total_loss_accum / max(1, num_batches)
    avg_psnr = psnr_accum / max(1, total_samples)
    avg_ssim = ssim_accum / max(1, total_samples)

    return {
        "val_loss": avg_loss,
        "psnr": avg_psnr,
        "ssim": avg_ssim
    }


def main():
    parser = argparse.ArgumentParser(description="Train AIRIS-Net for Semiconductor Image Restoration")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to config yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max_train_samples", "--max-train-samples", dest="max_train_samples", type=int, default=None, help="Limit train samples for fast testing")
    parser.add_argument("--max_val_samples", "--max-val-samples", dest="max_val_samples", type=int, default=None, help="Limit val samples for fast testing")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

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
    lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    patch_size = cfg["data"]["patch_size"]
    num_workers = cfg["training"]["num_workers"]
    save_every = cfg["training"]["save_every"]
    validate_every = cfg["training"]["validate_every"]
    gradient_clip = float(cfg["training"]["gradient_clip"])

    # Ensure dataset directories exist
    prepare_dataset_splits(source_dir="train/train/GT", base_data_dir="data")

    # Create DataLoaders
    train_loader = create_dataloader(
        clean_dir=cfg["data"]["train_dir"],
        batch_size=batch_size,
        patch_size=patch_size,
        is_train=True,
        grayscale=cfg["data"]["grayscale"],
        num_workers=num_workers,
        degradation_mode=cfg["data"]["degradation_mode"],
        max_samples=args.max_train_samples
    )

    val_loader = create_dataloader(
        clean_dir=cfg["data"]["val_dir"],
        batch_size=batch_size,
        patch_size=patch_size,
        is_train=False,
        grayscale=cfg["data"]["grayscale"],
        num_workers=num_workers,
        degradation_mode=cfg["data"]["degradation_mode"],
        max_samples=args.max_val_samples
    )

    # Initialize AIRIS-Net
    model_cfg = cfg["model"]
    model = AIRISNet(
        in_channels=model_cfg.get("in_channels", 1),
        base_channels=model_cfg.get("base_channels", 48),
        degradation_dim=model_cfg.get("degradation_dim", 64),
        use_local_expert=model_cfg.get("use_local_expert", True),
        use_global_expert=model_cfg.get("use_global_expert", True),
        use_frequency_expert=model_cfg.get("use_frequency_expert", True),
        use_adaptive_routing=model_cfg.get("use_adaptive_routing", True),
        use_integrity_mask=model_cfg.get("use_integrity_mask", True)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[AIRIS-Net] Initialized model with {total_params:,} trainable parameters.")

    # Initialize Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Initialize Loss
    loss_cfg = cfg["loss"]
    loss_fn = AIRISLoss(
        w_char=loss_cfg.get("w_char", 1.0),
        w_edge=loss_cfg.get("w_edge", 0.1),
        w_ssim=loss_cfg.get("w_ssim", 0.1),
        w_freq=loss_cfg.get("w_freq", 0.05),
        w_identity=loss_cfg.get("w_identity", 0.1),
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
        best_ssim = chk.get("val_ssim", 0.0)
        best_score = best_ssim * 20.0 + best_psnr

    chk_dir = cfg["paths"]["checkpoint_dir"]
    Path(chk_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Starting AIRIS-Net Training: Epochs {start_epoch} to {epochs}")
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
        print(f"Train Loss: {train_metrics['total']:.4f} | Char: {train_metrics['char']:.4f} | Edge: {train_metrics['edge']:.4f} | SSIM: {train_metrics['ssim']:.4f} | Freq: {train_metrics['freq']:.4f} | Id: {train_metrics['identity']:.4f} | Mask: {train_metrics['mask']:.4f} | Rel: {train_metrics['reliability']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        # Validate
        if epoch % validate_every == 0 or epoch == epochs:
            val_metrics = validate(model, val_loader, loss_fn, device)
            val_loss = val_metrics["val_loss"]
            psnr = val_metrics["psnr"]
            ssim = val_metrics["ssim"]
            current_score = ssim * 20.0 + psnr

            print(f"Val Loss: {val_loss:.4f} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} (Accuracy: {ssim*100:.1f}%)")

            # Check for best checkpoint
            is_best = current_score > best_score or ssim > best_ssim
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
                "config": cfg
            }

            if is_best:
                save_checkpoint(state, is_best=True, checkpoint_dir=chk_dir, filename=f"airis_epoch_{epoch:03d}.pth")
            elif epoch % save_every == 0:
                save_checkpoint(state, is_best=False, checkpoint_dir=chk_dir, filename=f"airis_epoch_{epoch:03d}.pth")

    print("\n" + "=" * 60)
    print(f"Training Complete! Best Validation SSIM: {best_ssim:.4f} ({best_ssim*100:.1f}%), Best PSNR: {best_psnr:.2f} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()
