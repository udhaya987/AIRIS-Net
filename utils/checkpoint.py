import os
from pathlib import Path
import torch
from typing import Dict, Any, Optional


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool = False,
    checkpoint_dir: str = "checkpoints",
    filename: Optional[str] = None
) -> Path:
    """
    Save training checkpoint.
    """
    chk_dir = Path(checkpoint_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        epoch = state.get("epoch", 0)
        filename = f"airis_epoch_{epoch:03d}.pth"

    filepath = chk_dir / filename
    torch.save(state, str(filepath))

    if is_best:
        best_path = chk_dir / "best_airis.pth"
        torch.save(state, str(best_path))
        print(f"[Checkpoint] Saved new best model checkpoint to {best_path}")

    return filepath


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Load checkpoint and resume model, optimizer, scheduler states.
    """
    chk_path = Path(checkpoint_path)
    if not chk_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {chk_path}")

    print(f"[Checkpoint] Loading checkpoint from {chk_path} on {device}")
    checkpoint = torch.load(str(chk_path), map_location=device)

    # Handle model state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_psnr = checkpoint.get("best_psnr", 0.0)
    print(f"[Checkpoint] Resumed from epoch {epoch} (best PSNR: {best_psnr:.2f} dB)")

    return checkpoint
