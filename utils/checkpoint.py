import os
from pathlib import Path
import torch
from typing import Dict, Any, Optional, Union


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool = False,
    checkpoint_dir: str = "checkpoints",
    filename: Optional[str] = None
) -> Path:
    """
    Save training checkpoint.
    Saves the specific epoch snapshot, updates latest_airis.pth,
    and updates best_airis.pth when is_best is True.
    """
    chk_dir = Path(checkpoint_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)

    epoch = state.get("epoch", 0)
    if filename is None:
        filename = f"airis_epoch_{epoch:03d}.pth"

    filepath = chk_dir / filename
    torch.save(state, str(filepath))

    # Always save latest
    latest_path = chk_dir / "latest_airis.pth"
    torch.save(state, str(latest_path))

    # Save best if flagged
    if is_best:
        best_path = chk_dir / "best_airis.pth"
        torch.save(state, str(best_path))
        print(f"[Checkpoint] Saved new best model checkpoint to {best_path}")

    return filepath


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load checkpoint and resume model, optimizer, scheduler states.
    Supports both raw state_dict and dictionary metadata containers.
    Raises clear FileNotFoundError if checkpoint is missing.
    """
    chk_path = Path(checkpoint_path)
    if not chk_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {chk_path.resolve()}.\n"
            f"Download or train the model before inference."
        )

    print(f"[Checkpoint] Loading checkpoint from {chk_path} on {device}")
    checkpoint = torch.load(str(chk_path), map_location=device)

    # Handle model state dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "params" in checkpoint:
            state_dict = checkpoint["params"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Attempt loading
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        # Fallback to non-strict if slight mismatch in optional keys
        print(f"[Checkpoint] Warning: Strict load failed ({e}). Attempting non-strict load...")
        model.load_state_dict(state_dict, strict=False)

    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and isinstance(checkpoint, dict) and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
    best_psnr = checkpoint.get("best_psnr", 0.0) if isinstance(checkpoint, dict) else 0.0
    print(f"[Checkpoint] Successfully loaded weights (Epoch: {epoch}, Best PSNR: {best_psnr:.2f} dB)")

    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": checkpoint}
