import pytest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airis.model import AIRISNet
from utils.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_save_and_load(tmp_path):
    """Verify model weights save and reload without distortion."""
    model_orig = AIRISNet(in_channels=1, base_channels=48, scale_factor=1)
    chk_dir = tmp_path / "checkpoints"
    chk_dir.mkdir(parents=True)

    state = {
        "epoch": 5,
        "model_state_dict": model_orig.state_dict(),
        "best_psnr": 28.5,
        "val_psnr": 28.5,
        "val_ssim": 0.82
    }
    saved_path = save_checkpoint(
        state=state,
        is_best=True,
        checkpoint_dir=str(chk_dir),
        filename="test_epoch_005.pth"
    )
    assert saved_path.exists()

    model_loaded = AIRISNet(in_channels=1, base_channels=48, scale_factor=1)
    res = load_checkpoint(saved_path, model_loaded, device=torch.device("cpu"))

    assert res["epoch"] == 5
    assert np_close_weights(model_orig, model_loaded)


def test_checkpoint_missing_error():
    """Verify loading a nonexistent checkpoint raises FileNotFoundError."""
    model = AIRISNet()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(Path("nonexistent_dir/nowhere.pth"), model)


def np_close_weights(m1, m2):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if not torch.allclose(p1, p2, atol=1e-6):
            return False
    return True
