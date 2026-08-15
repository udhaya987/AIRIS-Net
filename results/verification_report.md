# AIRIS-Net Technical Verification Report

## Environment
- **Python**: 3.12.10
- **PyTorch**: 2.13.0+cpu
- **Platform**: Windows (CPU fallback verified)

---

## Component Checks

| Module | Status | Verification Summary |
| :--- | :--- | :--- |
| **Project Structure** | PASS | Validated directories: `airis/`, `data/`, `utils/`, `configs/`, `checkpoints/`, `results/`, `sample_results/`, and CLI scripts. |
| **Dependencies** | PASS | Successfully loaded `torch`, `torchvision`, `cv2`, `numpy`, `skimage`, `PIL`, `yaml`, `pandas`, `lpips`, `timm`. |
| **Model Architecture (1x)** | PASS | `AIRISNet(scale_factor=1)` verified with 296,894 parameters; output shapes match input $(B, 1, H, W)$. |
| **Super-Resolution (2x)** | PASS | `AIRISNet(scale_factor=2)` verified: $(1, 1, 128, 128) \to (1, 1, 256, 256)$ and $(1, 1, 256, 256) \to (1, 1, 512, 512)$. |
| **Forward Pass & Routing** | PASS | Expert routing weights sum to $1.0000$ across Local CNN, Global Attention, and Frequency experts. |
| **Multi-Loss Backward** | PASS | `AIRISLoss` backward pass executed with valid gradients across all parameter groups. |
| **Degradation Engine** | PASS | Validated additive Gaussian noise, multiplicative speckle noise, spatial downsampling, and compound modes. |
| **Dataset & DataLoader** | PASS | `IndustrialRestorationDataset` and DataLoader tested with in-memory caching and random cropping. |
| **Checkpoint Management** | PASS | Validated saving and loading for `best_airis.pth`, `latest_airis.pth`, and epoch snapshots. |
| **KLA Batch Inference** | PASS | `kla_inference.py` executed across test directory with automatic hardware detection and latency tracking. |
| **Metric Calculations** | PASS | PSNR, SSIM, and AlexNet-based LPIPS verified with valid numerical bounds. |
| **Web Dashboard** | PASS | Streamlit app (`app.py`) starts cleanly with live degradation simulation and analytical map views. |
| **CPU Compatibility** | PASS | Full execution cycle verified on CPU without requiring dedicated GPU acceleration. |

---

## Output Artifacts

The following sample files were verified in `outputs/` and `sample_results/`:
- `outputs/restored.png` (Restored inspection image)
- `outputs/restoration_mask.png` (Spatial gating mask $M$)
- `outputs/reliability_map.png` (Estimated confidence map $R$)
- `outputs/routing_weights.txt` (Expert routing weights)
- `results/metrics.csv` (Per-image evaluation table)
