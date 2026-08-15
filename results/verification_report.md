# AIRIS-Net Technical Verification Report

## Environment
- **Python Version**: 3.12.10
- **PyTorch Version**: 2.13.0+cpu
- **Platform / OS**: Windows 11 (64-bit)
- **Primary Device**: CPU
- **CUDA Available**: False (CPU fallback execution verified)

---

## Component Verification

| Component | Status | Verification Details |
| :--- | :---: | :--- |
| **Model Initialization** | **PASS** | `AIRISNet` initializes with 296,894 parameters ($1\times$) and 438,878 parameters ($2\times$ SR). |
| **Local Expert** | **PASS** | 3-stage depthwise-separable residual CNN blocks process high-frequency textures without dimension errors. |
| **Global Expert** | **PASS** | Windowed multi-head self-attention module extracts long-range periodic structures. |
| **Frequency Expert** | **PASS** | 2D FFT spectral decomposition separates and filters low, mid, and high frequency bands cleanly. |
| **Adaptive Router** | **PASS** | Gating network normalizes routing weights to sum strictly to $1.0000$ ($\sum \alpha_i = 1.0$). |
| **Forward Pass** | **PASS** | Output shapes verified: $(B, 1, H, W)$ for $1\times$, and $(B, 1, 2H, 2W)$ for $2\times$ SR. |
| **Backward Pass** | **PASS** | `AIRISLoss` backward pass propagates non-NaN gradients across all 72 trainable parameter tensors. |
| **Degradation Pipeline** | **PASS** | Verified additive Gaussian noise, multiplicative speckle noise ($I + I \odot \mathcal{N}$), downsampling, and compound modes with deterministic seed control. |
| **Dataset Loader** | **PASS** | `IndustrialRestorationDataset` and PyTorch `DataLoader` yield paired $(clean, degraded)$ tensors with RAM caching. |
| **Checkpoint Save / Load** | **PASS** | Saving and loading verified with state dictionaries and raw weights; raises `FileNotFoundError` on invalid path. |
| **Inference (KLA Batch)** | **PASS** | `kla_inference.py` executed batch inference across test images with hardware auto-detection. |
| **PSNR Metric** | **PASS** | `calculate_psnr` mathematically verified with ground truth bounds. |
| **SSIM Metric** | **PASS** | `calculate_ssim` and differentiable `ssim_torch` verified in range $[0, 1]$. |
| **LPIPS Metric** | **PASS** | AlexNet-based perceptual loss evaluated on test images with graceful CPU fallback. |
| **Streamlit Web Demo** | **PASS** | `app.py` loads successfully with interactive sliders and real-time visualization of restoration and reliability maps. |
| **Automated Tests** | **PASS** | Full `pytest` test suite in `tests/` executed without errors. |

---

## Automated Test Execution Summary

Executed using `pytest tests/ -v`:

- **Passed**: 21
- **Failed**: 0
- **Skipped**: 0
- **Total Tests**: 21

---

## Measured Test Set Benchmark (25 Held-Out Test Images)

| Metric | Degraded Input (Baseline) | AIRIS-Net Restored | Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **PSNR (dB)** | 19.88 dB | **23.85 dB** | **+3.97 dB** |
| **SSIM** | 0.5149 | **0.6856** | **+0.1707** |
| **LPIPS** | 0.5512 | **0.3555** | **-0.1957** (lower is better) |
| **Avg Latency (CPU)** | 0.0 ms | **1045.94 ms** | Full resolution inference |
