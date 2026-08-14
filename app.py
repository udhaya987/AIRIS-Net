import os
import sys
from pathlib import Path
import time
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image

# Add current workspace to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
SWINIR_DIR = PROJECT_ROOT / "SwinIR"
if str(SWINIR_DIR) not in sys.path:
    sys.path.append(str(SWINIR_DIR))

from data.degradation import RandomDegradationPipeline
from utils.image_utils import load_image, compute_change_map
from utils.metrics import calculate_psnr, calculate_ssim
from utils.edge_utils import compute_sobel_edges_np, edge_consistency_score
from utils.frequency_utils import compute_fft_magnitude_np, frequency_consistency_score
from baseline_swinir import SwinIRRestorer
from inference import AIRISPredictor

# Streamlit Page Config
st.set_page_config(
    page_title="AIRIS-Net Industrial Restoration",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00e5ff;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-swinir {
        background-color: #0369a1;
        color: #e0f2fe;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-airis {
        background-color: #047857;
        color: #d1fae5;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_swinir_model():
    chk_path = PROJECT_ROOT / "SwinIR" / "model_zoo" / "004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth"
    if chk_path.exists():
        return SwinIRRestorer(model_path=str(chk_path))
    return None


@st.cache_resource
def get_airis_model(mtime: float = 0.0):
    chk_path = PROJECT_ROOT / "checkpoints" / "best_airis.pth"
    if chk_path.exists():
        return AIRISPredictor(checkpoint_path=str(chk_path))
    return None


def main():
    st.markdown('<div class="main-title">🔬 AIRIS-Net</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Adaptive Industrial Restoration & Integrity-Safeguarding Network for Semiconductor Inspection</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration")

    model_choice = st.sidebar.selectbox(
        "Restoration Model",
        options=["AIRIS-Net (Multi-Expert)", "SwinIR Baseline (Pretrained)"],
        index=0
    )

    if st.sidebar.button("🔄 Reload Model Weights"):
        st.cache_resource.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("🧪 Input & Synthetic Degradation")

    # Image source
    image_source = st.sidebar.radio(
        "Image Source",
        options=["Sample Semiconductor Images", "Upload Custom Image"],
        index=0
    )

    clean_img = None
    sample_name = ""

    if image_source == "Sample Semiconductor Images":
        sample_dir = PROJECT_ROOT / "train" / "train" / "GT"
        if not sample_dir.exists():
            sample_dir = PROJECT_ROOT / "data" / "train" / "clean"

        if sample_dir.exists():
            sample_files = sorted(list(sample_dir.glob("*.npy")))[:30]
            if sample_files:
                chosen_sample = st.sidebar.selectbox(
                    "Choose Semiconductor Sample",
                    options=[f.name for f in sample_files],
                    index=0
                )
                sample_path = sample_dir / chosen_sample
                clean_img = load_image(sample_path, grayscale=True)
                sample_name = chosen_sample
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image (PNG, JPG, NPY)",
            type=["png", "jpg", "jpeg", "npy"]
        )
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".npy"):
                clean_img = np.load(uploaded_file).astype(np.float32)
                if clean_img.max() > 1.0:
                    clean_img = clean_img / 255.0
            else:
                pil_img = Image.open(uploaded_file).convert("L")
                clean_img = np.array(pil_img, dtype=np.float32) / 255.0
            sample_name = uploaded_file.name

    # Fallback if no image loaded
    if clean_img is None:
        clean_img = np.ones((256, 256), dtype=np.float32) * 0.5
        sample_name = "Synthetic Pattern"

    # Degradation mode
    deg_mode = st.sidebar.selectbox(
        "Synthetic Degradation",
        options=["None (Raw Input)", "Gaussian Noise", "Gaussian Blur", "Motion Blur", "Contrast Degradation", "Uneven Illumination", "Mixed Degradation"],
        index=1
    )

    pipeline = RandomDegradationPipeline(seed=42)

    # Apply degradation
    if deg_mode == "None (Raw Input)":
        degraded_img = clean_img.copy()
        deg_meta = {"type": "none"}
    elif deg_mode == "Gaussian Noise":
        noise_sigma = st.sidebar.slider("Noise Sigma (0-50)", 5.0, 50.0, 25.0)
        degraded_img, deg_meta = pipeline.apply_gaussian_noise(clean_img, sigma=noise_sigma)
    elif deg_mode == "Gaussian Blur":
        ksize = st.sidebar.select_slider("Kernel Size", options=[3, 5, 7, 9], value=5)
        sigma = st.sidebar.slider("Blur Sigma", 0.5, 3.0, 1.5)
        degraded_img, deg_meta = pipeline.apply_gaussian_blur(clean_img, ksize=ksize, sigma=sigma)
    elif deg_mode == "Motion Blur":
        ksize = st.sidebar.slider("Motion Kernel Length", 3, 15, 7, step=2)
        angle = st.sidebar.slider("Angle (°)", 0.0, 360.0, 45.0)
        degraded_img, deg_meta = pipeline.apply_motion_blur(clean_img, kernel_size=ksize, angle=angle)
    elif deg_mode == "Contrast Degradation":
        contrast_factor = st.sidebar.slider("Contrast Factor", 0.2, 1.0, 0.5)
        degraded_img, deg_meta = pipeline.apply_contrast_degradation(clean_img, factor=contrast_factor)
    elif deg_mode == "Uneven Illumination":
        strength = st.sidebar.slider("Illumination Strength", 0.1, 0.9, 0.5)
        degraded_img, deg_meta = pipeline.apply_uneven_illumination(clean_img, strength=strength)
    elif deg_mode == "Mixed Degradation":
        num_stages = st.sidebar.slider("Number of Degradations", 1, 4, 2)
        degraded_img, deg_meta = pipeline.apply_mixed_degradation(clean_img, num_degradations=num_stages)

    # Run Restoration Model
    restored_img = None
    mask_img = None
    reliability_img = None
    routing_weights = None
    elapsed_time = 0.0

    if "SwinIR" in model_choice:
        swinir = get_swinir_model()
        if swinir is not None:
            t0 = time.time()
            restored_img = swinir.restore(degraded_img)
            elapsed_time = time.time() - t0
        else:
            st.error("SwinIR pretrained checkpoint not found in SwinIR/model_zoo.")
            restored_img = degraded_img.copy()
    else:
        airis_chk = PROJECT_ROOT / "checkpoints" / "best_airis.pth"
        mtime = airis_chk.stat().st_mtime if airis_chk.exists() else 0.0
        airis = get_airis_model(mtime)
        if airis is not None:
            pred = airis.predict(degraded_img)
            restored_img = pred["restored"]
            mask_img = pred["mask"]
            reliability_img = pred["reliability"]
            routing_weights = pred["routing_weights"]
            elapsed_time = pred["elapsed_seconds"]
        else:
            st.warning("AIRIS-Net checkpoint not found. Falling back to SwinIR baseline.")
            swinir = get_swinir_model()
            if swinir is not None:
                restored_img = swinir.restore(degraded_img)

    if restored_img is None:
        restored_img = degraded_img.copy()

    # Metrics computation
    psnr_deg = calculate_psnr(degraded_img, clean_img)
    psnr_res = calculate_psnr(restored_img, clean_img)
    ssim_deg = calculate_ssim(degraded_img, clean_img)
    ssim_res = calculate_ssim(restored_img, clean_img)
    edge_score = edge_consistency_score(restored_img, clean_img)
    freq_score = frequency_consistency_score(restored_img, clean_img)
    change_map = compute_change_map(degraded_img, restored_img)

    # Top KPI Metrics Row
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        psnr_delta = psnr_res - psnr_deg
        st.metric("PSNR (Restored)", f"{psnr_res:.2f} dB", f"{psnr_delta:+.2f} dB")
    with mcol2:
        ssim_delta = ssim_res - ssim_deg
        st.metric("SSIM (Restored)", f"{ssim_res:.4f}", f"{ssim_delta:+.4f}")
    with mcol3:
        st.metric("Edge Consistency", f"{edge_score:.4f}")
    with mcol4:
        st.metric("Frequency Fidelity", f"{freq_score:.4f}")
    with mcol5:
        st.metric("Inference Time", f"{elapsed_time*1000:.1f} ms")

    st.markdown("---")

    # Tabs for Visual Inspection
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Restoration Overview", "🛡️ Integrity & Confidence", "🔍 Edge & Frequency Analysis", "🧠 Expert Routing & Diagnostics"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("1. Degraded Input")
            st.image(degraded_img, clamp=True, use_container_width=True, caption=f"Input (PSNR: {psnr_deg:.2f} dB)")
        with c2:
            st.subheader(f"2. {model_choice.split()[0]} Restored")
            st.image(restored_img, clamp=True, use_container_width=True, caption=f"Restored (PSNR: {psnr_res:.2f} dB)")
        with c3:
            st.subheader("3. Clean Ground Truth")
            st.image(clean_img, clamp=True, use_container_width=True, caption=f"Reference: {sample_name}")

    with tab2:
        st.subheader("Integrity-Safeguarding & Reliability Maps")
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.write("**Restoration Change Map** ($|I_{\\text{restored}} - I_{\\text{degraded}}|$)")
            st.image(change_map, clamp=True, use_container_width=True, caption="Active Correction Magnitude")
        with ic2:
            st.write("**Restoration Mask $M$** (0: Preserve, 1: Restore)")
            if mask_img is not None:
                st.image(mask_img, clamp=True, use_container_width=True, caption="Selective Restoration Gate")
            else:
                st.info("Mask available in AIRIS-Net mode")
        with ic3:
            st.write("**Predicted Reliability Map $R$** (0: Low, 1: High)")
            if reliability_img is not None:
                st.image(reliability_img, clamp=True, use_container_width=True, caption="Confidence Field")
            else:
                st.info("Reliability map available in AIRIS-Net mode")

    with tab3:
        st.subheader("Edge and Frequency Structural Verification")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.write("**Input Sobel Edges**")
            st.image(compute_sobel_edges_np(degraded_img), clamp=True, use_container_width=True)
        with e2:
            st.write("**Restored Sobel Edges**")
            st.image(compute_sobel_edges_np(restored_img), clamp=True, use_container_width=True)
        with e3:
            st.write("**Input 2D FFT Spectrum**")
            st.image(compute_fft_magnitude_np(degraded_img), clamp=True, use_container_width=True)
        with e4:
            st.write("**Restored 2D FFT Spectrum**")
            st.image(compute_fft_magnitude_np(restored_img), clamp=True, use_container_width=True)

    with tab4:
        st.subheader("Adaptive Multi-Expert Routing & Degradation Diagnostics")
        if routing_weights is not None:
            rw1, rw2, rw3 = st.columns(3)
            with rw1:
                st.metric("Local CNN Expert", f"{routing_weights[0]*100:.1f}%")
                st.progress(float(routing_weights[0]))
                st.caption("Specialized in sharp edges, localized textures, and micro-defects.")
            with rw2:
                st.metric("Global Context Expert", f"{routing_weights[1]*100:.1f}%")
                st.progress(float(routing_weights[1]))
                st.caption("Specialized in long-range structural dependencies and geometric continuity.")
            with rw3:
                st.metric("Frequency FFT Expert", f"{routing_weights[2]*100:.1f}%")
                st.progress(float(routing_weights[2]))
                st.caption("Specialized in band-separated spectral filtering and periodic noise removal.")
        else:
            st.info("Select AIRIS-Net in the sidebar to view dynamic multi-expert routing coefficients.")

        st.write("---")
        st.write("**Degradation Characteristics Analysis**")
        st.json(deg_meta)


if __name__ == "__main__":
    main()
