"""
🧠 Brain Tumor Classifier — by X-ESTAL NITISH

A premium CNN-powered brain tumor prediction tool with a
cutting-edge glassmorphism UI, animated effects, and real-time
confidence visualisations.
"""

import streamlit as st
from PIL import Image

from utils import load_model, predict, CLASS_INFO, CLASS_NAMES

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Brain Tumor Classifier | X-ESTAL NITISH",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Premium CSS — Glassmorphism + Animations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ══════════ ANIMATED BACKGROUND ══════════ */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 25%, #1b0a2e 50%, #0d1b2a 75%, #0a0a1a 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* floating orbs */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background:
            radial-gradient(circle at 20% 80%, rgba(0, 200, 255, 0.06) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(138, 43, 226, 0.06) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(0, 255, 136, 0.03) 0%, transparent 50%);
        animation: orbFloat 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    @keyframes orbFloat {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(30px, -30px) rotate(120deg); }
        66% { transform: translate(-20px, 20px) rotate(240deg); }
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        position: relative;
        z-index: 1;
    }

    /* hide defaults */
    #MainMenu, footer, header {visibility: hidden;}

    /* ══════════ GLASSMORPHISM BASE ══════════ */
    .glass {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
    }

    /* ══════════ BRAND BAR ══════════ */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.7rem 1.5rem;
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }
    .brand-name {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 3px;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-tag {
        font-family: 'Rajdhani', sans-serif;
        color: rgba(255,255,255,0.35);
        font-size: 0.8rem;
        letter-spacing: 1px;
    }

    /* ══════════ HERO ══════════ */
    .hero-section {
        text-align: center;
        padding: 3rem 1rem 2rem;
        position: relative;
    }
    .hero-icon {
        font-size: 4rem;
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
        filter: drop-shadow(0 0 30px rgba(0, 200, 255, 0.4));
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: 2px;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 50%, #ff006e 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textShimmer 3s linear infinite;
        margin: 0.5rem 0 0.3rem;
    }
    @keyframes textShimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    .hero-sub {
        font-family: 'Rajdhani', sans-serif;
        color: rgba(255,255,255,0.5);
        font-size: 1.2rem;
        font-weight: 400;
        letter-spacing: 4px;
        text-transform: uppercase;
    }
    .hero-line {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        margin: 1rem auto 0;
        border-radius: 3px;
    }

    /* ══════════ UPLOAD ZONE ══════════ */
    .upload-zone {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(20px);
        border: 2px dashed rgba(0, 212, 255, 0.3);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        text-align: center;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    .upload-zone::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.05), transparent);
        animation: scanLine 3s ease-in-out infinite;
    }
    @keyframes scanLine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    .upload-zone:hover {
        border-color: rgba(0, 212, 255, 0.6);
        background: rgba(0, 212, 255, 0.04);
    }
    .upload-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .upload-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: rgba(255,255,255,0.8);
    }
    .upload-hint {
        color: rgba(255,255,255,0.3);
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* ══════════ SECTION HEADER ══════════ */
    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 1rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-header .dot {
        width: 8px; height: 8px;
        background: #00d4ff;
        border-radius: 50%;
        box-shadow: 0 0 10px #00d4ff;
        display: inline-block;
    }

    /* ══════════ MRI PREVIEW CARD ══════════ */
    .mri-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .mri-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
    }
    .mri-card img {
        border-radius: 14px;
    }
    .mri-meta {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    .mri-meta-item {
        background: rgba(0,212,255,0.08);
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        border: 1px solid rgba(0,212,255,0.15);
    }

    /* ══════════ RESULT CARD ══════════ */
    .result-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .result-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #7b2ff7, #ff006e);
    }

    /* ══════════ RESULT BADGE ══════════ */
    .result-badge-wrap {
        text-align: center;
        padding: 1.5rem 0;
    }
    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 1rem 2.2rem;
        border-radius: 60px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        letter-spacing: 2px;
        position: relative;
        animation: badgeGlow 2s ease-in-out infinite;
    }
    @keyframes badgeGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(var(--glow-rgb), 0.3); }
        50% { box-shadow: 0 0 40px rgba(var(--glow-rgb), 0.5), 0 0 60px rgba(var(--glow-rgb), 0.2); }
    }

    /* ══════════ STAT CARDS ══════════ */
    .stat-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.3s, border-color 0.3s;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0,212,255,0.3);
    }
    .stat-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .stat-label {
        font-family: 'Rajdhani', sans-serif;
        color: rgba(255,255,255,0.4);
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* ══════════ CONFIDENCE BARS ══════════ */
    .conf-row {
        display: flex;
        align-items: center;
        margin-bottom: 14px;
        gap: 12px;
    }
    .conf-label {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        min-width: 130px;
        color: rgba(255,255,255,0.75);
    }
    .conf-bar-track {
        flex: 1;
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        height: 32px;
        overflow: hidden;
        position: relative;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 12px;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        color: #fff;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .conf-bar-fill.active::after {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        animation: barShine 2s ease-in-out infinite;
    }
    @keyframes barShine {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    /* ══════════ ABOUT CARD ══════════ */
    .about-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.8rem;
        position: relative;
        overflow: hidden;
        margin-top: 1.5rem;
    }
    .about-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        border-radius: 4px 0 0 4px;
    }
    .about-card h4 {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.95rem;
        letter-spacing: 2px;
        margin: 0 0 0.8rem;
    }
    .about-card p {
        font-family: 'Rajdhani', sans-serif;
        color: rgba(255,255,255,0.55);
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 0;
    }

    /* ══════════ REFERENCE CARDS ══════════ */
    .ref-card {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 1.4rem;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    .ref-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        border-radius: 3px 0 0 3px;
    }
    .ref-card:hover {
        transform: translateY(-6px);
        border-color: rgba(255,255,255,0.15);
    }
    .ref-card-icon {
        font-size: 2rem;
        margin-bottom: 0.6rem;
    }
    .ref-card h4 {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.8rem;
        letter-spacing: 2px;
        margin: 0 0 0.6rem;
        color: rgba(255,255,255,0.85);
    }
    .ref-card .severity-tag {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 6px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .ref-card p {
        font-family: 'Rajdhani', sans-serif;
        color: rgba(255,255,255,0.4);
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0;
    }

    /* ══════════ DIVIDER ══════════ */
    .neon-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), rgba(123,47,247,0.3), transparent);
        margin: 2.5rem 0;
        border: none;
    }

    /* ══════════ FOOTER ══════════ */
    .footer-section {
        text-align: center;
        padding: 2rem 1rem 1rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .footer-disclaimer {
        background: rgba(255, 77, 77, 0.06);
        border: 1px solid rgba(255, 77, 77, 0.15);
        border-radius: 14px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.2rem;
        display: inline-block;
    }
    .footer-disclaimer p {
        font-family: 'Rajdhani', sans-serif;
        color: rgba(255,255,255,0.5);
        font-size: 0.9rem;
        margin: 0;
    }
    .footer-brand {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 3px;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .footer-copy {
        color: rgba(255,255,255,0.2);
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Brand Bar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(
    """
    <div class="brand-bar">
        <span class="brand-name">⚡ X-ESTAL NITISH</span>
        <span class="brand-tag">DEEP LEARNING · MEDICAL AI</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hero Section
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-icon">🧠</div>
        <div class="hero-title">BRAIN TUMOR CLASSIFIER</div>
        <div class="hero-sub">CNN-Powered Neural Diagnostic Engine</div>
        <div class="hero-line"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Load Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

model = load_model()

if model is None:
    st.error("Failed to load the AI model. Please check that `brain_tumor_model.keras` exists in the project root.")
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Upload Section
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(
    """
    <div class="upload-zone">
        <div class="upload-icon">🔬</div>
        <div class="upload-title">Upload Brain MRI Scan</div>
        <div class="upload-hint">Supports JPG · JPEG · PNG  ·  Max 10 MB</div>
    </div>
    """,
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "upload_mri",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prediction & Results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    col_img, col_gap, col_result = st.columns([1, 0.05, 1.4])

    # ── Left Column: MRI Preview ──
    with col_img:
        st.markdown(
            '<div class="section-header"><span class="dot"></span>MRI SCAN</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="mri-card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        w, h = image.size
        st.markdown(
            f"""
            <div class="mri-meta">
                <span class="mri-meta-item">📐 {w} × {h} px</span>
                <span class="mri-meta-item">🎨 {image.mode}</span>
                <span class="mri-meta-item">📁 {uploaded_file.size / 1024:.0f} KB</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Right Column: Results ──
    with col_result:
        with st.spinner("Analysing MRI scan ..."):
            try:
                result = predict(model, image)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        label = result["label"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        meta = CLASS_INFO[label]
        color = meta["color"]

        st.markdown(
            '<div class="section-header"><span class="dot"></span>DIAGNOSIS RESULT</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # ── Result Badge ──
        # Convert hex to RGB for glow
        hex_c = color.lstrip("#")
        r, g, b = int(hex_c[:2], 16), int(hex_c[2:4], 16), int(hex_c[4:], 16)

        st.markdown(
            f"""
            <div class="result-badge-wrap">
                <div class="result-badge" style="
                    background: {color}15;
                    color: {color};
                    border: 2px solid {color}60;
                    --glow-rgb: {r},{g},{b};
                ">
                    {meta['icon']}  {label.upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Stat Cards ──
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value" style="color:{color}">{confidence:.1%}</div>
                    <div class="stat-label">Confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{meta['icon']}</div>
                    <div class="stat-label">{meta['severity']} Severity</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value" style="color:#00d4ff">4</div>
                    <div class="stat-label">Classes Scanned</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confidence Bars ──
        st.markdown(
            '<div class="section-header" style="margin-top:0.5rem"><span class="dot"></span>CLASS PROBABILITIES</div>',
            unsafe_allow_html=True,
        )

        for class_name in CLASS_NAMES:
            prob = probs[class_name]
            info = CLASS_INFO[class_name]
            is_active = class_name == label
            bar_color = (
                f"linear-gradient(90deg, {info['color']}cc, {info['color']})"
                if is_active
                else "rgba(255,255,255,0.08)"
            )
            text_color = "#fff" if is_active else "rgba(255,255,255,0.35)"
            active_class = "active" if is_active else ""

            st.markdown(
                f"""
                <div class="conf-row">
                    <span class="conf-label">{info['icon']} {class_name}</span>
                    <div class="conf-bar-track">
                        <div class="conf-bar-fill {active_class}" style="
                            width: {max(prob * 100, 2.5):.1f}%;
                            background: {bar_color};
                            color: {text_color};
                        ">{prob:.1%}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)  # close result-card

    # ── About This Diagnosis ──
    st.markdown(
        f"""
        <div class="about-card" style="border-color: {color}30;">
            <div style="position:absolute;left:0;top:0;bottom:0;width:4px;background:{color};border-radius:4px 0 0 4px;"></div>
            <h4 style="color:{color}">{meta['icon']}  ABOUT {label.upper()}</h4>
            <p>{meta['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tumor Types Reference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-header"><span class="dot"></span>TUMOR CLASSIFICATION REFERENCE</div>',
    unsafe_allow_html=True,
)

ref_cols = st.columns(4, gap="medium")
for idx, class_name in enumerate(CLASS_NAMES):
    info = CLASS_INFO[class_name]
    with ref_cols[idx]:
        st.markdown(
            f"""
            <div class="ref-card" style="border-color: {info['color']}15;">
                <div style="position:absolute;left:0;top:0;bottom:0;width:3px;background:{info['color']};border-radius:3px 0 0 3px;"></div>
                <div class="ref-card-icon">{info['icon']}</div>
                <h4>{class_name.upper()}</h4>
                <span class="severity-tag" style="background:{info['color']}15;color:{info['color']};border:1px solid {info['color']}30;">
                    {info['severity']}
                </span>
                <p>{info['description'][:130]}…</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Footer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(
    """
    <div class="footer-section">
        <div class="footer-disclaimer">
            <p>⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational &amp; research
            purposes only. It is <em>not</em> a substitute for professional medical diagnosis.
            Always consult a qualified healthcare provider.</p>
        </div>
        <div class="footer-brand">⚡ X-ESTAL NITISH</div>
        <div class="footer-copy">Built with Streamlit & TensorFlow  ·  © 2026 All Rights Reserved</div>
    </div>
    """,
    unsafe_allow_html=True,
)
