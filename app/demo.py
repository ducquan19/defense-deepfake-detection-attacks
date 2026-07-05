"""
Deepfake Detection Defense — Interactive Demo
===============================================
Premium Gradio UI showcasing the Robust Detector (Adversarial Training)
with live adversarial attack simulation and JPEG defense visualization.

Usage:
    python app/demo.py
    # or: uv run app/demo.py
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import numpy as np
import torch
import torchattacks
from PIL import Image

from models.tiny_cnn import TinyCNN
from models.dino_mac import DinoMACForDeepfakeDetection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROBUST_CHECKPOINT = ROOT / "models" / "robust_detector.pth"
BASELINE_CHECKPOINT = ROOT / "models" / "tiny_cnn.pt"
DINOMAC_CHECKPOINT = ROOT / "models" / "dinomac_best_42_1781520775.pt"
SAMPLE_DIR = ROOT / "reports" / "samples"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TINY_IMG_SIZE = 128
DINO_IMG_SIZE = 224
# ImageNet normalization for DINOv2
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
DINO_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

ATTACK_NAMES = ["fgsm", "ifgsm", "pgd", "mifgsm", "nifgsm", "deepfool", "cw", "square"]

# Results from README — for the comparison dashboard
RESULTS_DATA = {
    "Clean":        {"baseline_acc": 62.44, "robust_acc": 57.63, "baseline_asr": None,  "robust_asr": None},
    "FGSM":         {"baseline_acc": 5.00,  "robust_acc": 37.47, "baseline_asr": 91.98, "robust_asr": 34.98},
    "I-FGSM":       {"baseline_acc": 2.21,  "robust_acc": 35.94, "baseline_asr": 96.46, "robust_asr": 37.65},
    "PGD":          {"baseline_acc": 23.14, "robust_acc": 49.90, "baseline_asr": 62.95, "robust_asr": 13.41},
    "MI-FGSM":      {"baseline_acc": 22.23, "robust_acc": 49.76, "baseline_asr": 64.41, "robust_asr": 13.66},
    "NI-FGSM":      {"baseline_acc": 31.05, "robust_acc": 51.53, "baseline_asr": 50.26, "robust_asr": 10.58},
    "DeepFool":     {"baseline_acc": 16.74, "robust_acc": 4.88,  "baseline_asr": 73.19, "robust_asr": 91.52},
    "C&W":          {"baseline_acc": 49.30, "robust_acc": 54.97, "baseline_asr": 21.04, "robust_asr": 4.62},
    "Square":       {"baseline_acc": 44.74, "robust_acc": 55.16, "baseline_asr": 28.35, "robust_asr": 4.28},
    "AutoAttack":   {"baseline_acc": 21.68, "robust_acc": 49.76, "baseline_asr": 65.27, "robust_asr": 13.65},
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_tinycnn(checkpoint_path: Path) -> TinyCNN:
    model = TinyCNN()
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def load_dinomac(checkpoint_path: Path) -> DinoMACForDeepfakeDetection:
    model = DinoMACForDeepfakeDetection(
        model_name="facebook/dinov2-with-registers-small",
        freeze_backbone=True,
    )
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


MODEL = load_tinycnn(ROBUST_CHECKPOINT)           # Robust TinyCNN (Adversarial Training)
BASELINE_MODEL = load_tinycnn(BASELINE_CHECKPOINT) # Baseline TinyCNN (no defense)
DINOMAC_MODEL = load_dinomac(DINOMAC_CHECKPOINT)   # DINOv2-MAC (pretrained large model)

# Named registry for dropdown → model + preprocess lookup
MODEL_REGISTRY = {
    "🤖 TinyCNN Thường (Dễ bị lừa)": ("tiny", BASELINE_MODEL),
    "🛡️ TinyCNN Miễn Dịch (Adversarial Training)": ("tiny", MODEL),
    "🧠 DINOv2-MAC (Large Vision Model)": ("dino", DINOMAC_MODEL),
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def preprocess_tiny(img: Image.Image) -> torch.Tensor:
    """PIL → [1, 3, 128, 128] float tensor in [0,1] for TinyCNN."""
    resized = img.convert("RGB").resize((TINY_IMG_SIZE, TINY_IMG_SIZE))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def preprocess_dino(img: Image.Image) -> torch.Tensor:
    """PIL → [1, 3, 224, 224] ImageNet-normalized tensor for DINOv2."""
    resized = img.convert("RGB").resize((DINO_IMG_SIZE, DINO_IMG_SIZE))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return (t - DINO_MEAN) / DINO_STD


def preprocess(img: Image.Image, model_type: str = "tiny") -> torch.Tensor:
    """Dispatch to the correct preprocessing based on model type."""
    if model_type == "dino":
        return preprocess_dino(img)
    return preprocess_tiny(img)


def tensor_to_pil(t: torch.Tensor, model_type: str = "tiny") -> Image.Image:
    """[1, 3, H, W] tensor → PIL. Undo normalization for DINOv2."""
    t = t.squeeze(0).detach().cpu()
    if model_type == "dino":
        # Undo ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = t * std + mean
    arr = t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def jpeg_roundtrip(t: torch.Tensor, quality: int = 75, model_type: str = "tiny") -> torch.Tensor:
    """JPEG smoothing defense on a single image tensor."""
    pil = tensor_to_pil(t, model_type)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")
    arr = np.asarray(decoded, dtype=np.float32) / 255.0
    out = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    if model_type == "dino":
        out = (out - DINO_MEAN) / DINO_STD
    return out


# ---------------------------------------------------------------------------
# Attack implementations (lightweight, for demo)
# ---------------------------------------------------------------------------
def fgsm_attack_demo(model, images, labels, epsilon=0.03):
    images_adv = images.clone().detach().requires_grad_(True)
    logits = model(images_adv)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    return (images + epsilon * images_adv.grad.sign()).clamp(0, 1)


def pgd_attack_demo(model, images, labels, epsilon=0.01, alpha=0.003, steps=10):
    adv = images.clone().detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(model(adv), labels)
        loss.backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            delta = (adv - images).clamp(-epsilon, epsilon)
            adv = (images + delta).clamp(0, 1)
    return adv.detach()


def ifgsm_attack_demo(model, images, labels, epsilon=0.03, steps=10):
    alpha = epsilon / steps
    adv = images.clone().detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(model(adv), labels)
        loss.backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            adv = torch.max(torch.min(adv, images + epsilon), images - epsilon)
            adv = adv.clamp(0, 1)
    return adv.detach()


def torchattacks_wrapper(attack_cls, model, images, labels, **kwargs):
    """Generic wrapper for torchattacks classes."""
    atk = attack_cls(model, **kwargs)
    return atk(images, labels)


# All 9 attacks from the project + variants
DEMO_ATTACKS = {
    # --- White-box Gradient (basic) ---
    "FGSM (ε=0.03)":          lambda m, x, y: fgsm_attack_demo(m, x, y, 0.03),
    "FGSM (ε=0.06)":          lambda m, x, y: fgsm_attack_demo(m, x, y, 0.06),
    # --- Iterative ---
    "I-FGSM (ε=0.03, 10 steps)":  lambda m, x, y: ifgsm_attack_demo(m, x, y, 0.03, 10),
    "PGD (ε=0.01, 10 steps)":     lambda m, x, y: pgd_attack_demo(m, x, y, 0.01, 0.003, 10),
    "PGD (ε=0.03, 20 steps)":     lambda m, x, y: pgd_attack_demo(m, x, y, 0.03, 0.005, 20),
    # --- Momentum-based ---
    "MI-FGSM (ε=0.03)":       lambda m, x, y: torchattacks_wrapper(torchattacks.MIFGSM, m, x, y, eps=0.03, steps=10, decay=1.0),
    "NI-FGSM (ε=0.03)":       lambda m, x, y: torchattacks_wrapper(torchattacks.NIFGSM, m, x, y, eps=0.03, steps=10, decay=1.0),
    # --- Optimization-based ---
    "DeepFool (50 steps)":     lambda m, x, y: torchattacks_wrapper(torchattacks.DeepFool, m, x, y, steps=50, overshoot=0.02),
    "C&W (L2, 50 steps)":      lambda m, x, y: torchattacks_wrapper(torchattacks.CW, m, x, y, c=1, kappa=0, steps=50, lr=0.01),
    # --- Black-box / Score-based ---
    "Square (ε=0.03)":         lambda m, x, y: torchattacks_wrapper(torchattacks.Square, m, x, y, eps=0.03, n_queries=100),
    # --- Ensemble ---
    "AutoAttack (APGD, ε=0.03)": lambda m, x, y: torchattacks_wrapper(torchattacks.APGD, m, x, y, eps=0.03, steps=100),
}


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
def predict_single(img: Image.Image | None, model_choice: str = ""):
    """Run detector on a single uploaded image (Tab 1)."""
    if img is None:
        return "<div class='empty-state'>⏳ Chờ tải ảnh...</div>"
    # Resolve model
    mtype, target_model = MODEL_REGISTRY.get(model_choice, ("tiny", MODEL))
    tensor = preprocess(img, mtype)
    with torch.no_grad():
        probs = target_model(tensor).softmax(dim=1)[0]
    p_real, p_fake = probs[0].item(), probs[1].item()
    label = "🟢 REAL" if p_real > p_fake else "🔴 FAKE"
    confidence = max(p_real, p_fake) * 100
    details = f"Real: {p_real*100:.1f}%  |  Fake: {p_fake*100:.1f}%"

    color = "var(--color-real)" if p_real > p_fake else "var(--color-fake)"
    bar_html = f"""
    <div class="result-card result-card--single">
        <div class="result-verdict" style="color:{color};">{label}</div>
        <div class="result-confidence">Độ tin cậy: {confidence:.1f}%</div>
        <div class="confidence-track">
            <div class="confidence-fill" style="width:{confidence:.0f}%; background:linear-gradient(90deg, {color}, {color}99);"></div>
        </div>
        <div class="result-details">{details}</div>
    </div>
    """
    return bar_html


def defense_demo(img: Image.Image | None, model_choice: str, attack_name: str, jpeg_quality: int):
    """Pipeline: Chọn Model -> Dự đoán Gốc -> Tấn công -> Phòng thủ JPEG."""
    if img is None:
        return None, None, None, "<div class='empty-state'>⏳ Vui lòng tải ảnh lên.</div>"

    # Resolve model and preprocessing type
    mtype, target_model = MODEL_REGISTRY.get(model_choice, ("tiny", BASELINE_MODEL))
    tensor = preprocess(img, mtype)
    attack_fn = DEMO_ATTACKS[attack_name]

    # 1. Dự đoán ảnh gốc (Tìm nhãn mà mô hình đang tin tưởng)
    with torch.no_grad():
        orig_probs = target_model(tensor).softmax(dim=1)[0]
    predicted_label = torch.argmax(orig_probs).unsqueeze(0)

    # 2. Tấn công (Cố gắng ép mô hình lật ngược dự đoán)
    adv_tensor = attack_fn(target_model, tensor, predicted_label)
    with torch.no_grad():
        atk_probs = target_model(adv_tensor).softmax(dim=1)[0]

    # 3. Phòng thủ (Nén JPEG để xóa nhiễu)
    defended_jpeg = jpeg_roundtrip(adv_tensor, quality=jpeg_quality, model_type=mtype)
    with torch.no_grad():
        def_probs = target_model(defended_jpeg).softmax(dim=1)[0]

    # --- Render HTML ---
    def _info(probs):
        is_real = probs[0] > probs[1]
        label = "🟢 REAL" if is_real else "🔴 FAKE"
        color = "var(--color-real)" if is_real else "var(--color-fake)"
        pct = max(probs[0], probs[1]).item() * 100
        return label, color, pct

    l_orig, c_orig, p_orig = _info(orig_probs)
    l_atk, c_atk, p_atk = _info(atk_probs)
    l_def, c_def, p_def = _info(def_probs)

    result_html = f"""
    <div class="summary-row">
        <div class="summary-card" style="border-top: 3px solid #64748b;">
            <div class="summary-title" style="color: #94a3b8;">1. Ảnh Gốc</div>
            <div class="summary-result" style="color:{c_orig};">{l_orig}<br/><span style="font-size:0.9rem; color:gray;">({p_orig:.1f}%)</span></div>
        </div>
        <div class="summary-card" style="border-top: 3px solid #ef4444;">
            <div class="summary-title" style="color: #fca5a5;">2. Bị Tấn Công</div>
            <div class="summary-result" style="color:{c_atk};">{l_atk}<br/><span style="font-size:0.9rem; color:gray;">({p_atk:.1f}%)</span></div>
        </div>
        <div class="summary-card" style="border-top: 3px solid #10b981;">
            <div class="summary-title" style="color: #6ee7b7;">3. Sau Phòng Thủ (JPEG)</div>
            <div class="summary-result" style="color:{c_def};">{l_def}<br/><span style="font-size:0.9rem; color:gray;">({p_def:.1f}%)</span></div>
        </div>
    </div>
    """

    # Tạo các bức ảnh đầu ra
    adv_pil = tensor_to_pil(adv_tensor, mtype)
    def_pil = tensor_to_pil(defended_jpeg, mtype)
    noise = (adv_tensor - tensor).abs()
    noise_pil = tensor_to_pil((noise / noise.max()).clamp(0, 1) if noise.max() > 0 else noise)

    return adv_pil, def_pil, noise_pil, result_html


def build_results_table():
    """Build HTML table for experiment results dashboard."""
    rows = ""
    for name, d in RESULTS_DATA.items():
        ba, ra = d["baseline_acc"], d["robust_acc"]
        diff = ra - ba
        diff_color = "var(--color-real)" if diff > 0 else ("var(--color-fake)" if diff < 0 else "var(--body-text-color-subdued)")
        diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"

        b_asr = f'{d["baseline_asr"]:.2f}%' if d["baseline_asr"] is not None else "—"
        r_asr = f'{d["robust_asr"]:.2f}%' if d["robust_asr"] is not None else "—"

        rows += f"""
        <tr>
            <td class="rt-name">{name}</td>
            <td class="rt-baseline">{ba:.2f}%</td>
            <td class="rt-robust">{ra:.2f}%</td>
            <td style="color:{diff_color}; font-weight:700;">{diff_str}</td>
            <td class="rt-asr-b">{b_asr}</td>
            <td class="rt-asr-r">{r_asr}</td>
        </tr>"""

    return f"""
    <div class="results-table-wrap">
    <table class="results-table">
        <thead>
            <tr>
                <th>Thuật toán</th>
                <th class="rt-baseline">Acc (Baseline)</th>
                <th class="rt-robust">Acc (Robust)</th>
                <th>Δ Accuracy</th>
                <th class="rt-asr-b">ASR (Baseline)</th>
                <th class="rt-asr-r">ASR (Robust)</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    </div>
    """


def load_sample_gallery():
    """Load attack sample images from reports/samples/."""
    gallery = []
    for attack in ATTACK_NAMES:
        adv_path = SAMPLE_DIR / f"adv_{attack}.jpg"
        noise_path = SAMPLE_DIR / f"noise_{attack}.jpg"
        if adv_path.exists():
            gallery.append((str(adv_path), f"{attack.upper()} — Adversarial"))
        if noise_path.exists():
            gallery.append((str(noise_path), f"{attack.upper()} — Noise"))
    return gallery


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#080b14",
    body_background_fill_dark="#080b14",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#94a3b8",
    body_text_color_subdued_dark="#94a3b8",
    background_fill_primary="#0f172a",
    background_fill_primary_dark="#0f172a",
    background_fill_secondary="#131c2f",
    background_fill_secondary_dark="#131c2f",
    border_color_primary="#1e293b",
    border_color_primary_dark="#1e293b",
    block_background_fill="#111a2e",
    block_background_fill_dark="#111a2e",
    block_border_color="#1e293b",
    block_border_color_dark="#1e293b",
    input_background_fill="#0b1120",
    input_background_fill_dark="#0b1120",
    input_border_color="#1e293b",
    input_border_color_dark="#1e293b",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #8b5cf6)",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root, :root.dark, :root .dark {
    --color-real: #22c55e;
    --color-fake: #ef4444;
}
body, .gradio-container {
    background: radial-gradient(circle at 15% 0%, #1a1033 0%, transparent 45%),
                radial-gradient(circle at 85% 10%, #0c2a4a 0%, transparent 40%),
                linear-gradient(180deg, #080b14 0%, #0a0e1a 100%) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
footer { display: none !important; }

#hero-title { text-align: center; padding: 28px 16px 6px; }
#hero-title h1 {
    font-size: 2.3rem; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
#hero-subtitle {
    text-align: center; color: var(--body-text-color-subdued);
    font-size: 1.02rem; margin-bottom: 20px; line-height: 1.7;
}

.empty-state {
    text-align: center; padding: 48px 16px;
    color: var(--body-text-color-subdued); font-size: 0.95rem;
}

.result-card--single { text-align: center; padding: 18px 0; }
.result-verdict { font-size: 2.1rem; font-weight: 800; }
.confidence-track {
    background: var(--border-color-primary); border-radius: 999px; height: 12px;
    overflow: hidden; margin: 8px auto; max-width: 320px;
}
.confidence-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }

.summary-row { display: flex; flex-wrap: wrap; gap: 12px; }
.summary-card {
    flex: 1 1 200px; text-align: center; padding: 14px;
    border-radius: 14px; border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
}
.summary-title { font-size: 0.9rem; font-weight: 600; margin-bottom: 6px; }
.summary-result { font-size: 1.4rem; font-weight: 700; }

.results-table-wrap { overflow-x: auto; border-radius: 16px; border: 1px solid var(--border-color-primary); }
.results-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.results-table thead tr { background: var(--background-fill-secondary); border-bottom: 2px solid var(--border-color-primary); }
.results-table th, .results-table td { padding: 10px 14px; text-align: left; }
.results-table .rt-baseline { color: #f87171; }
.results-table .rt-robust { color: #38bdf8; }
.results-table tbody tr:hover { background: var(--background-fill-secondary); }
"""

# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
def build_demo():
    with gr.Blocks(title="Deepfake Defense Demo") as demo:
        # Header
        gr.HTML("""
        <div id="hero-title">
            <h1>🛡️ Hệ thống phòng thủ Deepfake</h1>
        </div>
        <div id="hero-subtitle">
            Mô phỏng quy trình Tấn công Đối kháng và Đánh giá phương pháp Phòng thủ
        </div>
        """)

        with gr.Tabs():
            # ── Tab 1: Detector ──
            with gr.Tab("🔍 Nhận diện Nhanh", id="detect"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(type="pil", label="📷 Ảnh đầu vào", height=300)
                        detect_model = gr.Dropdown(
                            choices=list(MODEL_REGISTRY.keys()),
                            value=list(MODEL_REGISTRY.keys())[1],
                            label="Chọn mô hình"
                        )
                        detect_btn = gr.Button("🔍 Phân tích", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        result_html = gr.HTML(label="Kết quả", value="<div class='empty-state'>⏳ Tải ảnh lên để bắt đầu...</div>")

                detect_btn.click(fn=predict_single, inputs=[input_img, detect_model], outputs=[result_html])
                input_img.change(fn=predict_single, inputs=[input_img, detect_model], outputs=[result_html])

            # ── Tab 2: Attack + Defense Pipeline (Tối giản) ──
            with gr.Tab("⚔️🛡️ Tấn công & Phòng thủ", id="attack_defense"):
                with gr.Row():
                    # --- BẢNG ĐIỀU KHIỂN ---
                    with gr.Column(scale=1):
                        def_img = gr.Image(type="pil", label="1. Tải ảnh gốc", height=220)
                        
                        def_model = gr.Dropdown(
                            choices=list(MODEL_REGISTRY.keys()),
                            value=list(MODEL_REGISTRY.keys())[0],
                            label="2. Chọn Mô hình mục tiêu"
                        )
                        
                        def_method = gr.Dropdown(
                            choices=list(DEMO_ATTACKS.keys()),
                            value="PGD (ε=0.03, 20 steps)",
                            label="3. Chọn thuật toán Tấn Công"
                        )
                        
                        def_jpeg_q = gr.Slider(
                            10, 100, value=75, step=5, 
                            label="4. Cường độ Phòng Thủ (Chất lượng JPEG)"
                        )
                            
                        def_btn = gr.Button("🚀 Chạy Thử Nghiệm", variant="primary")

                    # --- KẾT QUẢ ---
                    with gr.Column(scale=2):
                        def_result_html = gr.HTML(
                            value="<div class='empty-state'>⏳ Thiết lập thông số và nhấn Chạy Thử Nghiệm...</div>"
                        )
                        
                        gr.Markdown("---")
                        with gr.Row():
                            def_adv_out = gr.Image(label="😈 Ảnh bị gắn nhiễu", height=200)
                            def_jpeg_out = gr.Image(label="🧹 Ảnh sau phòng thủ (JPEG)", height=200)
                            def_noise_out = gr.Image(label="🔬 Lớp nhiễu (Phóng đại)", height=200)

                def_btn.click(
                    fn=defense_demo,
                    inputs=[def_img, def_model, def_method, def_jpeg_q],
                    outputs=[def_adv_out, def_jpeg_out, def_noise_out, def_result_html],
                )

            # ── Tab 3: Results Dashboard ──
            with gr.Tab("📊 Kết quả Thực nghiệm", id="results"):
                gr.Markdown("### So sánh Baseline vs Robust Detector trên 9 thuật toán tấn công")
                gr.HTML(build_results_table())
                
                gr.Markdown("""
                ---
                #### 📝 Nhận xét
                - Mô hình **Robust** cải thiện Accuracy trung bình **+20-33%** trên các tấn công White-box (FGSM, I-FGSM, PGD)
                - **ASR giảm mạnh** từ >90% xuống còn ~13-37% cho hầu hết các tấn công
                """)
            
            # ── Tab 4: Sample Gallery ──
            with gr.Tab("🖼️ Minh họa Tấn công", id="gallery"):
                gallery_items = load_sample_gallery()
                if gallery_items:
                    gr.Gallery(value=gallery_items, columns=4, height=500, object_fit="contain")
                else:
                    gr.Markdown("*Chưa có ảnh mẫu. Chạy `python scripts/generate_samples.py` để sinh ảnh.*")
            
            # ── Tab 5: About ──
            with gr.Tab("ℹ️ Giới thiệu", id="about"):
                gr.Markdown(f"""
                ### 🛡️ Deepfake Detection Defense Research
                **Mục tiêu:** Nghiên cứu sự suy giảm hiệu suất của mô hình phát hiện Deepfake trước các tấn công đối kháng
                và đánh giá hiệu quả các phương pháp phòng thủ.
                
                - **Checkpoint:** `{ROBUST_CHECKPOINT.name}`
                - **Device:** `{DEVICE}`
                """)
    
    return demo

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_demo()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861, # Lưu ý port được đổi sang 7861 để tránh kẹt
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=THEME,
    )