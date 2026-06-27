# Defense in Deepfake Detection Attacks

Codebase nghiên cứu các phương pháp **phòng thủ đối kháng** (adversarial defense) cho mô hình nhận diện deepfake, bao gồm tái lập pipeline tấn công → phòng thủ → đánh giá độ bền vững.

> **Ghi chú đạo đức:** Dự án được thiết kế cho mục tiêu nghiên cứu phòng thủ, đánh giá độ bền vững và tái lập thí nghiệm. Không sử dụng codebase để tạo danh tính giả mạo, lừa đảo hoặc triển khai tấn công ngoài môi trường được phép.

---

## Pipeline nghiên cứu

```text
Clean Images
     │
     ▼
Baseline Detector  ──────────────► Stage 2: Clean Evaluation
(TinyCNN / DINO-MAC)                  accuracy, precision, recall, F1, AUC
     │
     ▼
Adversarial Attacks  (Stage 3)
  ├─ FGSM   (ε ∈ {0.001, 0.003, 0.005, 0.01, 0.03})
  ├─ I-FGSM (ε=0.03, steps=10)
  └─ PGD    (ε=0.01, α=0.001, steps=20, random start)
     │
     ▼
Attack Evaluation  (Stage 3)
  ACC_adv, F1_adv, AUC_adv, ASR
     │
     ├──── Stage 5: Preprocessing Defense
     │         JPEG re-encoding (quality=75)
     │
     └──── Stage 6: Adversarial Training
               L = L_clean + λ·L_adv  (λ=1.0, PGD on-the-fly)
                    │
                    ▼
              Robust Detector  ──► Stage 7: Robustness Comparison
                                    Baseline vs Robust (Markdown table)
```

---

## Kiến trúc Mô hình Nhận diện (DINO-MAC)

Dự án sử dụng **DINO-MAC** (DINOv2 backbone + Register Tokens + Multi-Aspect Classification Head) làm detector chủ lực. Model khai thác triệt để biểu diễn không gian của Foundation Models, tỏ ra hiệu quả trước hình ảnh bị suy giảm chất lượng và adversarial attacks.

Cho thí nghiệm nhẹ (không cần GPU), pipeline mặc định dùng **TinyCNN** với dữ liệu synthetic.

---

## Cấu trúc dự án

```text
.   
|-- app/                          # Ứng dụng Streamlit demo
|-- configs/
|   |-- experiment_baseline.yaml              # Baseline (TinyCNN, synthetic data)
|   |-- experiment_attack_eval.yaml           # Multi-attack evaluation (FGSM/I-FGSM/PGD)
|   |-- experiment_adversarial_training.yaml  # Adversarial training + robustness
|   `-- experiment_dino_mac.yaml              # DINO-MAC full training
|-- data/
|   |-- raw/                      # Dữ liệu gốc
|   |-- processed/                # Dữ liệu đã tiền xử lý
|   `-- generated/                # Ảnh deepfake hoặc dữ liệu sinh ra
|-- models/                       # Checkpoint mô hình cục bộ
|-- notebooks/                    # Notebook thử nghiệm và phân tích
|-- reports/
|   |-- runs/                     # Output của từng experiment run
|   `-- generate_report.py        # Tổng hợp kết quả → Markdown report
|-- scripts/
|   |-- run_pipeline.py                # Baseline pipeline
|   |-- run_attack_eval.py             # Multi-attack evaluation (Stage 2–5)
|   |-- run_adversarial_training.py    # Adversarial training + comparison (Stage 1–7)
|   `-- run_robustness_eval.py         # Standalone: so sánh 2 checkpoint
|-- src/
|   |-- attacks/
|   |   |-- fgsm.py               # FGSM: x_adv = x + ε·sign(∇L)
|   |   |-- ifgsm.py              # I-FGSM (BIM): iterative với L∞ clipping
|   |   `-- pgd.py                # PGD: random start + projection (Madry 2018)
|   |-- core/
|   |   `-- base.py               # BaseDetector, BaseAttack, BaseDefense, BaseEvaluator
|   |-- data/
|   |   `-- synthetic.py          # Synthetic dataset (checkerboard fake pattern)
|   |-- defenses/
|   |   |-- preprocessing.py      # JPEG smoothing defense
|   |   `-- adversarial_training.py  # AdversarialTrainer: L = L_clean + λ·L_adv
|   |-- evaluation/
|   |   `-- metrics.py            # compute_full_metrics, compute_asr, RobustnessEvaluator
|   |-- generation/
|   |   `-- toy_generator.py      # Toy deepfake generator (blend)
|   |-- models/
|   |   |-- tiny_cnn.py           # TinyCNN (lightweight baseline)
|   |   `-- dino_mac.py           # DINO-MAC (DINOv2 + MAC Head)
|   |-- pipeline.py               # Pipeline chính 7 stages
|   `-- utils/
|       `-- reproducibility.py    # seed_everything
`-- tests/
    `-- test_smoke.py             # 19 smoke tests (attacks, AT, metrics, evaluator)
```

---

## Cài đặt

**Khuyến nghị:** dùng `uv` để quản lý môi trường ảo (nhanh hơn pip):

```powershell
# Tạo môi trường và cài dependencies
uv sync
# Hoặc dùng pip truyền thống:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

> ⚠️ **Lưu ý NumPy:** Nếu dùng Anaconda, scipy/sklearn có thể compiled cho NumPy 1.x trong khi uv cài NumPy 2.x. Hãy chạy tests bằng `.venv\Scripts\python -m pytest` thay vì `pytest` của hệ thống.

---

## Chạy tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
# Expected: 19 passed
```

---

## Chạy pipeline mẫu (không cần GPU)

```powershell
# Baseline: train TinyCNN + clean evaluation
.venv\Scripts\python scripts/run_pipeline.py --config configs/experiment_baseline.yaml

# Multi-attack evaluation: FGSM × 5 epsilons, I-FGSM, PGD + JPEG defense
.venv\Scripts\python scripts/run_attack_eval.py

# Adversarial training + robustness comparison (baseline vs robust detector)
.venv\Scripts\python scripts/run_adversarial_training.py

# So sánh 2 checkpoint có sẵn (standalone)
.venv\Scripts\python scripts/run_robustness_eval.py `
    --baseline-ckpt models/tiny_cnn.pt `
    --robust-ckpt checkpoints/robust_detector.pth

# Tổng hợp tất cả kết quả → Markdown report
.venv\Scripts\python reports/generate_report.py
```

Kết quả được ghi vào:

```text
reports/runs/
├── baseline/               # clean_metrics.json
├── attack_eval/            # attack_report.json, defended_jpeg_metrics.json
└── adversarial_training/   # comparison_report.json, comparison_table.md
checkpoints/
└── robust_detector.pth     # Checkpoint sau adversarial training
```

---

## Chạy trên Google Colab / Kaggle

Để huấn luyện DINO-MAC trên GPU mạnh:

```bash
# 1. Clone mã nguồn
!git clone https://github.com/ducquan19/defense-deepfake-detection-attacks.git
%cd defense-deepfake-detection-attacks

# 2. Cài đặt
!pip install uv
!uv pip install --system -e ".[dev]"
!uv pip install --system torch torchvision transformers

# 3. Multi-attack evaluation với DINO-MAC
!python scripts/run_attack_eval.py --config configs/experiment_dino_mac.yaml

# 4. Adversarial training + comparison
!python scripts/run_adversarial_training.py --config configs/experiment_adversarial_training.yaml
```

---

## Adversarial Attacks

Tất cả attack implement `BaseAttack` — đầu ra là `AttackResult(adversarial_images, perturbations, metadata)`.

| Attack | Formula | Hyperparameters mặc định |
|--------|---------|--------------------------|
| **FGSM** | `x_adv = x + ε·sign(∇L)` | `ε ∈ {0.001, 0.003, 0.005, 0.01, 0.03}` |
| **I-FGSM** | `x_{t+1} = clip(x_t + α·sign(∇L), x±ε)` | `ε=0.03, steps=10, α=ε/steps` |
| **PGD** | `x_{t+1} = Proj_{B(x,ε)}(x_t + α·sign(∇L))` | `ε=0.01, α=0.001, steps=20, random_start=True` |

```python
from attacks import FGSMAttack, IFGSMAttack, PGDAttack, ATTACK_REGISTRY

attack = PGDAttack(epsilon=0.01, alpha=0.001, steps=20)
result = attack.perturb(model, images, labels)
# result.adversarial_images  → tensor [B, 3, H, W]
# result.metadata            → {"attack": "pgd", "epsilon": 0.01, ...}
```

---

## Defenses

| Defense | Mô tả |
|---------|-------|
| **JPEG Smoothing** | Re-encode qua JPEG (quality=75) để xóa high-freq perturbation |
| **Adversarial Training** | Huấn luyện với `L = L_clean + λ·L_adv`, sinh adversarial on-the-fly |

```python
from defenses.adversarial_training import AdversarialTrainer

trainer = AdversarialTrainer(
    model=model,
    optimizer=optimizer,
    device=device,
    attack_fn="pgd",   # "pgd" hoặc "fgsm"
    epsilon=0.01,
    alpha=0.001,
    steps=10,
    adv_lambda=1.0,
    checkpoint_dir="checkpoints",
)
trainer.fit(train_loader, epochs=10)
trainer.save_checkpoint("robust_detector.pth")
```

---

## Evaluation Metrics

```python
from evaluation.metrics import RobustnessEvaluator, compute_asr

evaluator = RobustnessEvaluator(model, test_loader, device)
report = evaluator.run(attacks=[FGSMAttack(0.01), IFGSMAttack(0.03), PGDAttack(0.01)])

# report["clean"]  → {accuracy, precision, recall, f1, roc_auc}
# report["fgsm"]   → {accuracy, precision, recall, f1, roc_auc, asr}
# report["pgd"]    → {accuracy, precision, recall, f1, roc_auc, asr}
```

**Attack Success Rate (ASR):**
```
ASR = # correctly-classified samples bị fool / # correctly-classified samples
```

---

## Base Classes

Tất cả component interface trong [`src/core/base.py`](src/core/base.py).

| Class | Mô tả |
|-------|-------|
| `BaseDataModule` | Chuẩn hóa `train/val/test_dataloader` |
| `BaseDeepfakeGenerator` | Sinh ảnh fake từ batch input |
| `BaseDetector` | Detector deepfake, output logits `[B, 2]` |
| `BaseAttack` | Tạo adversarial images, trả về `AttackResult` |
| `BaseDefense` | Biến đổi/làm sạch batch trước khi detect |
| `BaseEvaluator` | Đánh giá clean/attacked/defended theo cùng protocol |

**Quy ước tensor:**
- Ảnh: `[B, 3, H, W]`, `float32`, giá trị `[0, 1]`
- Label: `0 = real`, `1 = fake`
- Detector output: logits `[B, 2]`
- Metrics: accuracy, precision, recall, F1, ROC-AUC, ASR

---

## Chạy Streamlit Demo

```powershell
streamlit run app/streamlit_app.py
```

Demo hỗ trợ upload ảnh và chạy predictor baseline. Khi có checkpoint thật, cập nhật đường dẫn trong `configs/experiment_baseline.yaml`.

---

## Quản lý artifact

- Không commit dataset lớn, checkpoint, generated images hoặc report runtime lớn.
- Config YAML và source code phải được version control để tái lập thí nghiệm.
- Mỗi experiment tự động lưu: seed, snapshot config, `metrics_<seed>_<timestamp>.json`.
