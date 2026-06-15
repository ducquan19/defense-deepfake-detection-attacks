# Defense in Deepfake Detection Attacks

Codebase nghiên cứu các phương pháp phòng thủ cho mô hình nhận diện deepfake trước các cuộc tấn công đối kháng.

## Pipeline chính

1. Thu thập và chuẩn hóa dữ liệu ảnh thật/giả.
2. Sử dụng hoặc tích hợp mô hình sinh ảnh deepfake để tạo dữ liệu giả.
3. Giả lập kẻ tấn công vào bộ nhận diện deepfake bằng adversarial attacks.
4. Áp dụng các phương pháp phòng thủ như adversarial defense, preprocessing defense hoặc adversarial training.
5. Triển khai ứng dụng Streamlit đơn giản để demo.

> Ghi chú đạo đức: dự án này được thiết kế cho mục tiêu nghiên cứu phòng thủ, đánh giá độ bền vững và tái lập thí nghiệm. Không sử dụng codebase để tạo danh tính giả mạo, lừa đảo hoặc triển khai tấn công ngoài môi trường được phép.
>
## Kiến trúc Mô hình Nhận diện (DINO-MAC)

Dự án này sử dụng mô hình cốt lõi **DINO-MAC** (kết hợp xương sống DINOv2 của Facebook với các Register Tokens và khối Multi-Aspect Classification Head) để làm công cụ nhận diện Deepfake chủ lực. Việc lựa chọn mô hình này dựa trên các nghiên cứu mới nhất (State-Of-The-Art) giúp khai thác triệt để các biểu diễn không gian và khả năng miễn nhiễm nhiễu (noise immunity) của Foundation Models, tỏ ra cực kỳ hiệu quả trong việc chống lại hình ảnh bị suy giảm chất lượng và các cuộc tấn công đối kháng.

## Cấu trúc dự án

```text
.
|-- app/                         # Ứng dụng Streamlit demo
|-- configs/                     # Cấu hình YAML cho data/model/attack/defense
|-- data/                        # Dữ liệu cục bộ, không commit dataset lớn
|   |-- raw/                     # Dữ liệu gốc
|   |-- processed/               # Dữ liệu đã tiền xử lý
|   `-- generated/               # Ảnh deepfake hoặc dữ liệu sinh ra
|-- models/                      # Checkpoint mô hình cục bộ
|-- notebooks/                   # Notebook thử nghiệm và phân tích
|-- reports/                     # Metrics, biểu đồ, artifact cho báo cáo
|-- scripts/                     # Entry point chạy pipeline
|-- src/                         # Source code chính
|   |-- attacks/                 # Các phương pháp tấn công đối kháng
|   |-- core/                    # Base class và interface dùng chung
|   |-- data/                    # Dataset, dataloader, preprocessing
|   |-- defenses/                # Các phương pháp phòng thủ
|   |-- evaluation/              # Đánh giá và metrics
|   |-- generation/              # Sinh hoặc import ảnh deepfake
|   |-- models/                  # Mô hình nhận diện deepfake
|   `-- utils/                   # Tiện ích dùng chung
`-- tests/                       # Unit test và smoke test
```

## Cài đặt

Tạo môi trường ảo và cài dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,app]"
```

Nếu chỉ muốn kiểm tra pipeline nhẹ, không cần GPU:

```powershell
pip install -e ".[dev]"
pytest
```

## Chạy trên Google Colab / Kaggle

Để huấn luyện các mô hình lớn như DINO-MAC một cách mượt mà, bạn nên sử dụng GPU mạnh trên Google Colab hoặc Kaggle. Mở notebook và chạy các lệnh sau trong một cell:

```bash
# 1. Clone mã nguồn về môi trường Colab
!git clone https://github.com/ducquan19/defense-deepfake-detection-attacks.git
%cd defense-deepfake-detection-attacks

# 2. Cài đặt các thư viện cần thiết (bao gồm uv để cài nhanh)
!pip install uv
!uv pip install --system -e ".[dev]"
!uv pip install --system torch torchvision transformers

# 3. Chạy pipeline với cấu hình DINO-MAC (sử dụng GPU)
!uv run scripts/run_pipeline.py --config configs/experiment_dino_mac.yaml
```

## Chạy pipeline mẫu

Pipeline ban đầu dùng dữ liệu synthetic nhẹ để kiểm tra toàn bộ luồng nghiên cứu. Sau đó có thể thay bằng dataset thật và mô hình generator/detector thật.

```powershell
python scripts/run_pipeline.py --config configs/experiment_baseline.yaml
```

Kết quả mặc định được ghi vào:

```text
reports/runs/baseline/
```

## Chạy Streamlit Demo

```powershell
streamlit run app/streamlit_app.py
```

Demo hỗ trợ upload ảnh và chạy predictor baseline. Khi có checkpoint thật, cập nhật đường dẫn trong `configs/experiment_baseline.yaml`.

## Base Classes

Tất cả component interface trong `src/core/base.py`.

- `BaseDataModule`: chuẩn hóa `train_dataloader`, `val_dataloader`, `test_dataloader`.
- `BaseDeepfakeGenerator`: sinh ảnh fake/deepfake từ batch input.
- `BaseDetector`: mô hình nhận diện deepfake, output logits `[B, 2]`.
- `BaseAttack`: tạo adversarial images từ detector và labels.
- `BaseDefense`: biến đổi hoặc làm sạch batch ảnh trước khi detect.
- `BaseEvaluator`: đánh giá clean, attacked và defended theo cùng một protocol.

Quy ước chung:

- Tensor ảnh: `[B, 3, H, W]`, kiểu float, giá trị trong khoảng `[0, 1]`.
- Label: `0 = real`, `1 = fake`.
- Detector output: logits `[B, 2]`.
- Metrics chính: accuracy, ROC-AUC; có thể mở rộng thêm EER, F1, robustness curve.

## Quản lý artifact

- Không commit dataset lớn, checkpoint, generated images hoặc report runtime lớn.
- Config YAML và code xử lý phải được version control để tái lập thí nghiệm.
- Mỗi experiment nên có seed, snapshot config và file `metrics.json`.
