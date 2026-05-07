<!-- @format -->

# Đề tài 3: Community Structure Identification

## Tên đề tài

- **Tiếng Việt**: Xử lý bài toán Community Structure Identification sử dụng các thuật toán trong Khoa học Máy tính
- **Tiếng Anh**: Solving the Community Structure Identification problem using algorithms in Computer Science

## Mục tiêu

- Áp dụng các thuật toán phân cụm cộng đồng cho bài toán phân cụm hình ảnh
- Dataset: ImageNet-Hard
- So sánh và tối ưu hóa ít nhất 4 thuật toán

## Cấu trúc dự án

```
DADN_TTNT/
│
├── main.py                      # Entry point - Pipeline chính
├── tune_params.py               # Script tuning tham số
├── requirements.txt             # Dependencies
├── README.md
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py                # Cấu hình trung tâm
│   │
│   ├── models/                  # Core algorithms
│   │   ├── __init__.py
│   │   ├── feature_extractor.py   # Trích xuất đặc trưng (EfficientNet, DINOv2...)
│   │   ├── graph_builder.py       # Xây dựng đồ thị k-NN
│   │   └── clustering.py          # 4 thuật toán phân cụm
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── data_loader.py        # Load & tiền xử lý dữ liệu
│       ├── evaluation.py         # Metrics (NMI, ARI, Purity...)
│       ├── feature_cache.py      # Lưu/tải feature cache
│       ├── result_logger.py      # Lưu kết quả
│       └── visualization.py      # Vẽ biểu đồ, trực quan hóa
│
├── data/                        # Dataset ImageNet-Hard (.parquet)
│   ├── validation-0.parquet
│   ├── ...
│
├── results/                     # Output & cache
│   ├── results.csv
│   ├── tuning_results.csv
│   ├── comparison.png
│   ├── radar_chart.png
│   ├── clusters.png
│   ├── cluster_distribution.png
│   └── cache/
│       └── features_dinov2_vits14_full.npz
│
├── docs/                        # Documentation
│   ├── PIPELINE_DETAIL.md       # Chi tiết pipeline
│   ├── FINETUNING_GUIDE.md      # Hướng dẫn fine-tuning
│   └── LOGGING_GUIDE.md         # Hướng dẫn logging
│
└── notebooks/                   # Jupyter notebooks cho experiments
    ├── EDA.ipynb
    └── image_size_analysis.ipynb
```

## Các thuật toán được implement

| #   | Thuật toán  | Mô tả                                                                |
| --- | ----------- | -------------------------------------------------------------------- |
| 1   | **Infomap** | Dựa trên lý thuyết thông tin, tối thiểu hóa độ dài mô tả random walk |
| 2   | **Leiden**  | Cải tiến của Louvain, đảm bảo các cộng đồng kết nối tốt hơn          |
| 3   | **Louvain** | Tối ưu hóa modularity theo hướng greedy                              |
| 4   | **LPA**     | Label Propagation Algorithm - Lan truyền nhãn                        |

## Pipeline xử lý

```
1. Load Data → 2. Feature Extraction → 3. Build Graph → 4. Clustering → 5. Evaluation
     ↓                   ↓                    ↓              ↓              ↓
  Parquet           EfficientNet          k-NN Graph     4 Algorithms   NMI, ARI,
   Files            (1280 dims)           (Cosine)       Infomap,...    Purity
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy chương trình

```bash
# Chạy với toàn bộ dữ liệu
python main.py

# Chạy với sample nhỏ để test nhanh
python main.py --sample 500

# Tự động chọn thiết bị (GPU nếu có CUDA, ngược lại CPU)
python main.py --device auto

# Bắt buộc chạy trên CPU
python main.py --device cpu

# Yêu cầu chạy trên GPU (nếu không có CUDA sẽ tự fallback về CPU)
python main.py --device gpu

# Chạy full pipeline và lưu feature cache để dùng lại
python main.py --device gpu --save-feature-cache

# Chạy từ feature cache (bỏ qua bước extract feature)
python main.py --use-feature-cache

# Dùng đường dẫn cache cụ thể
python main.py --use-feature-cache --feature-cache-path results/cache/features_efficientnet_v2_l_full.npz
```

## Tái sử dụng feature đã trích xuất

Bạn có thể lưu lại `feature vectors` và `true labels` để không cần trích xuất lại từ ảnh ở các lần chạy sau.

### 1) Lần chạy đầu: tạo cache

```bash
python main.py --device gpu --save-feature-cache
```

Mặc định cache được lưu tại:

```text
results/cache/features_<model_name>_<sample|full>.npz
```

Ví dụ:

- `results/cache/features_efficientnet_v2_l_full.npz`
- `results/cache/features_efficientnet_v2_l_sample500.npz`

### 2) Lần chạy sau: dùng cache

```bash
python main.py --use-feature-cache
```

Khi dùng cache, pipeline sẽ bắt đầu từ bước xây dựng đồ thị + 4 thuật toán + đánh giá.

## Metrics đánh giá

| Metric         | Mô tả                                                                  |
| -------------- | ---------------------------------------------------------------------- |
| **NMI**        | Normalized Mutual Information - Đo lường thông tin chung giữa clusters |
| **ARI**        | Adjusted Rand Index - Đo lường sự tương đồng có điều chỉnh             |
| **Purity**     | Độ thuần khiết của các cluster                                         |
| **Modularity** | Chất lượng cấu trúc cộng đồng trên đồ thị                              |

## Tham số có thể điều chỉnh (Fine-tuning)

Trong `src/config.py`:

```python
class Config:
    # Feature Extraction
    MODEL_NAME = 'efficientnet_v2_l'  # Model backbone
    BATCH_SIZE = 32

    # Graph Construction
    K_NEIGHBORS = 10        # Số láng giềng k-NN (quan trọng!)
    SIMILARITY_METRIC = 'cosine'

    # Clustering
    LEIDEN_RESOLUTION = 1.0   # Tăng → nhiều clusters nhỏ
    LOUVAIN_RESOLUTION = 1.0
```

### Quick Tips để cải thiện kết quả:

1. **Tăng K_NEIGHBORS** lên 20-30
2. **Tune LEIDEN_RESOLUTION** theo số classes (~ N_classes / 10)
3. **L2 normalize features** trước khi build graph

📚 **Chi tiết đầy đủ**: Xem [docs/FINETUNING_GUIDE.md](docs/FINETUNING_GUIDE.md)

## Tài liệu

| File                                                 | Mô tả                           |
| ---------------------------------------------------- | ------------------------------- |
| [docs/PIPELINE_DETAIL.md](docs/PIPELINE_DETAIL.md)   | Chi tiết từng bước của pipeline |
| [docs/FINETUNING_GUIDE.md](docs/FINETUNING_GUIDE.md) | Hướng dẫn tối ưu hóa kết quả    |
