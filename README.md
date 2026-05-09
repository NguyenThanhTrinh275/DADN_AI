<!-- @format -->

# Đề tài 3: Community Structure Identification

## Tên đề tài

- **Tiếng Việt**: Xử lý bài toán Community Structure Identification sử dụng các thuật toán trong Khoa học Máy tính
- **Tiếng Anh**: Solving the Community Structure Identification problem using algorithms in Computer Science

## Mục tiêu

- Áp dụng các thuật toán phân cụm cộng đồng cho bài toán phân cụm hình ảnh
- Dataset: ImageNet-Hard (parquet: bytes ảnh + nhãn, có thể multi-label)
- So sánh và tối ưu hóa ít nhất 4 thuật toán (Infomap, Leiden, Louvain, LPA)

## Cấu trúc dự án

```
DADN_TTNT/
│
├── main.py                      # Entry point — pipeline đầy đủ
├── tune_params.py               # Grid search k và resolution (Leiden/Louvain), đọc feature cache
├── scripts/
│   └── extract_features_only.py # Chỉ extract + lưu .npz (GPU/Kaggle)
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py                # Cấu hình trung tâm (model, k-NN, PCA, TTA, resolution...)
│   ├── models/
│   │   ├── feature_extractor.py # EfficientNet / ResNet50 / DINOv2 (torch.hub), AMP, TTA
│   │   ├── graph_builder.py     # k-NN có trọng số, mutual k-NN, sim_power
│   │   └── clustering.py        # Infomap, Leiden, Louvain, LPA
│   └── utils/
│       ├── data_loader.py
│       ├── evaluation.py      # NMI, Accuracy (Hungarian), Purity, ARI, Modularity
│       ├── feature_cache.py
│       ├── feature_postprocess.py  # PCA whitening (sau extract hoặc sau load cache)
│       ├── result_logger.py
│       └── visualization.py    # comparison, radar, distribution, clusters.png
│
├── data/                        # ImageNet-Hard (*.parquet) — cấu hình DATA_PATH
├── results/
│   ├── results.csv
│   ├── comparison.png
│   ├── radar_chart.png
│   ├── clusters.png
│   ├── cluster_distribution.png
│   └── cache/
│       └── features_<model>_full.npz
│
├── docs/
│   ├── PIPELINE_DETAIL.md
│   ├── FINETUNING_GUIDE.md
│   └── LOGGING_GUIDE.md
│
└── notebooks/
    ├── EDA.ipynb
    ├── image_size_analysis.ipynb
    └── kaggle_feature_extract.ipynb   # Extract trên Kaggle → .npz
```

## Các thuật toán được implement

| # | Thuật toán  | Mô tả |
| --- | ----------- | ----- |
| 1 | **Infomap** | Random walk, Map Equation |
| 2 | **Leiden**  | Modularity + resolution (leidenalg) |
| 3 | **Louvain** | Multilevel + resolution (igraph) |
| 4 | **LPA**     | Label Propagation |

## Pipeline xử lý (tóm tắt)

```
Parquet → [Feature extraction ± TTA] → L2 normalize → [.npz cache RAW]
         → [PCA whitening nếu bật] → k-NN graph (cosine, mutual, sim_power)
         → 4 thuật toán → Metrics → plots + CSV
```

- **Feature**: torchvision (EfficientNet-V2-L, ResNet50) hoặc **DINOv2** qua `torch.hub` (input 518 cho ViT/14). Trên CUDA dùng mixed precision.
- **TTA / PCA**: bật trong [`src/config.py`](src/config.py) (`USE_TTA`, `USE_PCA_WHITEN`, `PCA_DIM`).
- **Đồ thị**: `K_NEIGHBORS`, `MUTUAL_KNN`, `SIM_POWER` — xem [`src/models/graph_builder.py`](src/models/graph_builder.py).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy chương trình

```bash
# Toàn bộ dữ liệu (theo DATA_PATH và SAMPLE_SIZE trong config / CLI)
python main.py

# Giới hạn số mẫu (debug)
python main.py --sample 500

# Thiết bị extract
python main.py --device auto    # mặc định
python main.py --device cpu
python main.py --device cuda

# Lưu feature cache (RAW trong .npz, chưa PCA — PCA áp sau khi load)
python main.py --save-feature-cache

# Chạy từ cache (bỏ qua extract; vẫn PCA + graph + clustering nếu bật trong config)
python main.py --use-feature-cache

# Đường dẫn cache tùy chỉnh
python main.py --use-feature-cache --feature-cache-path results/cache/features_dinov2_vitl14_full.npz

# Tắt ghi CSV
python main.py --no-log

# File CSV log tùy chỉnh
python main.py --log-path results/my_runs.csv
```

### clusters.png khi dùng `--use-feature-cache`

Pipeline **đọc lại parquet** (cùng `--sample` và `DATA_PATH` như lúc tạo cache) để lấy bytes ảnh, rồi vẽ `results/clusters.png`. Nếu không có parquet hoặc số dòng không khớp cache, mosaic sẽ bị bỏ qua (có cảnh báo).

## Tái sử dụng feature đã trích xuất

### 1) Tạo cache

```bash
python main.py --device cuda --save-feature-cache
```

Mặc định:

```text
results/cache/features_<MODEL_NAME>_<full|sample{N}>.npz
```

### 2) Dùng cache

```bash
python main.py --use-feature-cache
```

### 3) Extract chỉ để lấy .npz (máy yếu / Kaggle)

```bash
python scripts/extract_features_only.py --data-path data --output results/cache/features_<model>_full.npz
```

Chi tiết notebook: [`notebooks/kaggle_feature_extract.ipynb`](notebooks/kaggle_feature_extract.ipynb).

## Metrics đánh giá

| Metric | Mô tả |
| ------ | ----- |
| **NMI** | Normalized Mutual Information |
| **Accuracy** | Clustering accuracy (Hungarian / optimal assignment) |
| **Purity** | Purity; hỗ trợ multi-label (relaxed purity) |
| **ARI** | Adjusted Rand Index |
| **Modularity** | Trên đồ thị có trọng số |

Biểu đồ so sánh và radar gồm đủ các metric trên (trừ Modularity trên radar được clip không âm khi vẽ).

## Tham số chính

Chỉnh trong [`src/config.py`](src/config.py):

- **Model / extract**: `MODEL_NAME`, `BATCH_SIZE`, `USE_TTA`, `TTA_SCALES`, `TTA_FLIPS`
- **Hậu xử lý vector**: `USE_PCA_WHITEN`, `PCA_DIM`
- **Đồ thị**: `K_NEIGHBORS`, `SIMILARITY_METRIC`, `MUTUAL_KNN`, `SIM_POWER`
- **Phân cụm**: `LEIDEN_RESOLUTION`, `LOUVAIN_RESOLUTION`
- **Dữ liệu**: `DATA_PATH`, `RESULTS_PATH`, `SAMPLE_SIZE` (hoặc `--sample` khi chạy)

Tuning k và resolution: chạy [`tune_params.py`](tune_params.py) (đọc feature cache; composite score ưu tiên accuracy — xem docstring trong file).

### Gợi ý nhanh

1. Đồng bộ `--sample` và tên file cache (`full` vs `sample500`).
2. `DATA_PATH` trỏ thư mục chứa `*.parquet` (glob toàn bộ file trong thư mục).
3. ImageNet-Hard: số class ~830 — có thể tune resolution để số cluster gần số class khi cần accuracy.

## Tài liệu

| File | Mô tả |
| ---- | ----- |
| [docs/PIPELINE_DETAIL.md](docs/PIPELINE_DETAIL.md) | Chi tiết từng bước |
| [docs/FINETUNING_GUIDE.md](docs/FINETUNING_GUIDE.md) | Gợi ý tối ưu tham số |
| [docs/LOGGING_GUIDE.md](docs/LOGGING_GUIDE.md) | Cấu trúc CSV `results.csv` |
