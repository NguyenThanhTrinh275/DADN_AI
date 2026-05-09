<!-- @format -->

# Hướng dẫn fine-tuning và tối ưu

## Mục tiêu

Cải thiện **NMI, Accuracy, ARI, Purity** (và modularity trên đồ thị) bằng cách chỉnh backbone, đồ thị k-NN, và resolution Leiden/Louvain.

Tham số trung tâm: [`src/config.py`](../src/config.py). Grid search có sẵn: [`tune_params.py`](../tune_params.py) (đọc feature cache `.npz`, tune `k` và resolution; composite score trong file có trọng số accuracy).

---

## 1. Feature extraction

### 1.1 Model (MODEL_NAME)

Được khai báo trong [`src/models/feature_extractor.py`](../src/models/feature_extractor.py) (`SUPPORTED_MODELS`):

| MODEL_NAME | Input | Chiều vector |
| ---------- | ----- | ------------- |
| `efficientnet_v2_l` | 480 | 1280 |
| `resnet50` | 224 | 2048 |
| `dinov2_vits14` | 518 | 384 |
| `dinov2_vitb14` | 518 | 768 |
| `dinov2_vitl14` | 518 | 1024 |

DINOv2 tải qua **Internet** (torch.hub) lần đầu. Trên GPU, extract dùng **AMP**.

### 1.2 Chuẩn hóa — đã có trong code

Sau extract, vector được **L2-normalize** trong `FeatureExtractor.extract_features` / TTA — không cần chèn thêm bước thủ công.

### 1.3 TTA (Test-Time Augmentation)

Trong `config`: `USE_TTA`, `TTA_SCALES`, `TTA_FLIPS`. Bật để ổn định feature (đổi giá phải extract lại hoặc có cache mới).

### 1.4 PCA whitening

Trong `config`: `USE_PCA_WHITEN`, `PCA_DIM`. Được gọi trong [`main.py`](../main.py) sau extract **và** sau khi load cache — giảm nhiễu chiều trước k-NN.

### 1.5 Cache để thử nhanh graph/cluster

```bash
python main.py --save-feature-cache
python main.py --use-feature-cache
```

Hoặc chỉ tạo `.npz` trên máy mạnh / Kaggle:

```bash
python scripts/extract_features_only.py --data-path data --output results/cache/features_<model>_full.npz
```

---

## 2. Đồ thị k-NN

### 2.1 Tham số trong config

- **`K_NEIGHBORS`**: ảnh hưởng mật độ đồ thị và số cluster sau Leiden/Louvain.
- **`SIMILARITY_METRIC`**: thường `cosine`.
- **`MUTUAL_KNN`**: chỉ giữ cạnh khi hai chiều đều là k-NN (đồ thị thưa hơn).
- **`SIM_POWER`**: lũy thừa similarity để nhấn mạnh láng giềng rất giống.

Cài đặt thực tế: [`src/models/graph_builder.py`](../src/models/graph_builder.py) (`build_knn_graph`).

### 2.2 Grid search k và resolution

Chạy sau khi đã có file cache đúng model:

```bash
python tune_params.py --model dinov2_vitl14
```

Chỉnh grid trong `tune_params.py` (`COARSE_K_VALUES`, `COARSE_RESOLUTION_VALUES`). Kết quả gợi ý cập nhật `K_NEIGHBORS`, `LEIDEN_RESOLUTION`, `LOUVAIN_RESOLUTION` trong `config`.

### 2.3 Heuristic resolution (ImageNet-Hard ~830 class)

Thường cần **số cluster** gần số class để metric accuracy (Hungarian) không bị “trần”. Xem báo cáo trong `print_report` của `tune_params.py` (top theo accuracy và độ lệch `n_clusters` so với `N_TARGET_CLASSES`).

---

## 3. Thuật toán phân cụm

- **Leiden / Louvain**: nhạy với `resolution` — tăng → thường nhiều cluster nhỏ hơn (phụ thuộc đồ thị).
- **Infomap / LPA**: không dùng hai tham số resolution trên; vẫn được log cùng các cột config trong CSV để so sánh lịch sử.

---

## 4. Đánh giá

Metrics đầy đủ: [`src/utils/evaluation.py`](../src/utils/evaluation.py). Đặc biệt **Accuracy** cần alignment cluster–class (Hungarian); multi-label dùng proxy nhãn đơn giống NMI/ARI.

---

## 5. Checklist thực nghiệm

1. [ ] Cố định một backbone + extract cache (full hoặc sample có chủ đích).
2. [ ] Bật/tắt TTA và PCA; thử `PCA_DIM` ∈ {128, 256, 384}.
3. [ ] Tune `K_NEIGHBORS`, `MUTUAL_KNN`, `SIM_POWER`.
4. [ ] Chạy `tune_params.py`, áp bộ (`k`, resolution) tốt vào `config`.
5. [ ] So sánh `results.csv` / biểu đồ trong `results/`.

---

## 6. Ý tưởng nâng cao (chưa có sẵn trong repo)

Các hướng sau là **gợi ý** nếu mở rộng sau: ensemble nhiều partition, Bayesian optimization, adaptive k theo mật độ local, v.v. Code hiện tại không bao gồm các module đó; có thể thêm notebook thử nghiệm riêng.

**Lưu ý:** Bảng “expected improvements” mang tính minh họa; kết quả thực tế phụ thuộc dataset, seed, và tham số graph/cluster.
