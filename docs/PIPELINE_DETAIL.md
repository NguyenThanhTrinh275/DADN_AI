<!-- @format -->

# Pipeline chi tiết: Community Structure Identification

## Tổng quan

```
┌──────────────┐    ┌─────────────────────┐    ┌──────────────────┐    ┌─────────────┐
│  STEP 1      │    │  STEP 2             │    │  STEP 3          │    │  STEP 4     │
│  Load Data   │───►│  Feature extraction │───►│  Build k-NN graph│───►│ Community   │
│  (Parquet)   │    │  (+TTA, cache RAW)  │    │  weighted graph  │    │ detection   │
└──────────────┘    └─────────────────────┘    └──────────────────┘    └─────────────┘
                             │                         │
                             ▼                         ▼
                     PCA whitening                  Similarity on
                     (nếu USE_PCA_WHITEN)           cosine space
                             │                         │
                             └────────────┬────────────┘
                                          ▼
                               ┌─────────────────────┐
                               │  STEP 5             │
                               │  Evaluation + plots │
                               │  NMI, Acc, ARI...   │
                               └─────────────────────┘
```

Luồng tệp chính: [`main.py`](../main.py).

---

## STEP 1: Load data ([`src/utils/data_loader.py`](../src/utils/data_loader.py))

- Đọc mọi file `*.parquet` trong `config.DATA_PATH`, ghép một `DataFrame`.
- Cột điển hình: `image` (struct có `bytes`), `label` (có thể multi-value), `english_label`, ...
- `extract_images_and_labels(..., return_multilabel=True)` trả list bytes ảnh và `label_sets` cho đánh giá.

---

## STEP 2: Feature extraction ([`src/models/feature_extractor.py`](../src/models/feature_extractor.py))

**Backbone hỗ trợ** (xem `SUPPORTED_MODELS`):

- `efficientnet_v2_l` — 480px, 1280-d
- `resnet50` — 224px, 2048-d
- `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14` — 518px (ViT/14), qua `torch.hub.load('facebookresearch/dinov2', ...)`

**Sau forward**: vector được **L2-normalize** (sklearn).

**Tùy chọn config**:

- `USE_TTA`: trung bình feature qua `TTA_SCALES` × `TTA_FLIPS` (`extract_features_tta`).
- Trên CUDA: mixed precision (`torch.cuda.amp.autocast`) trong `extract_features` / TTA.

**Cache** ([`src/utils/feature_cache.py`](../src/utils/feature_cache.py)): lưu **RAW** sau extract (chưa PCA) để đổi `PCA_DIM` không cần extract lại.

---

## STEP 2b: PCA whitening ([`src/utils/feature_postprocess.py`](../src/utils/feature_postprocess.py))

- Nếu `USE_PCA_WHITEN`: `pca_whiten(features, PCA_DIM)` rồi L2 lại.
- Áp dụng **sau** extract **và** sau khi load `.npz` (cùng logic trong `main.py`), để graph/cluster luôn trên không gian đã whitening.

---

## STEP 3: Graph ([`src/models/graph_builder.py`](../src/models/graph_builder.py))

- `sklearn.neighbors.kneighbors_graph` (cosine hoặc euclidean), mode distance → similarity (cosine: `1 - d`, clip `[0,1]`).
- Trọng số có thể nâng lũy thừa: `SIM_POWER`.
- Symmetric: `maximum(W, W.T)` hoặc **mutual k-NN** khi `MUTUAL_KNN=True`: `minimum(W, W.T)`.
- Đồ thị `igraph` không hướng, `es['weight']`.

---

## STEP 4: Community detection ([`src/models/clustering.py`](../src/models/clustering.py))

- **Infomap**, **Leiden** (`LEIDEN_RESOLUTION`), **Louvain** (`LOUVAIN_RESOLUTION`), **LPA**.
- Louvain/Leiden dùng edge weights khi có.

---

## STEP 5: Evaluation ([`src/utils/evaluation.py`](../src/utils/evaluation.py))

| Metric | Ghi chú |
| ------ | ------- |
| NMI / ARI | Multi-label: nhãn đơn đại diện (min trong set đã sort) |
| Purity | Multi-label: relaxed purity |
| **Accuracy** | Hungarian alignment giữa cluster ID và class |
| Modularity | `igraph` trên cùng đồ thị đã build |

---

## Visualization ([`src/utils/visualization.py`](../src/utils/visualization.py))

- `comparison.png`: nhiều metric (gồm Accuracy).
- `radar_chart.png`
- `cluster_distribution.png`
- `clusters.png`: mosaic theo cluster; `cluster_pick` — largest-by-size hoặc random (khi reload ảnh từ cache).

**`--use-feature-cache`**: không có bytes ảnh trong RAM sau bước cache → `main.py` **đọc lại parquet** (cùng `--sample`, `DATA_PATH`) để vẽ `clusters.png` nếu số mẫu khớp `features`.

---

## Config trung tâm ([`src/config.py`](../src/config.py))

Các trường đại diện (giá trị số có thể thay khi fine-tune):

- `DATA_PATH`, `RESULTS_PATH`
- `MODEL_NAME`, `BATCH_SIZE`
- `K_NEIGHBORS`, `SIMILARITY_METRIC`, `MUTUAL_KNN`, `SIM_POWER`
- `USE_TTA`, `TTA_SCALES`, `TTA_FLIPS`
- `USE_PCA_WHITEN`, `PCA_DIM`
- `LEIDEN_RESOLUTION`, `LOUVAIN_RESOLUTION`
- `SAMPLE_SIZE`, `RANDOM_STATE`

Hyperparameter search: [`tune_params.py`](../tune_params.py) (grid `k` × resolution trên feature cache).

---

## Luồng dữ liệu (tóm tắt)

```
Parquet → bytes ảnh ──► CNN/DINOv2 ──► features RAW ──► [save .npz]
                                           │
                                           ▼
                                    PCA whiten (optional)
                                           │
                                           ▼
                              k-NN graph → 4 algorithms → metrics + figures
```
