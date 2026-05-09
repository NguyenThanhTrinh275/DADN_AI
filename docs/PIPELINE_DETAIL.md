<!-- @format -->

# Pipeline Chi Tiết: Community Structure Identification

## Tổng Quan Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMMUNITY STRUCTURE IDENTIFICATION                   │
│                          FOR IMAGE CLUSTERING                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐
│   STEP 1    │    │     STEP 2      │    │     STEP 3      │    │   STEP 4    │
│  Load Data  │───►│    Feature      │───►│  Build Graph    │───►│  Community  │
│  (Parquet)  │    │   Extraction    │    │    (k-NN)       │    │  Detection  │
└─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────┘
      │                    │                       │                     │
      ▼                    ▼                       ▼                     ▼
 ~11,000 ảnh         1280-dim vectors         Similarity Graph      4 Algorithms
 100 classes         EfficientNet-V2-L        Cosine Similarity     Infomap, etc.
                                                                         │
                                                                         ▼
                                           ┌─────────────────┐    ┌─────────────┐
                                           │   STEP 5        │◄───│  Cluster    │
                                           │   Evaluation    │    │   Labels    │
                                           │   & Compare     │    │             │
                                           └─────────────────┘    └─────────────┘
                                                    │
                                                    ▼
                                             NMI, ARI, Purity
                                                Modularity
```

---

## STEP 1: Data Loading (data_loader.py)

### Mô tả

Tải dữ liệu từ các file parquet của ImageNet-Hard dataset.

### Input

- Các file `.parquet` trong folder `data/`
- Mỗi file chứa: `image` (bytes), `label`, `english_label`, `origin`

### Output

- `images`: List các bytes ảnh
- `true_labels`: Ground truth labels

### Code Flow

```python
# 1. Tìm tất cả file parquet
all_files = glob.glob("data/*.parquet")

# 2. Concat tất cả thành 1 DataFrame
df = pd.concat([pd.read_parquet(f) for f in all_files])

# 3. Extract images và labels
images = df['image'].apply(lambda x: x['bytes']).tolist()
labels = df['label'].apply(lambda x: x[0]).values
```

### Điểm có thể tối ưu

- **Sampling strategy**: Stratified sampling để đảm bảo cân bằng classes
- **Data augmentation**: Có thể thêm augmented images để tăng data

---

## STEP 2: Feature Extraction (feature_extractor.py)

### Mô tả

Sử dụng pre-trained CNN để chuyển ảnh thành vector số cao chiều.

### Tại sao dùng EfficientNet-V2-L?

| Model                 | ImageNet Acc | Params | Feature Dim |
| --------------------- | ------------ | ------ | ----------- |
| ResNet50              | 76.1%        | 25M    | 2048        |
| EfficientNet-B7       | 84.3%        | 66M    | 2560        |
| **EfficientNet-V2-L** | **85.7%**    | 118M   | 1280        |

EfficientNet-V2-L có accuracy cao nhất và feature dimension vừa phải.

### Code Flow

```python
# 1. Load pre-trained model
model = efficientnet_v2_l(weights=DEFAULT)
model.classifier = nn.Identity()  # Loại bỏ classification head

# 2. Transform ảnh
transform = Compose([
    Resize(512),
    CenterCrop(480),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Extract features
features = model(batch_images)  # Shape: (batch_size, 1280)
```

### Điểm có thể tối ưu

- **Model selection**: Thử các backbone khác (DINOv2, CLIP, etc.)
- **Feature normalization**: L2-normalize vectors
- **Multi-scale features**: Kết hợp features từ nhiều layers
- **Fine-tuning backbone**: Unfreeze một số layers và train

---

## STEP 3: Graph Construction (graph_builder.py)

### Mô tả

Xây dựng đồ thị tương đồng từ feature vectors sử dụng k-Nearest Neighbors.

### Tại sao dùng k-NN Graph?

- Giảm noise: Chỉ giữ k kết nối mạnh nhất
- Sparse graph: Hiệu quả với dữ liệu lớn
- Local structure: Bảo toàn cấu trúc cục bộ

### Code Flow

```python
# 1. Tính k-NN adjacency matrix
adj = kneighbors_graph(features, n_neighbors=k, metric='cosine')

# 2. Chuyển thành symmetric (undirected)
adj = adj + adj.T
adj[adj > 1] = 1

# 3. Tạo igraph
edges = list(zip(*adj.nonzero()))
graph = ig.Graph(edges=edges, directed=False)
```

### Cosine Similarity vs Euclidean

```
Cosine Similarity: sim(A, B) = (A · B) / (|A| × |B|)
- Đo góc giữa 2 vectors
- Không phụ thuộc vào magnitude
- Phù hợp với high-dimensional data ✓

Euclidean Distance: dist(A, B) = ||A - B||
- Đo khoảng cách tuyệt đối
- Bị ảnh hưởng bởi magnitude
- Có thể bị curse of dimensionality
```

### Điểm có thể tối ưu (QUAN TRỌNG)

- **k value**: Tăng k → sparse graph → ít clusters, Giảm k → dense graph → nhiều clusters
- **Weighted edges**: Sử dụng similarity score làm edge weight
- **Mutual k-NN**: Chỉ kết nối nếu cả 2 nodes đều là k-NN của nhau
- **Threshold-based**: Chỉ giữ edges có similarity > threshold

---

## STEP 4: Community Detection (clustering.py)

### 4 Thuật Toán Được Implement

#### 1. INFOMAP

```
┌──────────────────────────────────────────────────────────────┐
│ Nguyên lý: Information Theory + Random Walk                  │
│                                                              │
│ - Mô hình hóa random walk trên đồ thị                        │
│ - Tối thiểu hóa Map Equation (description length)            │
│ - Communities = vùng mà walker dành nhiều thời gian          │
│                                                              │
│ Ưu điểm: Phát hiện cấu trúc hierarchical                     │
│ Nhược điểm: Chậm với đồ thị rất lớn                          │
└──────────────────────────────────────────────────────────────┘
```

#### 2. LEIDEN

```
┌──────────────────────────────────────────────────────────────┐
│ Nguyên lý: Cải tiến của Louvain                              │
│                                                              │
│ 3 Phases:                                                    │
│ 1. Local moving: Di chuyển nodes giữa communities            │
│ 2. Refinement: Đảm bảo communities connected tốt             │
│ 3. Aggregation: Gộp communities thành super-nodes            │
│                                                              │
│ Resolution parameter γ:                                      │
│ - γ < 1: Ít clusters lớn                                     │
│ - γ = 1: Cân bằng                                            │
│ - γ > 1: Nhiều clusters nhỏ                                  │
│                                                              │
│ Ưu điểm: Kết quả ổn định, không có poorly connected          │
└──────────────────────────────────────────────────────────────┘
```

#### 3. LOUVAIN

```
┌──────────────────────────────────────────────────────────────┐
│ Nguyên lý: Greedy Modularity Optimization                    │
│                                                              │
│ Modularity Q = Σ [(edges trong community) -                 │
│                   (expected edges if random)]                │
│                                                              │
│ 2 Phases:                                                    │
│ 1. Đưa mỗi node vào community tăng Q nhiều nhất             │
│ 2. Gộp communities thành nodes mới, lặp lại                 │
│                                                              │
│ Ưu điểm: Rất nhanh O(n log n)                               │
│ Nhược điểm: Có thể tạo poorly connected communities         │
└──────────────────────────────────────────────────────────────┘
```

#### 4. LABEL PROPAGATION (LPA)

```
┌──────────────────────────────────────────────────────────────┐
│ Nguyên lý: Lan truyền nhãn                                  │
│                                                              │
│ Algorithm:                                                   │
│ 1. Mỗi node có nhãn riêng                                   │
│ 2. Mỗi iteration: node cập nhật nhãn = mode của láng giềng  │
│ 3. Lặp đến khi ổn định                                      │
│                                                              │
│ Ưu điểm: Cực nhanh, near-linear time                        │
│ Nhược điểm: Kết quả không ổn định (random order)            │
└──────────────────────────────────────────────────────────────┘
```

---

## STEP 5: Evaluation (evaluation.py)

### Các Metrics

| Metric         | Range    | Ý nghĩa                            | Công thức                                   |
| -------------- | -------- | ---------------------------------- | ------------------------------------------- |
| **NMI**        | [0,1]    | Thông tin chung giữa pred và true  | I(C;K) / sqrt(H(C)×H(K))                    |
| **ARI**        | [-1,1]   | Rand Index có điều chỉnh           | (RI - Expected_RI) / (max_RI - Expected_RI) |
| **Purity**     | [0,1]    | % samples đúng class trong cluster | Σ max(class count) / N                      |
| **Modularity** | [-0.5,1] | Chất lượng cấu trúc community      | Σ (e_ii - a_i²)                             |

### Diễn giải kết quả

- **NMI > 0.5**: Tốt
- **ARI > 0.3**: Tốt
- **Purity > 0.6**: Chấp nhận được (nhưng có thể misleading nếu clusters quá nhỏ)
- **Modularity > 0.3**: Có cấu trúc cộng đồng rõ ràng

---

## Chi Tiết Từng Module

### config.py - Cấu hình trung tâm

```python
class Config:
    # Data
    DATA_PATH = 'data'
    RESULTS_PATH = 'results'

    # Feature Extraction
    MODEL_NAME = 'efficientnet_v2_l'
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Graph
    K_NEIGHBORS = 10
    SIMILARITY_METRIC = 'cosine'

    # Clustering
    LEIDEN_RESOLUTION = 1.0
    LOUVAIN_RESOLUTION = 1.0
```

### Luồng dữ liệu

```
Parquet Files
     │
     ▼
DataFrame (image bytes, labels)
     │
     ▼
List[bytes] ─────┐
     │           │
     ▼           │
PIL Images       │
     │           │
     ▼           │
Tensors          │
     │           │
     ▼           │
CNN Features ────┼──► k-NN Graph ──► Communities ──► Evaluation
(1280-dim)       │         │              │              │
     │           │         ▼              ▼              ▼
     │           │    Edges list     Labels array    Metrics dict
     │           │                                        │
     ▼           ▼                                        ▼
Comparison with Ground Truth ─────────────────────► Final Report
```
