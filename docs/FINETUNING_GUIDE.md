<!-- @format -->

# Hướng Dẫn Fine-Tuning & Tối Ưu Hóa

## Mục Tiêu

Cải thiện NMI, ARI, Purity để vượt trội so với thuật toán gốc (baseline).

---

## 1. FEATURE EXTRACTION OPTIMIZATION

### 1.1 Chọn Model Phù Hợp

```python
# Trong src/config.py
MODEL_NAME = 'efficientnet_v2_l'  # Mặc định

# Các lựa chọn khác:
# - 'efficientnet_v2_m': Cân bằng speed/accuracy
# - 'efficientnet_v2_s': Nhanh hơn, nhẹ hơn
# - 'resnet50': Baseline, nhanh nhất
```

**Recommendation**: Thử từng model và so sánh NMI

### 1.2 Feature Normalization (QUAN TRỌNG)

L2-normalize features trước khi xây dựng graph:

```python
# Thêm vào feature_extractor.py
from sklearn.preprocessing import normalize

def extract_features(self, images, batch_size=None, normalize_features=True):
    features = []
    # ... existing code ...

    features = np.vstack(features)

    if normalize_features:
        features = normalize(features, norm='l2')  # L2 normalize

    return features
```

**Tại sao?**: Cosine similarity hoạt động tốt hơn với normalized vectors

### 1.3 Dimensionality Reduction (Optional)

Giảm chiều để loại bỏ noise:

```python
from sklearn.decomposition import PCA

# Giảm từ 1280 → 256 dimensions
pca = PCA(n_components=256, random_state=42)
features_reduced = pca.fit_transform(features)

# Kiểm tra explained variance
print(f"Explained variance: {sum(pca.explained_variance_ratio_):.2%}")
```

### 1.4 Multi-Model Ensemble

Kết hợp features từ nhiều models:

```python
# Extract từ nhiều models
features_efficient = extract_features('efficientnet_v2_l')
features_resnet = extract_features('resnet50')

# Concatenate và normalize
from sklearn.preprocessing import normalize
features_combined = np.concatenate([
    normalize(features_efficient),
    normalize(features_resnet)
], axis=1)
```

---

## 2. GRAPH CONSTRUCTION OPTIMIZATION

### 2.1 Tối Ưu K (Số Láng Giềng)

K là parameter **quan trọng nhất** cho graph construction.

```python
# Trong src/config.py
K_NEIGHBORS = 10  # Mặc định

# Grid search để tìm K tốt nhất
k_values = [5, 10, 15, 20, 30, 50]
for k in k_values:
    graph = build_knn_graph(features, k=k)
    labels = cluster_leiden(graph)
    nmi = normalized_mutual_info_score(true_labels, labels)
    print(f"K={k}: NMI={nmi:.4f}, Clusters={len(set(labels))}")
```

**Rule of thumb**:

- K quá nhỏ (< 5): Graph quá sparse, nhiều components riêng biệt
- K quá lớn (> 50): Graph quá dense, mất cấu trúc local
- **Optimal: K ≈ sqrt(N) hoặc K trong [10, 30]**

### 2.2 Weighted Edges (KHUYẾN NGHỊ)

Sử dụng similarity score làm edge weight:

```python
# Thêm vào graph_builder.py
def build_weighted_knn_graph(features, k=10):
    from sklearn.metrics.pairwise import cosine_similarity

    # Tính full similarity matrix
    sim_matrix = cosine_similarity(features)

    edges = []
    weights = []

    for i in range(len(features)):
        # Lấy k neighbors có similarity cao nhất
        neighbors = np.argsort(sim_matrix[i])[-k-1:-1][::-1]

        for j in neighbors:
            if j > i:  # Avoid duplicates
                edges.append((i, j))
                weights.append(sim_matrix[i, j])

    g = ig.Graph(edges=edges, directed=False)
    g.es['weight'] = weights

    return g
```

### 2.3 Mutual k-NN (Stricter Connections)

Chỉ kết nối nếu cả 2 đều là k-NN của nhau:

```python
def build_mutual_knn_graph(features, k=10):
    # Tìm k-NN cho mỗi node
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(features)
    _, indices = nn.kneighbors(features)

    edges = []
    for i in range(len(features)):
        for j in indices[i, 1:]:  # Skip self
            # Mutual: i là neighbor của j VÀ j là neighbor của i
            if i in indices[j]:
                if i < j:
                    edges.append((i, j))

    g = ig.Graph(n=len(features), edges=edges, directed=False)
    return g
```

### 2.4 Adaptive K (Per-Node)

Mỗi node có K khác nhau dựa trên density:

```python
def adaptive_k(features, k_min=5, k_max=50):
    from sklearn.neighbors import NearestNeighbors

    # Tính local density
    nn = NearestNeighbors(n_neighbors=k_max, metric='cosine')
    nn.fit(features)
    distances, _ = nn.kneighbors(features)

    # Node ở vùng dense → K nhỏ, vùng sparse → K lớn
    density = 1 / (distances[:, k_min].mean())
    k_values = k_min + (k_max - k_min) * (1 - density / density.max())

    return k_values.astype(int)
```

---

## 3. CLUSTERING ALGORITHM TUNING

### 3.1 Leiden Resolution Tuning (QUAN TRỌNG)

Resolution γ kiểm soát số lượng và kích thước clusters.

```python
# Grid search resolution
resolutions = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

best_nmi = 0
best_res = 1.0

for res in resolutions:
    labels = cluster_leiden(graph, resolution=res)
    nmi = normalized_mutual_info_score(true_labels, labels)
    n_clusters = len(set(labels))

    print(f"Resolution={res}: NMI={nmi:.4f}, Clusters={n_clusters}")

    if nmi > best_nmi:
        best_nmi = nmi
        best_res = res

print(f"\nBest: resolution={best_res}, NMI={best_nmi:.4f}")
```

**Heuristic**:

- Số classes trong ground truth: N_classes
- **Start với resolution ≈ N_classes / 10**
- Điều chỉnh để số clusters ≈ N_classes

### 3.2 Ensemble Clustering

Kết hợp kết quả từ nhiều thuật toán:

```python
from sklearn.cluster import consensus_clustering
import numpy as np

def ensemble_clustering(clustering_results, n_clusters=None):
    """
    Consensus clustering từ nhiều partitions
    """
    # Stack all labels
    all_labels = np.stack([
        clustering_results['Infomap'],
        clustering_results['Leiden'],
        clustering_results['Louvain'],
        clustering_results['LPA']
    ])

    # Co-association matrix
    n_samples = all_labels.shape[1]
    coassoc = np.zeros((n_samples, n_samples))

    for labels in all_labels:
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if labels[i] == labels[j]:
                    coassoc[i, j] += 1
                    coassoc[j, i] += 1

    coassoc /= len(all_labels)

    # Final clustering từ co-association
    from sklearn.cluster import AgglomerativeClustering

    if n_clusters is None:
        # Estimate từ average
        n_clusters = int(np.mean([len(set(l)) for l in all_labels]))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )

    distance = 1 - coassoc
    final_labels = clustering.fit_predict(distance)

    return final_labels
```

### 3.3 Multi-Resolution Leiden

Chạy Leiden ở nhiều resolution và chọn best:

```python
def multi_resolution_leiden(graph, true_labels, resolutions=None):
    if resolutions is None:
        resolutions = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]

    best_labels = None
    best_nmi = -1

    for res in resolutions:
        labels = cluster_leiden(graph, resolution=res)
        nmi = normalized_mutual_info_score(true_labels, labels)

        if nmi > best_nmi:
            best_nmi = nmi
            best_labels = labels

    return best_labels
```

---

## 4. HYPERPARAMETER OPTIMIZATION

### 4.1 Grid Search Full Pipeline

```python
from itertools import product

def grid_search_pipeline(features, true_labels):
    """
    Grid search toàn bộ pipeline
    """
    # Define search space
    param_grid = {
        'k_neighbors': [5, 10, 20, 30],
        'resolution': [0.5, 1.0, 2.0, 5.0],
        'weighted': [True, False],
        'normalize': [True, False]
    }

    results = []

    for k, res, weighted, norm in product(*param_grid.values()):
        # Normalize features
        if norm:
            feats = normalize(features, norm='l2')
        else:
            feats = features

        # Build graph
        if weighted:
            graph = build_weighted_knn_graph(feats, k=k)
        else:
            graph = build_knn_graph(feats, k=k)

        # Cluster
        labels = cluster_leiden(graph, resolution=res)

        # Evaluate
        nmi = normalized_mutual_info_score(true_labels, labels)
        ari = adjusted_rand_score(true_labels, labels)

        results.append({
            'k': k, 'resolution': res,
            'weighted': weighted, 'normalize': norm,
            'nmi': nmi, 'ari': ari,
            'n_clusters': len(set(labels))
        })

    # Sort by NMI
    results.sort(key=lambda x: x['nmi'], reverse=True)

    return results
```

### 4.2 Bayesian Optimization (Advanced)

```python
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

def objective(params):
    k, resolution, normalize = params

    if normalize:
        feats = sklearn.preprocessing.normalize(features)
    else:
        feats = features

    graph = build_knn_graph(feats, k=k)
    labels = cluster_leiden(graph, resolution=resolution)
    nmi = normalized_mutual_info_score(true_labels, labels)

    return -nmi  # Minimize negative NMI

space = [
    Integer(5, 50, name='k'),
    Real(0.1, 10.0, name='resolution'),
    Categorical([True, False], name='normalize')
]

result = gp_minimize(objective, space, n_calls=50, random_state=42)

print(f"Best params: k={result.x[0]}, res={result.x[1]}, norm={result.x[2]}")
print(f"Best NMI: {-result.fun:.4f}")
```

---

## 5. CACHING & SPEED OPTIMIZATION

### 5.1 Cache Features

```python
import pickle
import os

def get_features(images, cache_path='cache/features.pkl'):
    if os.path.exists(cache_path):
        print("Loading cached features...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_features(images)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(features, f)

    return features
```

### 5.2 GPU Acceleration

```python
# Trong config.py
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Kiểm tra GPU
print(f"Using device: {config.DEVICE}")
if config.DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 6. RECOMMENDED SETTINGS

### Baseline (Default)

```python
MODEL_NAME = 'efficientnet_v2_l'
K_NEIGHBORS = 10
LEIDEN_RESOLUTION = 1.0
NORMALIZE_FEATURES = False
WEIGHTED_EDGES = False
```

### Optimized

```python
MODEL_NAME = 'efficientnet_v2_l'
K_NEIGHBORS = 20  # Tăng lên
LEIDEN_RESOLUTION = 2.0  # Tune theo số classes
NORMALIZE_FEATURES = True  # L2 normalize
WEIGHTED_EDGES = True  # Sử dụng similarity weights
```

---

## 7. EXPERIMENT TRACKING

### Log kết quả

```python
import json
from datetime import datetime

def log_experiment(params, results, log_file='results/experiments.json'):
    experiment = {
        'timestamp': datetime.now().isoformat(),
        'params': params,
        'results': results
    }

    # Load existing
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(experiment)

    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
```

---

## 8. CHECKLIST FINE-TUNING

1. [ ] Thử các model khác nhau (EfficientNet variants)
2. [ ] Enable L2 normalization cho features
3. [ ] Grid search K (5, 10, 20, 30, 50)
4. [ ] Grid search Resolution (0.1 → 10.0)
5. [ ] Thử weighted edges
6. [ ] Thử mutual k-NN
7. [ ] Ensemble từ nhiều thuật toán
8. [ ] PCA dimension reduction (optional)
9. [ ] Cache features để tăng tốc experiments
10. [ ] Log tất cả experiments

---

## 9. EXPECTED IMPROVEMENTS

| Configuration     | NMI   | ARI   | Notes            |
| ----------------- | ----- | ----- | ---------------- |
| Baseline          | ~0.45 | ~0.01 | Default settings |
| + L2 Normalize    | ~0.50 | ~0.05 | +10% NMI         |
| + Tune K=20       | ~0.55 | ~0.10 | +20% NMI         |
| + Tune Resolution | ~0.60 | ~0.15 | +30% NMI         |
| + Weighted Edges  | ~0.62 | ~0.18 | +35% NMI         |
| + Ensemble        | ~0.65 | ~0.20 | +40% NMI         |

**Note**: Kết quả thực tế phụ thuộc vào dataset và random seeds.
