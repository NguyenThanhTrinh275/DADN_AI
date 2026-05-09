"""
Module xây dựng đồ thị tương đồng từ feature vectors
"""
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph
import igraph as ig

from src.config import config


def build_knn_graph(features, k=None, metric=None, mutual=None, sim_power=None):
    """
    Xây dựng k-NN graph từ feature vectors.

    Quy trình:
    1. Tính khoảng cách → similarity giữa các vector
    2. Mỗi node nối tới k láng giềng gần nhất
    3. Khuếch đại similarity bằng `sim_power` để giảm ảnh hưởng edge yếu
    4. Áp `mutual k-NN` (giữ edge khi cả 2 chiều cùng có) hoặc symmetric (max)
    5. Trả về undirected graph có trọng số

    Args:
        features: Ma trận đặc trưng (n_samples, n_features)
        k: Số láng giềng gần nhất (mặc định từ config)
        metric: Độ đo khoảng cách ('cosine', 'euclidean')
        mutual: True → mutual k-NN (giữ edge khi cả 2 chiều cùng có).
                False → symmetric union. Mặc định lấy từ config.
        sim_power: Lũy thừa khuếch đại similarity (mặc định từ config)

    Returns:
        igraph.Graph: Đồ thị tương đồng
    """
    if k is None:
        k = config.K_NEIGHBORS
    if metric is None:
        metric = config.SIMILARITY_METRIC
    if mutual is None:
        mutual = getattr(config, 'MUTUAL_KNN', False)
    if sim_power is None:
        sim_power = getattr(config, 'SIM_POWER', 1.0)

    mode_str = "mutual" if mutual else "symmetric"
    print(f"Building {k}-NN graph ({mode_str}, sim_power={sim_power}) with {metric} similarity...")
    print(f"Input: {features.shape[0]} nodes, {features.shape[1]} dimensions")

    dist_matrix = kneighbors_graph(
        features,
        n_neighbors=k,
        mode='distance',
        metric=metric,
        include_self=False,
    ).tocoo()

    if metric == 'cosine':
        similarities = np.clip(1.0 - dist_matrix.data, 0.0, 1.0)
    else:
        similarities = 1.0 / (1.0 + dist_matrix.data)

    if sim_power != 1.0:
        similarities = np.power(similarities, sim_power)

    n = features.shape[0]
    W = coo_matrix(
        (similarities, (dist_matrix.row, dist_matrix.col)),
        shape=(n, n),
    ).tocsr()

    if mutual:
        W_sym = W.minimum(W.T)
    else:
        W_sym = W.maximum(W.T)

    W_coo = W_sym.tocoo()
    mask = (W_coo.row < W_coo.col) & (W_coo.data > 0)
    sources = W_coo.row[mask]
    targets = W_coo.col[mask]
    weights = W_coo.data[mask]

    edges = list(zip(sources.tolist(), targets.tolist()))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es['weight'] = weights.tolist()

    print(f"Graph created: {g.vcount()} nodes, {g.ecount()} edges")
    print(f"Average degree: {np.mean(g.degree()):.2f}")
    if g.ecount() > 0:
        edge_weights = np.array(g.es['weight'])
        print(f"Edge weight range: [{edge_weights.min():.4f}, {edge_weights.max():.4f}]")
    n_components = len(g.connected_components())
    if n_components > 1:
        print(f"  ⚠ Đồ thị có {n_components} thành phần liên thông (mutual k-NN có thể quá thưa)")

    return g


def build_weighted_knn_graph(features, k=None, metric=None):
    """
    Xây dựng k-NN graph có trọng số (edge weight = similarity)
    
    Args:
        features: Ma trận đặc trưng
        k: Số láng giềng
        metric: Độ đo
    
    Returns:
        igraph.Graph với edge weights
    """
    return build_knn_graph(features, k=k, metric=metric)


def get_graph_statistics(graph):
    """
    Tính các thống kê của đồ thị
    
    Args:
        graph: igraph.Graph
    
    Returns:
        dict chứa các thống kê
    """
    stats = {
        'n_nodes': graph.vcount(),
        'n_edges': graph.ecount(),
        'density': graph.density(),
        'avg_degree': np.mean(graph.degree()),
        'max_degree': max(graph.degree()),
        'min_degree': min(graph.degree()),
        'n_components': len(graph.connected_components()),
        'is_connected': graph.is_connected(),
    }
    
    return stats
