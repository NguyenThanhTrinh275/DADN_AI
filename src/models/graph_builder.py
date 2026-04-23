"""
Module xây dựng đồ thị tương đồng từ feature vectors
"""
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph
import igraph as ig

from src.config import config


def build_knn_graph(features, k=None, metric=None):
    """
    Xây dựng k-NN graph từ feature vectors
    
    Quy trình:
    1. Tính độ tương đồng/khoảng cách giữa các vector
    2. Mỗi node kết nối với k láng giềng gần nhất
    3. Chuyển thành undirected graph (symmetric)
       
    Args:
        features: Ma trận đặc trưng (n_samples, n_features)
        k: Số láng giềng gần nhất (mặc định từ config)
        metric: Độ đo khoảng cách ('cosine', 'euclidean')
    
    Returns:
        igraph.Graph: Đồ thị tương đồng
    """
    if k is None:
        k = config.K_NEIGHBORS
    if metric is None:
        metric = config.SIMILARITY_METRIC
    
    print(f"Building {k}-NN graph with {metric} similarity...")
    print(f"Input: {features.shape[0]} nodes, {features.shape[1]} dimensions")
    
    # Tạo k-NN graph theo distance, sau đó chuyển thành similarity làm edge weight
    dist_matrix = kneighbors_graph(
        features, 
        n_neighbors=k,
        mode='distance',
        metric=metric,
        include_self=False
    )

    dist_matrix = dist_matrix.tocoo()
    if metric == 'cosine':
        # cosine distance trong [0, 2], similarity = 1 - distance
        similarities = 1.0 - dist_matrix.data
    else:
        # Chuyển distance sang similarity dương
        similarities = 1.0 / (1.0 + dist_matrix.data)

    weighted = coo_matrix((similarities, (dist_matrix.row, dist_matrix.col)), shape=dist_matrix.shape)
    weighted = weighted.maximum(weighted.T).tocoo()

    mask = weighted.row != weighted.col
    sources = weighted.row[mask]
    targets = weighted.col[mask]
    weights = weighted.data[mask]

    edges = list(zip(sources.tolist(), targets.tolist()))
    g = ig.Graph(n=features.shape[0], edges=edges, directed=False)
    g.es['weight'] = weights.tolist()

    g.simplify(combine_edges='max')  # Gộp cạnh trùng, giữ similarity lớn hơn
    
    print(f"Graph created: {g.vcount()} nodes, {g.ecount()} edges")
    print(f"Average degree: {np.mean(g.degree()):.2f}")
    if g.ecount() > 0:
        edge_weights = np.array(g.es['weight'])
        print(f"Edge weight range: [{edge_weights.min():.4f}, {edge_weights.max():.4f}]")
    
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
