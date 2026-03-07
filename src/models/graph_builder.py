"""
Module xây dựng đồ thị tương đồng từ feature vectors
"""
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
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
    
    # Tạo k-NN adjacency matrix
    adj_matrix = kneighbors_graph(
        features, 
        n_neighbors=k,
        mode='connectivity',
        metric=metric,
        include_self=False
    )
    
    # Chuyển thành symmetric (undirected graph)
    adj_matrix = adj_matrix + adj_matrix.T
    adj_matrix[adj_matrix > 1] = 1
    
    # Tạo igraph từ adjacency matrix
    sources, targets = adj_matrix.nonzero()
    edges = list(zip(sources, targets))
    
    g = ig.Graph(n=features.shape[0], edges=edges, directed=False)
    g.simplify()  # Loại bỏ self-loops và multiple edges
    
    print(f"Graph created: {g.vcount()} nodes, {g.ecount()} edges")
    print(f"Average degree: {np.mean(g.degree()):.2f}")
    
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
    if k is None:
        k = config.K_NEIGHBORS
    if metric is None:
        metric = config.SIMILARITY_METRIC
    
    print(f"Building weighted {k}-NN graph with {metric} similarity...")
    
    # Tính similarity matrix
    if metric == 'cosine':
        sim_matrix = cosine_similarity(features)
    else:
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix = euclidean_distances(features)
        # Chuyển distance thành similarity
        sim_matrix = 1 / (1 + dist_matrix)
    
    # Tạo k-NN graph từ similarity
    edges = []
    weights = []
    
    for i in range(len(features)):
        # Lấy k láng giềng gần nhất (không bao gồm chính nó)
        similarities = sim_matrix[i].copy()
        similarities[i] = -np.inf  # Loại bỏ self-loop
        
        top_k_indices = np.argsort(similarities)[-k:]
        
        for j in top_k_indices:
            if j > i:  # Tránh duplicate edges
                edges.append((i, j))
                weights.append(similarities[j])
    
    g = ig.Graph(n=len(features), edges=edges, directed=False)
    g.es['weight'] = weights
    
    print(f"Weighted graph: {g.vcount()} nodes, {g.ecount()} edges")
    
    return g


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
