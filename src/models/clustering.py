"""
Module các thuật toán phân cụm cộng đồng (Community Detection)

Bao gồm 4 thuật toán:
1. Infomap - Dựa trên lý thuyết thông tin
2. Leiden - Cải tiến của Louvain
3. Louvain - Tối ưu hóa modularity
4. LPA - Label Propagation Algorithm
"""
import random
import numpy as np
import leidenalg

from src.config import config


def _get_weights(graph):
    """Lấy weights nếu đồ thị có thuộc tính weight"""
    return graph.es['weight'] if 'weight' in graph.es.attributes() else None


def cluster_infomap(graph):
    """
    Thuật toán Infomap - dựa trên lý thuyết thông tin
    
    Nguyên lý:
    - Mô hình hóa random walk trên đồ thị
    - Tối thiểu hóa độ dài mô tả (Map Equation)
    - Các cộng đồng = các vùng mà random walker dành nhiều thời gian
    
    Ưu điểm:
    - Phát hiện cấu trúc phân cấp (hierarchical)
    - Không cần biết trước số cluster
    - Hiệu quả với flow-based communities
    
    Args:
        graph: igraph.Graph
    
    Returns:
        numpy array chứa cluster labels
    """
    print("Running Infomap...")
    weights = _get_weights(graph)
    
    try:
        from infomap import Infomap
        
        im = Infomap(silent=True)
        
        # Thêm các cạnh vào Infomap
        for edge in graph.es:
            source, target = edge.tuple
            if weights is not None:
                im.add_link(source, target, edge['weight'])
            else:
                im.add_link(source, target)
        
        im.run()
        
        # Lấy kết quả phân cụm (module_id bắt đầu từ 1 trong infomap package)
        labels = np.zeros(graph.vcount(), dtype=int)
        for node in im.tree:
            if node.is_leaf:
                labels[node.node_id] = node.module_id
        
    except ImportError:
        print("  → Infomap package không có, sử dụng igraph's implementation")
        communities = graph.community_infomap(edge_weights=weights)
        labels = np.array(communities.membership)
    
    # Chuẩn hóa labels về 0-indexed (infomap package trả từ 1, igraph trả từ 0)
    unique_ids = np.unique(labels)
    id_map = {old: new for new, old in enumerate(sorted(unique_ids))}
    labels = np.array([id_map[l] for l in labels])

    n_clusters = len(set(labels))
    print(f"  → Infomap: {n_clusters} clusters found")
    return labels


def cluster_leiden(graph, resolution=None):
    """
    Thuật toán Leiden - cải tiến của Louvain
    
    Nguyên lý:
    - Tối ưu hóa modularity như Louvain
    - Thêm bước refinement để đảm bảo communities connected tốt hơn
    - Tránh được vấn đề "poorly connected communities" của Louvain
    
    Ưu điểm:
    - Kết quả ổn định và chất lượng cao hơn Louvain
    - Đảm bảo các cộng đồng được kết nối tốt
    - Có thể điều chỉnh resolution
    
    Args:
        graph: igraph.Graph
        resolution: Tham số resolution (cao = nhiều cluster nhỏ)
    
    Returns:
        numpy array chứa cluster labels
    """
    if resolution is None:
        resolution = config.LEIDEN_RESOLUTION
    weights = _get_weights(graph)
    
    print(f"Running Leiden (resolution={resolution})...")
    
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution
    )
    
    labels = np.array(partition.membership)
    n_clusters = len(set(labels))
    print(f"  → Leiden: {n_clusters} clusters found")
    
    return labels


def cluster_louvain(graph, resolution=None):
    """
    Thuật toán Louvain - tối ưu hóa modularity
    
    Nguyên lý:
    - Greedy optimization của modularity
    - 2 phases: node moving & network aggregation
    - Lặp lại cho đến khi không tăng modularity được
    
    Ưu điểm:
    - Nhanh, hiệu quả với đồ thị lớn
    - Không cần biết trước số cluster
    
    Args:
        graph: igraph.Graph
        resolution: Tham số resolution
    
    Returns:
        numpy array chứa cluster labels
    """
    if resolution is None:
        resolution = config.LOUVAIN_RESOLUTION
    weights = _get_weights(graph)
    
    print(f"Running Louvain (resolution={resolution})...")
    
    random.seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    communities = graph.community_multilevel(weights=weights, resolution=resolution)
    labels = np.array(communities.membership)
    
    n_clusters = len(set(labels))
    print(f"  → Louvain: {n_clusters} clusters found")
    
    return labels


def cluster_lpa(graph):
    """
    Label Propagation Algorithm (LPA) - Thuật toán lan truyền nhãn
    
    Nguyên lý:
    - Mỗi node ban đầu có một nhãn riêng
    - Iteratively: mỗi node cập nhật nhãn = nhãn phổ biến nhất của láng giềng
    - Hội tụ khi không có node nào thay đổi nhãn
    
    Ưu điểm:
    - Rất nhanh, gần như linear time O(E)
    - Đơn giản, dễ hiểu
    - Không cần biết trước số cluster
    
    Nhược điểm:
    - Kết quả có thể không ổn định (do random order)
    - Có thể tạo ra cluster quá lớn
    
    Args:
        graph: igraph.Graph
    
    Returns:
        numpy array chứa cluster labels
    """
    print("Running Label Propagation Algorithm (LPA)...")
    weights = _get_weights(graph)

    random.seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    communities = graph.community_label_propagation(weights=weights)
    labels = np.array(communities.membership)
    
    n_clusters = len(set(labels))
    print(f"  → LPA: {n_clusters} clusters found")
    
    return labels


def run_all_algorithms(graph):
    """
    Chạy tất cả các thuật toán phân cụm
    
    Args:
        graph: igraph.Graph
    
    Returns:
        dict: {algorithm_name: labels}
    """
    print("\n" + "="*50)
    print("Running all clustering algorithms...")
    print("="*50)
    
    results = {}
    
    results['Infomap'] = cluster_infomap(graph)
    results['Leiden'] = cluster_leiden(graph)
    results['Louvain'] = cluster_louvain(graph)
    results['LPA'] = cluster_lpa(graph)
    
    print("\nSummary:")
    for algo, labels in results.items():
        print(f"  {algo}: {len(set(labels))} clusters")
    
    return results
