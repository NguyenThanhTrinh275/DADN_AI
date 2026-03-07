"""
Module đánh giá chất lượng phân cụm
"""
import numpy as np
from collections import Counter
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)


def compute_nmi(true_labels, pred_labels):
    """
    Normalized Mutual Information (NMI)
    
    Đo lường thông tin chung giữa phân cụm dự đoán và ground truth
    Range: [0, 1], 1 = hoàn hảo
    """
    return normalized_mutual_info_score(true_labels, pred_labels)


def compute_ari(true_labels, pred_labels):
    """
    Adjusted Rand Index (ARI)
    
    Đo lường sự tương đồng giữa hai phân cụm, có điều chỉnh cho random chance
    Range: [-1, 1], 1 = hoàn hảo, 0 = random
    """
    return adjusted_rand_score(true_labels, pred_labels)


def compute_purity(true_labels, pred_labels):
    """
    Purity Score
    
    Đo độ "thuần khiết" của các cluster
    Mỗi cluster đếm số phần tử thuộc class phổ biến nhất
    Range: [0, 1], 1 = mỗi cluster chỉ chứa 1 class
    """
    clusters = set(pred_labels)
    total_correct = 0
    
    for cluster_id in clusters:
        cluster_indices = np.where(pred_labels == cluster_id)[0]
        cluster_true_labels = true_labels[cluster_indices]
        most_common_count = Counter(cluster_true_labels).most_common(1)[0][1]
        total_correct += most_common_count
    
    return total_correct / len(true_labels)


def compute_modularity(graph, labels):
    """
    Modularity Score
    
    Đo chất lượng cấu trúc cộng đồng trên đồ thị
    So sánh số edges trong community vs random graph
    Range: [-0.5, 1], thường > 0.3 được coi là có cấu trúc cộng đồng
    """
    return graph.modularity(labels)


def compute_fmi(true_labels, pred_labels):
    """
    Fowlkes-Mallows Index
    
    Geometric mean của precision và recall
    Range: [0, 1]
    """
    return fowlkes_mallows_score(true_labels, pred_labels)


def compute_v_measure(true_labels, pred_labels):
    """
    V-Measure
    
    Harmonic mean của homogeneity và completeness
    Range: [0, 1]
    """
    return v_measure_score(true_labels, pred_labels)


def evaluate_clustering(true_labels, pred_labels, graph=None):
    """
    Đánh giá toàn diện chất lượng phân cụm
    
    Args:
        true_labels: Nhãn ground truth
        pred_labels: Nhãn dự đoán
        graph: igraph.Graph (optional, để tính modularity)
    
    Returns:
        dict chứa tất cả metrics
    """
    results = {
        'NMI': compute_nmi(true_labels, pred_labels),
        'ARI': compute_ari(true_labels, pred_labels),
        'Purity': compute_purity(true_labels, pred_labels),
        'FMI': compute_fmi(true_labels, pred_labels),
        'V-Measure': compute_v_measure(true_labels, pred_labels),
        'Homogeneity': homogeneity_score(true_labels, pred_labels),
        'Completeness': completeness_score(true_labels, pred_labels),
    }
    
    if graph is not None:
        results['Modularity'] = compute_modularity(graph, pred_labels)
    
    return results


def evaluate_all_algorithms(true_labels, clustering_results, graph=None):
    """
    Đánh giá tất cả các thuật toán
    
    Args:
        true_labels: Nhãn ground truth
        clustering_results: dict {algorithm_name: labels}
        graph: igraph.Graph
    
    Returns:
        dict: {algorithm_name: {metric: value}}
    """
    all_results = {}
    
    for algo_name, pred_labels in clustering_results.items():
        all_results[algo_name] = evaluate_clustering(true_labels, pred_labels, graph)
    
    return all_results


def print_evaluation_results(results, metrics=None):
    """
    In bảng kết quả đánh giá
    
    Args:
        results: dict {algorithm_name: {metric: value}}
        metrics: List metrics cần hiển thị (mặc định: NMI, ARI, Purity, Modularity)
    """
    if metrics is None:
        metrics = ['NMI', 'ARI', 'Purity', 'Modularity']
    
    # Header
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    header = f"{'Algorithm':<15}"
    for m in metrics:
        header += f"{m:<12}"
    print(header)
    print("-"*70)
    
    # Rows
    for algo, scores in results.items():
        row = f"{algo:<15}"
        for m in metrics:
            value = scores.get(m, 0)
            row += f"{value:<12.4f}"
        print(row)
    
    print("-"*70)
    
    # Best algorithm for each metric
    print("\nBest algorithm per metric:")
    for m in metrics:
        best_algo = max(results.keys(), key=lambda a: results[a].get(m, 0))
        best_value = results[best_algo].get(m, 0)
        print(f"  {m}: {best_algo} ({best_value:.4f})")


def get_cluster_statistics(labels, true_labels=None):
    """
    Tính thống kê về các cluster
    
    Args:
        labels: Cluster labels
        true_labels: Ground truth (optional)
    
    Returns:
        dict chứa thống kê
    """
    unique_clusters = np.unique(labels)
    cluster_sizes = [np.sum(labels == c) for c in unique_clusters]
    
    stats = {
        'n_clusters': len(unique_clusters),
        'avg_cluster_size': np.mean(cluster_sizes),
        'std_cluster_size': np.std(cluster_sizes),
        'min_cluster_size': np.min(cluster_sizes),
        'max_cluster_size': np.max(cluster_sizes),
        'cluster_sizes': dict(zip(unique_clusters, cluster_sizes)),
    }
    
    if true_labels is not None:
        stats['n_true_classes'] = len(np.unique(true_labels))
    
    return stats
