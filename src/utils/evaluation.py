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


def _is_multilabel_array(true_labels):
    """Kiểm tra true_labels có phải dạng multi-label theo từng mẫu hay không"""
    if len(true_labels) == 0:
        return False

    sample = true_labels[0]
    return isinstance(sample, (list, tuple, set, np.ndarray))


def _normalize_label_set(label_value):
    """Chuẩn hóa label của một mẫu về set"""
    if isinstance(label_value, np.ndarray):
        values = label_value.tolist()
    elif isinstance(label_value, (list, tuple, set)):
        values = list(label_value)
    else:
        values = [label_value]

    cleaned = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        cleaned.add(value)

    return cleaned


def _to_label_sets(true_labels):
    """Chuyển mảng true labels về list các set nhãn"""
    return [_normalize_label_set(label_value) for label_value in true_labels]


def _to_single_labels_from_sets(label_sets):
    """Tạo nhãn đại diện 1 giá trị từ multi-label để dùng cho metric chuẩn"""
    single_labels = []
    for label_set in label_sets:
        if not label_set:
            single_labels.append(-1)
            continue

        # Ổn định theo thứ tự tăng dần khi có nhiều label
        single_labels.append(sorted(label_set)[0])

    return np.array(single_labels)


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


def compute_relaxed_purity(true_label_sets, pred_labels):
    """
    Multi-label Purity (relaxed)

    Mỗi cluster được gán 1 nhãn đại diện tốt nhất.
    Một mẫu được tính đúng nếu nhãn đại diện nằm trong tập true labels của mẫu đó.
    """
    clusters = set(pred_labels)
    total_correct = 0

    for cluster_id in clusters:
        cluster_indices = np.where(pred_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        label_counter = Counter()
        for idx in cluster_indices:
            for label in true_label_sets[idx]:
                label_counter[label] += 1

        if not label_counter:
            continue

        best_label, _ = label_counter.most_common(1)[0]
        cluster_correct = sum(
            1 for idx in cluster_indices if best_label in true_label_sets[idx]
        )
        total_correct += cluster_correct

    return total_correct / len(true_label_sets)


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
        dict chứa 4 metrics: NMI, Purity, ARI, Modularity
    """
    pred_labels = np.asarray(pred_labels)

    if _is_multilabel_array(true_labels):
        true_label_sets = _to_label_sets(true_labels)
        true_labels_single = _to_single_labels_from_sets(true_label_sets)
        purity = compute_relaxed_purity(true_label_sets, pred_labels)
        nmi = compute_nmi(true_labels_single, pred_labels)
        ari = compute_ari(true_labels_single, pred_labels)
    else:
        purity = compute_purity(true_labels, pred_labels)
        nmi = compute_nmi(true_labels, pred_labels)
        ari = compute_ari(true_labels, pred_labels)

    modularity = compute_modularity(graph, pred_labels) if graph is not None else np.nan

    return {
        'NMI': nmi,
        'Purity': purity,
        'ARI': ari,
        'Modularity': modularity,
    }


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
        metrics: List metrics cần hiển thị (mặc định: NMI, Purity, ARI, Modularity)
    """
    if metrics is None:
        metrics = ['NMI', 'Purity', 'ARI', 'Modularity']
    
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
        if _is_multilabel_array(true_labels):
            true_label_sets = _to_label_sets(true_labels)
            all_true_labels = set()
            for label_set in true_label_sets:
                all_true_labels.update(label_set)
            stats['n_true_classes'] = len(all_true_labels)
        else:
            stats['n_true_classes'] = len(np.unique(true_labels))
    
    return stats
