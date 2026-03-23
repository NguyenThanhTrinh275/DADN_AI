"""
Module visualization cho kết quả phân cụm
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.config import config
from src.utils.data_loader import bytes_to_image


def ensure_results_dir():
    """Tạo thư mục results nếu chưa tồn tại"""
    os.makedirs(config.RESULTS_PATH, exist_ok=True)


def plot_metrics_comparison(results, save_path=None):
    """
    Vẽ biểu đồ so sánh các thuật toán theo từng metric
    
    Args:
        results: dict {algorithm_name: {metric: value}}
        save_path: Đường dẫn lưu file (None = không lưu)
    """
    ensure_results_dir()
    
    if save_path is None:
        save_path = os.path.join(config.RESULTS_PATH, 'comparison.png')
    
    algorithms = list(results.keys())
    metrics = ['NMI', 'Purity', 'ARI', 'Modularity']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [results[algo].get(metric, 0) for algo in algorithms]
        bars = axes[idx].bar(algorithms, values, color=colors[idx], edgecolor='black')

        axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel(metric, fontsize=12)
        axes[idx].set_ylim(0, 1.1 if metric != 'ARI' else max(values) + 0.1)

        # Hiển thị giá trị trên bar
        for bar, v in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{v:.3f}',
                ha='center',
                va='bottom',
                fontsize=11
            )

        axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved comparison plot to {save_path}")


def plot_radar_chart(results, save_path=None):
    """
    Vẽ radar chart so sánh các thuật toán
    
    Args:
        results: dict {algorithm_name: {metric: value}}
        save_path: Đường dẫn lưu file
    """
    ensure_results_dir()
    
    if save_path is None:
        save_path = os.path.join(config.RESULTS_PATH, 'radar_chart.png')
    
    algorithms = list(results.keys())
    metrics = ['NMI', 'Purity', 'ARI', 'Modularity']

    # Số lượng metrics
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Đóng vòng

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    for idx, algo in enumerate(algorithms):
        values = [max(0, results[algo].get(m, 0)) for m in metrics]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    ax.set_title('Algorithm Comparison (Radar Chart)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved radar chart to {save_path}")


def visualize_clusters(images, labels, n_samples=5, n_clusters=10, save_path=None):
    """
    Hiển thị mẫu ảnh từ mỗi cluster
    
    Args:
        images: List các bytes ảnh
        labels: Cluster labels
        n_samples: Số ảnh mỗi cluster
        n_clusters: Số cluster tối đa hiển thị
        save_path: Đường dẫn lưu file
    """
    ensure_results_dir()
    
    if save_path is None:
        save_path = os.path.join(config.RESULTS_PATH, 'clusters.png')
    
    unique_labels = np.unique(labels)
    n_clusters = min(len(unique_labels), n_clusters)
    
    # Sắp xếp theo kích thước cluster (lớn nhất trước)
    cluster_sizes = [(c, np.sum(labels == c)) for c in unique_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    top_clusters = [c[0] for c in cluster_sizes[:n_clusters]]
    
    fig, axes = plt.subplots(n_clusters, n_samples, figsize=(3*n_samples, 3*n_clusters))
    
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    for i, cluster_id in enumerate(top_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_size = len(cluster_indices)
        
        sample_indices = np.random.choice(
            cluster_indices,
            size=min(n_samples, cluster_size),
            replace=False
        )
        
        for j in range(n_samples):
            if j < len(sample_indices):
                idx = sample_indices[j]
                img = bytes_to_image(images[idx])
                axes[i, j].imshow(img)
            axes[i, j].axis('off')
        
        # Title cho hàng đầu tiên của mỗi cluster
        axes[i, 0].set_title(f'Cluster {cluster_id}\n(n={cluster_size})', 
                            fontsize=10, loc='left')
    
    plt.suptitle(f'Sample Images from Top {n_clusters} Clusters', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved cluster visualization to {save_path}")


def plot_cluster_distribution(clustering_results, save_path=None):
    """
    Vẽ biểu đồ phân bố số lượng cluster của các thuật toán
    
    Args:
        clustering_results: dict {algorithm_name: labels}
        save_path: Đường dẫn lưu file
    """
    ensure_results_dir()
    
    if save_path is None:
        save_path = os.path.join(config.RESULTS_PATH, 'cluster_distribution.png')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for idx, (algo, labels) in enumerate(clustering_results.items()):
        unique, counts = np.unique(labels, return_counts=True)
        
        # Sort by count (descending)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_indices][:20]  # Top 20 clusters
        sorted_labels = unique[sorted_indices][:20]
        
        axes[idx].bar(range(len(sorted_counts)), sorted_counts, color=colors[idx])
        axes[idx].set_xlabel('Cluster (sorted by size)')
        axes[idx].set_ylabel('Number of images')
        axes[idx].set_title(f'{algo} - {len(unique)} clusters', fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Cluster Size Distribution (Top 20)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved distribution plot to {save_path}")
