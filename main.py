"""
Đề tài 3: Community Structure Identification cho bài toán phân cụm hình ảnh
Dataset: ImageNet-Hard
Thuật toán: Infomap, Leiden, Louvain, Label Propagation (LPA)

Pipeline:
1. Load Data (Parquet) → Ảnh bytes
2. Feature Extraction (EfficientNet) → Vector cao chiều  
3. Build Graph (k-NN + Cosine Similarity) → Đồ thị cộng đồng
4. Community Detection (4 thuật toán) → Cluster labels
5. Evaluate & Compare với Ground Truth → NMI, ARI, Purity

Cách chạy:
    python main.py
    
Hoặc với sample size nhỏ để test:
    python main.py --sample 500
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

# Import các module từ src package
from src.config import config
from src.utils.data_loader import load_data, extract_images_and_labels
from src.models.feature_extractor import FeatureExtractor
from src.models.graph_builder import build_knn_graph, get_graph_statistics
from src.models.clustering import run_all_algorithms
from src.utils.evaluation import evaluate_all_algorithms, print_evaluation_results
from src.utils.visualization import (
    plot_metrics_comparison,
    plot_radar_chart,
    visualize_clusters,
    plot_cluster_distribution
)


def main(sample_size=None):
    """
    Pipeline chính cho Community Structure Identification
    
    Args:
        sample_size: Số lượng mẫu (None = toàn bộ dataset)
    """
    print("="*70)
    print("  COMMUNITY STRUCTURE IDENTIFICATION FOR IMAGE CLUSTERING")
    print("  Dataset: ImageNet-Hard")
    print("  Algorithms: Infomap, Leiden, Louvain, LPA")
    print("="*70)
    
    # =========================================================
    # STEP 1: TẢI DỮ LIỆU
    # =========================================================
    print("\n" + "="*50)
    print("[STEP 1/5] Loading data from parquet files...")
    print("="*50)
    
    df = load_data(sample_size=sample_size)
    images, true_labels = extract_images_and_labels(df)
    
    # =========================================================
    # STEP 2: TRÍCH XUẤT ĐẶC TRƯNG
    # =========================================================
    print("\n" + "="*50)
    print("[STEP 2/5] Extracting features using EfficientNet...")
    print("="*50)
    
    extractor = FeatureExtractor()
    features = extractor.extract_features(images)
    print(f"Feature matrix shape: {features.shape}")
    
    # =========================================================
    # STEP 3: XÂY DỰNG ĐỒ THỊ
    # =========================================================
    print("\n" + "="*50)
    print("[STEP 3/5] Building k-NN similarity graph...")
    print("="*50)
    
    graph = build_knn_graph(features)
    
    # In thống kê đồ thị
    stats = get_graph_statistics(graph)
    print(f"\nGraph Statistics:")
    print(f"  - Nodes: {stats['n_nodes']}")
    print(f"  - Edges: {stats['n_edges']}")
    print(f"  - Average degree: {stats['avg_degree']:.2f}")
    print(f"  - Is connected: {stats['is_connected']}")
    
    # =========================================================
    # STEP 4: PHÂN CỤM CỘNG ĐỒNG
    # =========================================================
    print("\n" + "="*50)
    print("[STEP 4/5] Running community detection algorithms...")
    print("="*50)
    
    clustering_results = run_all_algorithms(graph)
    
    # =========================================================
    # STEP 5: ĐÁNH GIÁ VÀ SO SÁNH
    # =========================================================
    print("\n" + "="*50)
    print("[STEP 5/5] Evaluating and comparing results...")
    print("="*50)
    
    evaluation_results = evaluate_all_algorithms(
        true_labels, 
        clustering_results, 
        graph
    )
    
    # In kết quả
    print_evaluation_results(evaluation_results)
    
    # =========================================================
    # VISUALIZATION
    # =========================================================
    print("\n" + "="*50)
    print("Generating visualizations...")
    print("="*50)
    
    # So sánh metrics
    plot_metrics_comparison(evaluation_results)
    
    # Radar chart
    plot_radar_chart(evaluation_results)
    
    # Phân bố cluster
    plot_cluster_distribution(clustering_results)
    
    # Visualize clusters của thuật toán tốt nhất
    best_algo = max(evaluation_results.keys(), 
                   key=lambda x: evaluation_results[x]['NMI'])
    print(f"\nVisualizing clusters from best algorithm: {best_algo}")
    visualize_clusters(images, clustering_results[best_algo])
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETED SUCCESSFULLY!")
    print("  Results saved to 'results/' folder")
    print("="*70)
    
    return {
        'features': features,
        'graph': graph,
        'clustering_results': clustering_results,
        'evaluation_results': evaluation_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Community Structure Identification for Image Clustering'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=None,
        help='Number of samples to use (default: all)'
    )
    
    args = parser.parse_args()
    
    results = main(sample_size=args.sample)
