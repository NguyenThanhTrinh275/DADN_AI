import argparse
import os
import warnings
warnings.filterwarnings('ignore')

from src.config import config
from src.utils.data_loader import load_data, extract_images_and_labels
from src.models.feature_extractor import FeatureExtractor
from src.models.graph_builder import build_knn_graph, get_graph_statistics
from src.models.clustering import run_all_algorithms
from src.utils.feature_cache import save_feature_cache, load_feature_cache
from src.utils.feature_postprocess import pca_whiten
from src.utils.evaluation import evaluate_all_algorithms, print_evaluation_results, get_cluster_statistics
from src.utils.visualization import (
    plot_metrics_comparison,
    plot_radar_chart,
    visualize_clusters,
    plot_cluster_distribution
)
from src.utils.result_logger import ResultLogger


def default_cache_path(sample_size):
    sample_tag = f"sample{sample_size}" if sample_size is not None else "full"
    return os.path.join(
        config.RESULTS_PATH,
        "cache",
        f"features_{config.MODEL_NAME}_{sample_tag}.npz"
    )


def main(
    sample_size=None,
    device='auto',
    use_feature_cache=False,
    save_feature_cache_enabled=False,
    feature_cache_path=None,
    log_results=True,
    log_path='results/results.csv'
):
    
    """
    Pipeline chính cho Community Structure Identification
    Args:
        sample_size: Số lượng mẫu (None = toàn bộ dataset)
        device: Thiết bị chạy feature extraction ('auto', 'cpu', 'gpu')
        use_feature_cache: Dùng feature/label đã lưu sẵn
        save_feature_cache_enabled: Lưu feature/label sau khi extract
        feature_cache_path: Đường dẫn file cache .npz
        log_results: Ghi kết quả vào CSV (default: True)
        log_path: Đường dẫn file CSV để lưu kết quả
    """
    if feature_cache_path is None:
        feature_cache_path = default_cache_path(sample_size)

    print("="*70)
    print("  COMMUNITY STRUCTURE IDENTIFICATION FOR IMAGE CLUSTERING")
    print("  Dataset: ImageNet-Hard")
    print("  Algorithms: Infomap, Leiden, Louvain, LPA")
    print("="*70)

    images = None
    true_label_sets = None

    if use_feature_cache:
        print("\n" + "="*50)
        print("[STEP 1/5] Loading features and labels from cache...")
        print("="*50)

        cache_data = load_feature_cache(feature_cache_path)
        features = cache_data['features']
        true_label_sets = cache_data['true_label_sets']
        cache_metadata = cache_data.get('metadata', {})

        print(f"Loaded cache: {feature_cache_path}")
        print(f"Feature matrix shape: {features.shape}")
        if cache_metadata:
            print(f"Cache metadata: {cache_metadata}")

        print("\n" + "="*50)
        print("[STEP 2/5] Skipping feature extraction (using cached data)...")
        print("="*50)
    else:
        # =========================================================
        # STEP 1: TẢI DỮ LIỆU
        # =========================================================
        print("\n" + "="*50)
        print("[STEP 1/5] Loading data from parquet files...")
        print("="*50)

        df = load_data(sample_size=sample_size)
        images, _, true_label_sets = extract_images_and_labels(
            df,
            return_multilabel=True
        )

        # =========================================================
        # STEP 2: TRÍCH XUẤT ĐẶC TRƯNG
        # =========================================================
        print("\n" + "="*50)
        print("[STEP 2/5] Extracting features...")
        print("="*50)

        extractor = FeatureExtractor(device=device)
        if config.USE_TTA:
            print(f"  Sử dụng TTA: scales={config.TTA_SCALES}, flips={config.TTA_FLIPS}")
            features = extractor.extract_features_tta(
                images,
                scales=config.TTA_SCALES,
                flips=config.TTA_FLIPS,
            )
        else:
            features = extractor.extract_features(images)
        print(f"Feature matrix shape: {features.shape}")

        # Cache lưu features RAW (chưa PCA) để có thể đổi PCA_DIM mà không cần re-extract
        if save_feature_cache_enabled:
            cache_metadata = {
                'model_name': extractor.model_name,
                'device': extractor.device,
                'sample_size': sample_size,
                'feature_dim': int(features.shape[1]) if features.ndim == 2 else None,
                'n_samples': int(features.shape[0]),
                'tta_scales': list(config.TTA_SCALES) if config.USE_TTA else None,
                'tta_flips':  list(config.TTA_FLIPS)  if config.USE_TTA else None,
            }
            save_feature_cache(feature_cache_path, features, true_label_sets, cache_metadata)
            print(f"Saved feature cache to: {feature_cache_path}")

    # PCA whitening áp dụng cho cả cached lẫn freshly-extracted features
    if config.USE_PCA_WHITEN:
        print("\n  Áp dụng PCA whitening...")
        features = pca_whiten(features, n_components=config.PCA_DIM, random_state=config.RANDOM_STATE)
        print(f"  Sau PCA whitening: {features.shape}")

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
        true_label_sets,
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
    
    # Visualize clusters của thuật toán tốt nhất theo NMI
    best_algo = max(
        evaluation_results.keys(),
        key=lambda x: evaluation_results[x]['NMI']
    )
    print(f"\nVisualizing clusters from best algorithm: {best_algo}")

    viz_images = images
    cluster_pick_mode = 'largest'

    if viz_images is None and use_feature_cache:
        print(
            "Đang tải lại ảnh từ parquet để vẽ clusters.png "
            "(cùng sample_size và DATA_PATH với lúc tạo cache)..."
        )
        try:
            df_viz = load_data(sample_size=sample_size)
            viz_images, _, _ = extract_images_and_labels(df_viz, return_multilabel=True)
            if len(viz_images) != len(features):
                print(
                    f"[WARN] len(images)={len(viz_images)} != len(features)={len(features)}. "
                    "Bỏ qua clusters.png — kiểm tra --sample và DATA_PATH khớp file cache."
                )
                viz_images = None
            else:
                cluster_pick_mode = 'random'
        except FileNotFoundError as exc:
            print(f"[WARN] Không đọc được parquet để vẽ clusters.png: {exc}")
            viz_images = None

    if viz_images is not None:
        visualize_clusters(
            viz_images,
            clustering_results[best_algo],
            cluster_pick=cluster_pick_mode,
        )
    else:
        print(
            "Skip cluster image visualization "
            "(không có bytes ảnh hoặc parquet không khớp cache)."
        )
    
    # =========================================================
    # LOGGING KẾT QUẢ VÀO CSV
    # =========================================================
    if log_results:
        print("\n" + "="*50)
        print("Logging results to CSV...")
        print("="*50)
        
        logger = ResultLogger(log_path)
        
        # Chuẩn bị dữ liệu đầu vào cho logger
        # Thêm thông tin cluster statistics vào evaluation_results
        clustering_results_with_stats = {}
        for algo_name, pred_labels in clustering_results.items():
            stats = get_cluster_statistics(pred_labels)
            # Merge evaluation metrics với statistics
            clustering_results_with_stats[algo_name] = {
                **evaluation_results[algo_name],
                'n_clusters': stats['n_clusters'],
                'avg_cluster_size': stats['avg_cluster_size'],
            }
        
        # Ghi kết quả vào CSV
        logger.log_run(
            sample_size=sample_size,
            model_name=config.MODEL_NAME,
            k_neighbors=config.K_NEIGHBORS,
            leiden_resolution=config.LEIDEN_RESOLUTION,
            louvain_resolution=config.LOUVAIN_RESOLUTION,
            clustering_results=clustering_results_with_stats,
            notes=f'Best algorithm: {best_algo}'
        )
        logger.print_summary()
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETED SUCCESSFULLY!")
    print("  Results saved to 'results/' folder")
    if log_results:
        print(f"  Metrics logged to '{log_path}'")
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
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'gpu', 'cuda'],
        help="Device for feature extraction: auto/cpu/gpu (default: auto)"
    )
    parser.add_argument(
        '--use-feature-cache',
        action='store_true',
        help='Load precomputed features + labels from cache and skip extraction'
    )
    parser.add_argument(
        '--save-feature-cache',   
        action='store_true',
        help='Save extracted features + labels to cache for later runs'
    )
    parser.add_argument(
        '--feature-cache-path',
        type=str,
        default=None,
        help='Path to feature cache file (.npz). Default: results/cache/features_<model>_<sample|full>.npz'
    )
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Disable logging results to CSV'
    )
    parser.add_argument(
        '--log-path',
        type=str,
        default='results/results.csv',
        help='Path to save results CSV (default: results/results.csv)'
    )
    
    args = parser.parse_args()
    
    results = main(
        sample_size=args.sample,
        device=args.device,
        use_feature_cache=args.use_feature_cache,
        save_feature_cache_enabled=args.save_feature_cache,
        feature_cache_path=args.feature_cache_path,
        log_results=not args.no_log,
        log_path=args.log_path
    )
