"""
Module logging kết quả các lần chạy vào file CSV

Mỗi lần chạy thành công sẽ tự động thêm một dòng mới vào results.csv
với các thông tin: sample_size, model, parameters, metrics, v.v.
"""

import csv
import os
from datetime import datetime
import numpy as np
from pathlib import Path


class ResultLogger:
    """
    Ghi kết quả chạy vào file CSV một cách tự động
    
    Ví dụ sử dụng:
        logger = ResultLogger('results/results.csv')
        logger.log_run(
            sample_size=1000,
            model_name='efficientnet_b7',
            k_neighbors=10,
            leiden_resolution=30.0,
            louvain_resolution=30.0,
            clustering_results={'Leiden': {...}, 'Louvain': {...}, ...},
            graph_stats={'n_nodes': 1000}
        )
    """
    
    def __init__(self, csv_path='results/results.csv'):
        """
        Khởi tạo logger
        
        Args:
            csv_path: Đường dẫn file CSV để lưu kết quả
        """
        self.csv_path = csv_path
        self.csv_path_obj = Path(csv_path)
        
        # Tạo thư mục nếu chưa tồn tại
        self.csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Kiểm tra file CSV đã tồn tại hay chưa
        self.file_exists = os.path.exists(csv_path)
        
        # Định nghĩa các cột CSV
        self.fieldnames = [
            'timestamp',           # Thời gian chạy (YYYY-MM-DD HH:MM:SS)
            'sample_size',         # Số lượng sample được sử dụng
            'model_name',          # Tên model feature extraction (e.g., dinov2_vits14)
            'k_neighbors',         # Số láng giềng trong k-NN graph
            'leiden_resolution',   # Resolution parameter cho Leiden
            'louvain_resolution',  # Resolution parameter cho Louvain
            'algorithm',           # Tên thuật toán phân cụm (Infomap, Leiden, Louvain, LPA)
            'n_clusters',          # Số cluster được phát hiện
            'avg_cluster_size',    # Kích thước trung bình của cluster
            'nmi',                 # Normalized Mutual Information
            'purity',              # Purity Score
            'ari',                 # Adjusted Rand Index
            'modularity',          # Modularity Score
            'notes',               # Ghi chú thêm (tùy chọn)
        ]
        
        # Nếu file chưa tồn tại, tạo header
        if not self.file_exists:
            self._create_header()
    
    def _create_header(self):
        """Tạo header row cho CSV file"""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        print(f"✓ Tạo file CSV mới: {self.csv_path}")
    
    def log_run(self, sample_size, model_name, k_neighbors, 
                leiden_resolution, louvain_resolution,
                clustering_results, graph_stats=None, notes=''):
        """
        Ghi một lần chạy vào CSV
        
        Args:
            sample_size: Số lượng sample (None nếu là full dataset)
            model_name: Tên model (e.g., 'dinov2_vits14')
            k_neighbors: Số k trong k-NN
            leiden_resolution: Resolution cho Leiden
            louvain_resolution: Resolution cho Louvain
            clustering_results: dict {algo_name: eval_dict}
                Ví dụ: {
                    'Leiden': {'NMI': 0.5, 'Purity': 0.7, 'ARI': 0.6, 'Modularity': 0.8},
                    'Louvain': {'NMI': 0.48, 'Purity': 0.68, 'ARI': 0.55, 'Modularity': 0.75},
                    ...
                }
            graph_stats: dict thống kê graph (optional)
                Có thể chứa: n_nodes, n_edges, avg_cluster_size, v.v.
            notes: Ghi chú thêm về lần chạy này
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Nếu sample_size là None, viết là 'full'
        sample_size_str = str(sample_size) if sample_size is not None else 'full'
        
        # Ghi kết quả của từng thuật toán
        rows_to_write = []
        
        for algo_name, metrics in clustering_results.items():
            row = {
                'timestamp': timestamp,
                'sample_size': sample_size_str,
                'model_name': model_name,
                'k_neighbors': k_neighbors,
                'leiden_resolution': leiden_resolution,
                'louvain_resolution': louvain_resolution,
                'algorithm': algo_name,
                'nmi': self._format_value(metrics.get('NMI')),
                'purity': self._format_value(metrics.get('Purity')),
                'ari': self._format_value(metrics.get('ARI')),
                'modularity': self._format_value(metrics.get('Modularity')),
                'n_clusters': metrics.get('n_clusters', ''),
                'avg_cluster_size': self._format_value(metrics.get('avg_cluster_size')),
                'notes': notes,
            }
            rows_to_write.append(row)
        
        # Ghi vào CSV
        self._append_rows(rows_to_write)
    
    def _format_value(self, value):
        """
        Format giá trị để ghi vào CSV
        
        Args:
            value: Giá trị cần format
        
        Returns:
            String đã format hoặc chuỗi rỗng nếu None/NaN
        """
        if value is None:
            return ''
        
        if isinstance(value, float):
            if np.isnan(value):
                return ''
            return f'{value:.6f}'
        
        return str(value)
    
    def _append_rows(self, rows):
        """
        Thêm các dòng vào CSV
        
        Args:
            rows: List của dict chứa dữ liệu mỗi dòng
        """
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(rows)
        
        print(f"✓ Ghi {len(rows)} dòng vào {self.csv_path}")
    
    def print_summary(self):
        """In tóm tắt số dòng đã lưu"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                n_records = len(lines) - 1  # Trừ header
                print(f"\n📊 Tổng số lần chạy đã lưu: {n_records}")
        except FileNotFoundError:
            print(f"File {self.csv_path} không tồn tại")


def get_cluster_statistics_for_logging(pred_labels):
    """
    Tính thống kê cluster để dùng trong logging
    
    Args:
        pred_labels: Cluster labels từ một thuật toán
    
    Returns:
        dict chứa n_clusters và avg_cluster_size
    """
    unique_clusters = np.unique(pred_labels)
    cluster_sizes = [np.sum(pred_labels == c) for c in unique_clusters]
    
    return {
        'n_clusters': len(unique_clusters),
        'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
    }


def prepare_results_for_logging(clustering_results, graph_info=None):
    """
    Chuẩn bị dữ liệu từ clustering_results để sử dụng với ResultLogger
    
    Args:
        clustering_results: dict {algo_name: pred_labels}
        graph_info: dict thông tin graph (optional)
    
    Returns:
        dict {algo_name: {metrics dict}}
    
    Ví dụ:
        clustering_results = {
            'Leiden': array([0, 0, 1, 1, 2, ...]),
            'Louvain': array([0, 1, 0, 1, 2, ...]),
            ...
        }
        
        # Sau evaluation
        eval_results = {
            'Leiden': {'NMI': 0.5, 'Purity': 0.7, 'ARI': 0.6, 'Modularity': 0.8},
            'Louvain': {'NMI': 0.48, ...},
            ...
        }
        
        # Chuẩn bị cho logging
        logging_data = prepare_results_for_logging(clustering_results, eval_results)
    """
    prepared = {}
    
    for algo_name, pred_labels in clustering_results.items():
        stats = get_cluster_statistics_for_logging(pred_labels)
        
        # Lấy evaluation metrics nếu có
        metrics = {}
        if graph_info is not None and algo_name in graph_info:
            metrics = graph_info[algo_name]
        
        # Merge statistics với metrics
        prepared[algo_name] = {**metrics, **stats}
    
    return prepared
