"""
Cấu hình và tham số cho dự án
"""
import torch

class Config:
    # Đường dẫn dữ liệu
    DATA_PATH = 'data'
    RESULTS_PATH = 'results'
    
    # Feature Extraction
    MODEL_NAME = 'efficientnet_v2_l'  # efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s, resnet50
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Graph Construction
    K_NEIGHBORS = 10  # Số láng giềng cho k-NN graph
    SIMILARITY_METRIC = 'cosine'  # 'cosine' hoặc 'euclidean'
    
    # Clustering
    LEIDEN_RESOLUTION = 1.0
    LOUVAIN_RESOLUTION = 1.0
    
    # Misc
    SAMPLE_SIZE = None  # None = toàn bộ, số nguyên để test nhanh
    RANDOM_STATE = 42

config = Config()
