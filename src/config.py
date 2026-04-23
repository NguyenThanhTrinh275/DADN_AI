import torch

class Config:
    DATA_PATH = 'data'
    RESULTS_PATH = 'results'
    
    # Feature Extraction
    MODEL_NAME = 'dinov2_vits14'  # efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s, efficientnet_b7, resnet50, dinov2_vits14
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Graph Construction
    K_NEIGHBORS = 3  # Số láng giềng cho k-NN graph
    SIMILARITY_METRIC = 'cosine'  # 'cosine' hoặc 'euclidean'
    
    # Clustering
    LEIDEN_RESOLUTION = 130.0   
    LOUVAIN_RESOLUTION = 130.0 
    
    # Misc
    SAMPLE_SIZE = None
    RANDOM_STATE = 42

config = Config()
