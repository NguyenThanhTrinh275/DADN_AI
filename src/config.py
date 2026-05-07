import torch

class Config:
    DATA_PATH = 'data'
    RESULTS_PATH = 'results'
    
    MODEL_NAME = 'dinov2_vits14'  # efficientnet_v2_l, dinov2_vits14
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    K_NEIGHBORS = 3  
    SIMILARITY_METRIC = 'cosine'
    
    LEIDEN_RESOLUTION = 130.0   
    LOUVAIN_RESOLUTION = 130.0 
    
    SAMPLE_SIZE = None
    RANDOM_STATE = 42

config = Config()
