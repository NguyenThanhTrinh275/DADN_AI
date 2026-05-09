import torch

class Config:
    DATA_PATH = 'data'
    RESULTS_PATH = 'results'
    
    MODEL_NAME = 'dinov2_vitl14'  # dinov2_vits14, dinov2_vitb14, dinov2_vitl14, efficientnet_v2_l
    BATCH_SIZE = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    K_NEIGHBORS = 30
    SIMILARITY_METRIC = 'cosine'
    MUTUAL_KNN = True
    SIM_POWER = 4.0

    USE_TTA = True
    TTA_FLIPS = (False, True)
    TTA_SCALES = (518,)

    USE_PCA_WHITEN = True
    PCA_DIM = 256

    LEIDEN_RESOLUTION = 40.0
    LOUVAIN_RESOLUTION = 46.0

    SAMPLE_SIZE = None
    RANDOM_STATE = 42

config = Config()
