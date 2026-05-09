"""
Hậu xử lý feature vectors: PCA whitening + L2-renormalize.

PCA whitening loại bỏ correlation giữa các dim và scale phương sai về 1,
giúp cosine similarity / k-NN graph ít bị nhiễu bởi các dim phương sai thấp.
Đặc biệt hữu ích cho features từ DINOv2 / ViT.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def pca_whiten(features: np.ndarray, n_components: int = 256, random_state: int = 42) -> np.ndarray:
    """
    Áp PCA whitening lên features rồi L2-normalize.

    Args:
        features: Ma trận (n_samples, feature_dim) đã L2-normalize
        n_components: Số chiều giữ lại sau PCA (mặc định 256)
        random_state: Seed cho reproducibility

    Returns:
        numpy array (n_samples, n_components) đã L2-normalize
    """
    n_components = min(n_components, features.shape[0], features.shape[1])
    print(f"  PCA whitening: {features.shape[1]} → {n_components} dims")

    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
    reduced = pca.fit_transform(features)

    explained = pca.explained_variance_ratio_.sum()
    print(f"  ✓ Giữ lại {explained*100:.1f}% phương sai")

    return normalize(reduced, norm='l2').astype(np.float32)
