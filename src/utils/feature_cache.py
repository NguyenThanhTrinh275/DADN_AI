"""
Module lưu/tải feature cache để bỏ qua bước extract lại từ ảnh
"""
import json
import os
import numpy as np


def save_feature_cache(cache_path, features, true_label_sets, metadata=None):
    """
    Lưu features và nhãn multi-label xuống file .npz

    Args:
        cache_path: Đường dẫn file cache (.npz)
        features: numpy array shape (n_samples, feature_dim)
        true_label_sets: list[set] nhãn ground-truth cho từng ảnh
        metadata: dict metadata bổ sung
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    label_values = np.array([sorted(list(label_set)) for label_set in true_label_sets], dtype=object)
    payload = {
        'features': features,
        'true_label_sets': label_values,
        'metadata': json.dumps(metadata or {}, ensure_ascii=False),
    }
    np.savez_compressed(cache_path, **payload)


def load_feature_cache(cache_path):
    """
    Tải features và nhãn từ file cache

    Args:
        cache_path: Đường dẫn file cache (.npz)

    Returns:
        dict gồm: features, true_label_sets, metadata
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Không tìm thấy feature cache: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)

    features = data['features']
    raw_label_sets = data['true_label_sets'].tolist()
    true_label_sets = [set(labels if isinstance(labels, list) else [labels]) for labels in raw_label_sets]

    metadata_raw = data['metadata'].item() if 'metadata' in data.files else '{}'
    try:
        metadata = json.loads(metadata_raw)
    except (json.JSONDecodeError, TypeError):
        metadata = {}

    if len(features) != len(true_label_sets):
        raise ValueError(
            f"Feature cache không hợp lệ: n_features={len(features)} nhưng n_labels={len(true_label_sets)}"
        )

    return {
        'features': features,
        'true_label_sets': true_label_sets,
        'metadata': metadata,
    }
