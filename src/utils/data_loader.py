"""
Module tải và tiền xử lý dữ liệu
"""
import pandas as pd
import numpy as np
import glob
import os
from PIL import Image
import io

from src.config import config


def load_data(path=None, sample_size=None):
    """
    Tải dữ liệu từ các file parquet
    
    Args:
        path: Đường dẫn đến thư mục chứa data
        sample_size: Số lượng mẫu (None = toàn bộ)
    
    Returns:
        DataFrame chứa dữ liệu
    """
    if path is None:
        path = config.DATA_PATH
    
    all_files = glob.glob(os.path.join(path, "*.parquet"))
    
    if not all_files:
        raise FileNotFoundError(f"Không tìm thấy file parquet trong {path}")
    
    df = pd.concat((pd.read_parquet(f) for f in all_files), ignore_index=True)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=config.RANDOM_STATE)
    
    print(f"Tổng số lượng ảnh: {len(df)}")
    print(f"Các cột: {df.columns.tolist()}")
    return df


def bytes_to_image(image_bytes):
    """Chuyển bytes thành PIL Image"""
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')


def extract_images_and_labels(df):
    """
    Trích xuất danh sách ảnh và nhãn từ DataFrame
    
    Args:
        df: DataFrame từ load_data()
    
    Returns:
        images: List các bytes ảnh
        labels: numpy array các nhãn
    """
    images = df['image'].apply(lambda x: x['bytes']).tolist()
    
    # Label có thể là numpy array, lấy phần tử đầu tiên
    labels = df['label'].apply(
        lambda x: x[0] if hasattr(x, '__len__') else x
    ).values
    
    print(f"Số lượng ảnh: {len(images)}")
    print(f"Số lượng classes: {len(np.unique(labels))}")
    
    return images, labels


def get_english_labels(df):
    """Lấy tên tiếng Anh của các nhãn"""
    return df['english_label'].apply(
        lambda x: x[0] if hasattr(x, '__len__') else x
    ).values
