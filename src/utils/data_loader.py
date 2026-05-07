"""
Module tải và tiền xử lý dữ liệu
"""
import argparse
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
    
    # Tìm tất cả các file có đuôi .parquet trong thư mục cấu hình.
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


def _normalize_label_values(label_value):
    """Chuẩn hóa label về list giá trị"""
    if isinstance(label_value, np.ndarray):
        values = label_value.tolist()
    elif isinstance(label_value, (list, tuple, set)):
        values = list(label_value)
    else:
        values = [label_value]

    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        cleaned.append(value)

    return cleaned


def extract_images_and_labels(df, return_multilabel=False):
    """
    Trích xuất danh sách ảnh và nhãn từ DataFrame
    
    Args:
        df: DataFrame từ load_data()
        return_multilabel: True để trả thêm danh sách labels cho từng ảnh
    
    Returns:
        images: List các bytes ảnh
        labels: numpy array nhãn chính (phần tử đầu tiên)
        label_sets: List[set] các nhãn đầy đủ cho từng ảnh (nếu return_multilabel=True)
    """
    images = df['image'].apply(lambda x: x['bytes']).tolist()

    normalized_labels = df['label'].apply(_normalize_label_values)

    # Dùng nhãn nhỏ nhất (sorted) để primary label luôn deterministic
    labels = normalized_labels.apply(lambda values: sorted(values)[0]).values

    label_sets = normalized_labels.apply(set).tolist()
    multi_label_count = sum(1 for label_set in label_sets if len(label_set) > 1)
    
    print(f"Số lượng ảnh: {len(images)}")
    print(f"Số lượng classes: {len(np.unique(labels))}")

    if multi_label_count > 0:
        print(f"Số ảnh có nhiều hơn 1 label: {multi_label_count}")

    if return_multilabel:
        return images, labels, label_sets

    return images, labels


def get_english_labels(df):
    """Lấy tên tiếng Anh của các nhãn"""
    return df['english_label'].apply(
        lambda x: x[0] if hasattr(x, '__len__') else x
    ).values


def _build_arg_parser():
    """Tạo parser cho chế độ chạy trực tiếp module data_loader"""
    parser = argparse.ArgumentParser(
        description='Test riêng module data_loader'
    )
    parser.add_argument(
        '--path',
        type=str,
        default=None,
        help='Đường dẫn thư mục chứa parquet (mặc định: config.DATA_PATH)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=10,
        help='Số lượng mẫu để test nhanh (mặc định: 10)'
    )
    parser.add_argument(
        '--multilabel',
        action='store_true',
        help='Trả về và in thống kê true label sets'
    )
    return parser


def _run_self_test(path=None, sample_size=10, return_multilabel=False):
    """Chạy smoke test cho module data_loader"""
    print("=" * 60)
    print("DATA LOADER SELF-TEST")
    print("=" * 60)
    print(f"Input path: {path or config.DATA_PATH}")
    print(f"Sample size: {sample_size}")
    print(f"Return multilabel: {return_multilabel}")

    df = load_data(path=path, sample_size=sample_size)

    if return_multilabel:
        images, labels, label_sets = extract_images_and_labels(df, return_multilabel=True)
    else:
        images, labels = extract_images_and_labels(df, return_multilabel=False)
        label_sets = None

    print("\nBasic checks:")
    print(f"  - n_rows(df): {len(df)}")
    print(f"  - n_images:   {len(images)}")
    print(f"  - n_labels:   {len(labels)}")
    print(f"  - image/label aligned: {len(images) == len(labels)}")

    if images:
        print(f"  - first image bytes length: {len(images[0])}")

    if len(labels) > 0:
        print(f"  - unique labels (primary): {len(np.unique(labels))}")
        preview_count = min(5, len(labels))
        print(f"  - first {preview_count} primary labels: {labels[:preview_count].tolist()}")

    if label_sets is not None:
        preview_count = min(5, len(label_sets))
        print(f"  - first {preview_count} label sets: {[sorted(list(s)) for s in label_sets[:preview_count]]}")

    print("\nSelf-test completed.")


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    _run_self_test(
        path=args.path,
        sample_size=args.sample,
        return_multilabel=args.multilabel,
    )