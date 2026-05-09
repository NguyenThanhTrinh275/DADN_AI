#!/usr/bin/env python3
"""
Trích xuất đặc trưng và lưu file .npz (features RAW, chưa PCA).

Tương thích với `python main.py --use-feature-cache` — PCA được áp trong main sau khi load cache.

Ví dụ (local):
    python scripts/extract_features_only.py --data-path data --output results/cache/features_dinov2_vitb14_full.npz

Ví dụ (Kaggle):
    python scripts/extract_features_only.py \\
        --data-path /kaggle/input/imagenet-hard \\
        --output /kaggle/working/features_dinov2_vitb14_full.npz \\
        --project-root /kaggle/working/DADN_TTNT \\
        --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract image features to .npz (RAW, no PCA). Use with main.py --use-feature-cache.",
    )
    p.add_argument(
        "--data-path",
        required=True,
        help="Thư mục chứa file .parquet (glob *.parquet).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Đường dẫn file .npz đầu ra. Mặc định: <project-root>/results/cache/features_<MODEL>_<full|sampleN>.npz",
    )
    p.add_argument(
        "--project-root",
        default=None,
        help="Thư mục gốc repo (chứa src/). Mặc định: thư mục hiện tại.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Giới hạn số mẫu (None = toàn bộ).",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="cuda | cpu | auto (giống FeatureExtractor)",
    )
    p.add_argument(
        "--model-name",
        default=None,
        help="Ghi đè config.MODEL_NAME (vd dinov2_vitl14).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = os.path.abspath(args.project_root or os.getcwd())
    if root not in sys.path:
        sys.path.insert(0, root)

    os.chdir(root)

    from src.config import config

    if args.model_name:
        config.MODEL_NAME = args.model_name

    from src.utils.data_loader import load_data, extract_images_and_labels
    from src.models.feature_extractor import FeatureExtractor
    from src.utils.feature_cache import save_feature_cache

    data_path = os.path.abspath(args.data_path)
    print(f"DATA_PATH = {data_path}")

    df = load_data(path=data_path, sample_size=args.sample_size)
    images, _, true_label_sets = extract_images_and_labels(df, return_multilabel=True)

    extractor = FeatureExtractor(device=args.device)
    if config.USE_TTA:
        print(f"  TTA: scales={config.TTA_SCALES}, flips={config.TTA_FLIPS}")
        features = extractor.extract_features_tta(
            images,
            scales=config.TTA_SCALES,
            flips=config.TTA_FLIPS,
        )
    else:
        features = extractor.extract_features(images)

    print(f"Features shape: {features.shape}")

    if args.output:
        out = os.path.abspath(args.output)
    else:
        tag = "full" if args.sample_size is None else f"sample{args.sample_size}"
        subdir = os.path.join(root, "results", "cache")
        os.makedirs(subdir, exist_ok=True)
        out = os.path.join(subdir, f"features_{config.MODEL_NAME}_{tag}.npz")

    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)

    metadata = {
        "model_name": extractor.model_name,
        "device": str(extractor.device),
        "sample_size": args.sample_size,
        "feature_dim": int(features.shape[1]) if features.ndim == 2 else None,
        "n_samples": int(features.shape[0]),
        "tta_scales": list(config.TTA_SCALES) if config.USE_TTA else None,
        "tta_flips": list(config.TTA_FLIPS) if config.USE_TTA else None,
        "data_path": data_path,
    }
    save_feature_cache(out, features, true_label_sets, metadata)
    print(f"Saved feature cache: {out}")


if __name__ == "__main__":
    main()
