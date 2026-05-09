"""
Hyperparameter Tuning: k (k-NN) và Resolution (Leiden / Louvain)
==================================================================

Mục tiêu:
    Tìm bộ tham số tối ưu cho pipeline phân cụm ảnh ImageNet-Hard
    bằng Grid Search 2 pha:

    Phase 1 — Coarse Search:
        k          ∈ {5, 10, 15, 20, 25, 30}
        resolution ∈ {1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 100.0}

    Phase 2 — Fine-Tuning:
        Zoom vào ±2 bước xung quanh bộ tham số tốt nhất từ Phase 1.

Điều kiện tiên quyết:
    Feature cache (.npz) phải tồn tại. Nếu chưa có, hãy chạy:
        python main.py --save-feature-cache

Cách dùng:
    # Chạy đầy đủ cả hai pha
    python tune_params.py

    # Chỉ chạy Phase 1 (Coarse)
    python tune_params.py --phase 1

    # Chỉ chạy Phase 2 (Fine-Tuning) sau khi đã có kết quả Phase 1
    python tune_params.py --phase 2

    # Đặt đường dẫn cache tùy chỉnh
    python tune_params.py --cache-path results/cache/features_dinov2_vits14_full.npz

    # Chọn file output riêng
    python tune_params.py --output results/tuning_results.csv

Metric tổng hợp (Composite Score) — ưu tiên Accuracy:
    Score = 0.50 × Accuracy + 0.20 × NMI + 0.15 × ARI + 0.10 × Purity + 0.05 × Modularity
"""

import argparse
import os
import sys
import time
import csv
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from src.config import config
from src.utils.feature_cache import load_feature_cache
from src.utils.feature_postprocess import pca_whiten
from src.models.graph_builder import build_knn_graph
from src.models.clustering import cluster_leiden, cluster_louvain
from src.utils.evaluation import evaluate_clustering, get_cluster_statistics

N_TARGET_CLASSES = 830

# Không gian tham số Phase 1 (Coarse)
# K cao + resolution thấp hơn để n_clusters bám sát N_TARGET_CLASSES = 830
COARSE_K_VALUES          = [10, 15, 20, 25, 30]
COARSE_RESOLUTION_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]

# Trọng số Composite Score (ưu tiên Accuracy — mục tiêu chính của project)
WEIGHT_ACCURACY   = 0.50
WEIGHT_NMI        = 0.20
WEIGHT_ARI        = 0.15
WEIGHT_PURITY     = 0.10
WEIGHT_MODULARITY = 0.05

# Tên cột CSV output
CSV_FIELDNAMES = [
    "timestamp",
    "phase",
    "k_neighbors",
    "resolution",
    "algorithm",
    "n_clusters",
    "avg_cluster_size",
    "nmi",
    "accuracy",
    "purity",
    "ari",
    "modularity",
    "composite_score",
    "elapsed_sec",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def composite_score(nmi: float, ari: float, accuracy: float, purity: float, modularity: float) -> float:
    """Tính Composite Score từ 5 metrics."""
    # Clamp để tránh NaN/âm ảnh hưởng
    nmi = max(nmi, 0.0)
    ari = max(ari, 0.0)
    accuracy = max(accuracy, 0.0)
    purity = max(purity, 0.0)
    modularity = max(modularity, 0.0)
    return (
        WEIGHT_NMI * nmi + WEIGHT_ARI * ari + WEIGHT_ACCURACY * accuracy
        + WEIGHT_PURITY * purity + WEIGHT_MODULARITY * modularity
    )


def ensure_csv_header(csv_path: str) -> None:
    """Tạo header CSV nếu file chưa tồn tại."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
        print(f"  ✓ Tạo file CSV: {csv_path}")


def append_row(csv_path: str, row: dict) -> None:
    """Ghi thêm một dòng vào CSV (append mode)."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)


def fmt(v) -> str:
    """Format số thực 6 chữ số, trả '' nếu NaN/None."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return f"{v:.6f}"


# ══════════════════════════════════════════════════════════════════════════════
# CORE: EVALUATE ONE (k, resolution) PAIR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_pair(
    features: np.ndarray,
    true_label_sets,
    k: int,
    resolution: float,
    phase: str,
    csv_path: str,
) -> list[dict]:
    """
    Build đồ thị với k, sau đó chạy Leiden + Louvain với resolution.
    Trả về list 2 dòng kết quả (Leiden, Louvain).
    """
    t0 = time.time()

    # Xây dựng đồ thị
    graph = build_knn_graph(features, k=k)
    t_graph = time.time() - t0

    rows = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    algo_runners = [
        ("Leiden",  lambda g, r: cluster_leiden(g, resolution=r)),
        ("Louvain", lambda g, r: cluster_louvain(g, resolution=r)),
    ]

    for algo_name, runner in algo_runners:
        t_algo_start = time.time()

        pred_labels = runner(graph, resolution)
        metrics = evaluate_clustering(true_label_sets, pred_labels, graph)

        elapsed = t_graph + (time.time() - t_algo_start)

        nmi_v        = metrics.get("NMI", float("nan"))
        ari_v        = metrics.get("ARI", float("nan"))
        accuracy_v   = metrics.get("Accuracy", float("nan"))
        purity_v     = metrics.get("Purity", float("nan"))
        modularity_v = metrics.get("Modularity", float("nan"))
        score_v      = composite_score(nmi_v, ari_v, accuracy_v, purity_v, modularity_v)

        stats = get_cluster_statistics(pred_labels)

        row = {
            "timestamp":       timestamp,
            "phase":           phase,
            "k_neighbors":     k,
            "resolution":      resolution,
            "algorithm":       algo_name,
            "n_clusters":      stats["n_clusters"],
            "avg_cluster_size": fmt(stats["avg_cluster_size"]),
            "nmi":             fmt(nmi_v),
            "accuracy":        fmt(accuracy_v),
            "purity":          fmt(purity_v),
            "ari":             fmt(ari_v),
            "modularity":      fmt(modularity_v),
            "composite_score": fmt(score_v),
            "elapsed_sec":     fmt(elapsed),
        }

        append_row(csv_path, row)
        rows.append(row)

        print(
            f"    [{algo_name}] k={k:>2d} res={resolution:>6.1f} | "
            f"NMI={nmi_v:.4f} ARI={ari_v:.4f} Acc={accuracy_v:.4f} "
            f"Purity={purity_v:.4f} Mod={modularity_v:.4f} Score={score_v:.4f} "
            f"Clusters={stats['n_clusters']} ({elapsed:.1f}s)"
        )

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: COARSE SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def run_coarse_search(features, true_label_sets, csv_path: str) -> list[dict]:
    """Chạy Grid Search thô trên toàn bộ không gian tham số."""
    all_rows = []
    total = len(COARSE_K_VALUES) * len(COARSE_RESOLUTION_VALUES)

    print("\n" + "═" * 70)
    print("  PHASE 1 — COARSE SEARCH")
    print(f"  k values          : {COARSE_K_VALUES}")
    print(f"  Resolution values : {COARSE_RESOLUTION_VALUES}")
    print(f"  Total combinations: {total} × 2 algorithms = {total * 2} runs")
    print("═" * 70)

    count = 0
    t_phase_start = time.time()

    for k, resolution in product(COARSE_K_VALUES, COARSE_RESOLUTION_VALUES):
        count += 1
        print(f"\n[{count}/{total}] k={k}, resolution={resolution}")
        rows = evaluate_pair(features, true_label_sets, k, resolution, "coarse", csv_path)
        all_rows.extend(rows)

    elapsed_total = time.time() - t_phase_start
    print(f"\n✓ Phase 1 hoàn thành trong {elapsed_total/60:.1f} phút")
    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════

def _build_fine_grid(best_k: int, best_res: float) -> tuple[list[int], list[float]]:
    """
    Tạo lưới tinh chỉnh quanh giá trị tốt nhất từ Phase 1.
    - k: ±2 bước quanh best_k (bước 1, min=1)
    - resolution: ±2 bước với step = max(1, best_res * 0.15)
    """
    # k grid
    k_step = 1
    k_candidates = sorted(set(
        max(1, best_k + d * k_step) for d in range(-2, 3)
    ))
    # Loại bỏ các k đã có trong coarse để không chạy lại
    k_fine = [k for k in k_candidates if k not in COARSE_K_VALUES]
    if not k_fine:
        # Nếu tất cả đã có, vẫn bao gồm best_k để có điểm tham chiếu
        k_fine = [best_k]

    # resolution grid
    res_step = max(1.0, round(best_res * 0.15, 1))
    res_candidates = sorted(set(
        round(max(0.1, best_res + d * res_step), 1) for d in range(-2, 3)
    ))
    res_fine = [r for r in res_candidates if r not in COARSE_RESOLUTION_VALUES]
    if not res_fine:
        res_fine = [best_res]

    return k_fine, res_fine


def run_fine_search(
    features, true_label_sets, coarse_rows: list[dict], csv_path: str
) -> list[dict]:
    """Tinh chỉnh quanh bộ tham số tốt nhất tìm được từ Phase 1."""

    # Tìm bộ tham số tốt nhất theo Composite Score (Leiden hoặc Louvain)
    best_row = max(
        coarse_rows,
        key=lambda r: float(r["composite_score"]) if r["composite_score"] else -1,
    )
    best_k   = int(best_row["k_neighbors"])
    best_res = float(best_row["resolution"])

    k_fine, res_fine = _build_fine_grid(best_k, best_res)
    total = len(k_fine) * len(res_fine)

    print("\n" + "═" * 70)
    print("  PHASE 2 — FINE-TUNING")
    print(f"  Best (Phase 1): k={best_k}, resolution={best_res}")
    print(f"  Fine k values          : {k_fine}")
    print(f"  Fine resolution values : {res_fine}")
    print(f"  Total combinations     : {total} × 2 algorithms = {total * 2} runs")
    print("═" * 70)

    all_rows = []
    count = 0
    t_phase_start = time.time()

    for k, resolution in product(k_fine, res_fine):
        count += 1
        print(f"\n[{count}/{total}] k={k}, resolution={resolution}")
        rows = evaluate_pair(features, true_label_sets, k, resolution, "fine", csv_path)
        all_rows.extend(rows)

    elapsed_total = time.time() - t_phase_start
    print(f"\n✓ Phase 2 hoàn thành trong {elapsed_total/60:.1f} phút")
    return all_rows


def print_report(all_rows: list[dict], top_n: int = 5) -> None:
    """In báo cáo top-N bộ tham số tốt nhất theo Composite Score."""
    if not all_rows:
        print("Không có kết quả để báo cáo.")
        return

    # Sắp xếp giảm dần theo composite_score
    def safe_score(r):
        try:
            return float(r["composite_score"])
        except (ValueError, TypeError):
            return -1.0

    sorted_rows = sorted(all_rows, key=safe_score, reverse=True)
    top_rows = sorted_rows[:top_n]

    sep = "═" * 110
    print(f"\n{sep}")
    print(f"  TOP {top_n} CONFIGURATIONS (ranked by Composite Score)")
    print(sep)
    header = (
        f"{'Rank':<5} {'Phase':<7} {'k':>4} {'Resolution':>11} {'Algorithm':<9} "
        f"{'NMI':>7} {'ARI':>7} {'Acc':>7} {'Purity':>7} {'Mod':>7} {'Score':>7} {'Clusters':>9}"
    )
    print(header)
    print("-" * 110)

    for rank, row in enumerate(top_rows, start=1):
        nmi_v  = float(row["nmi"])        if row["nmi"]        else float("nan")
        ari_v  = float(row["ari"])        if row["ari"]        else float("nan")
        acc_v  = float(row["accuracy"])   if row.get("accuracy") else float("nan")
        pur_v  = float(row["purity"])     if row["purity"]     else float("nan")
        mod_v  = float(row["modularity"]) if row["modularity"] else float("nan")
        sc_v   = float(row["composite_score"]) if row["composite_score"] else float("nan")

        print(
            f"{rank:<5} {row['phase']:<7} {int(row['k_neighbors']):>4} "
            f"{float(row['resolution']):>11.1f} {row['algorithm']:<9} "
            f"{nmi_v:>7.4f} {ari_v:>7.4f} {acc_v:>7.4f} {pur_v:>7.4f} {mod_v:>7.4f} "
            f"{sc_v:>7.4f} {row['n_clusters']:>9}"
        )

    print(sep)

    # ── Helpers cho ranking phụ ───────────────────────────────────────────────
    def safe_acc(r):
        try:
            return float(r["accuracy"])
        except (ValueError, TypeError, KeyError):
            return -1.0

    def n_clusters_gap(r):
        try:
            return abs(int(r["n_clusters"]) - N_TARGET_CLASSES)
        except (ValueError, TypeError):
            return float("inf")

    def print_top_block(title, rows_subset, key_fn, reverse=True):
        print(f"\n  {title}")
        print("-" * 110)
        print(
            f"  {'Rank':<5} {'Phase':<7} {'k':>4} {'Resolution':>11} {'Algorithm':<9} "
            f"{'NMI':>7} {'ARI':>7} {'Acc':>7} {'Purity':>7} {'Mod':>7} {'Score':>7} {'Clusters':>9}"
        )
        sub = sorted(rows_subset, key=key_fn, reverse=reverse)[:top_n]
        for rank, row in enumerate(sub, start=1):
            nmi_v = float(row["nmi"])      if row["nmi"]      else float("nan")
            ari_v = float(row["ari"])      if row["ari"]      else float("nan")
            acc_v = float(row["accuracy"]) if row.get("accuracy") else float("nan")
            pur_v = float(row["purity"])   if row["purity"]   else float("nan")
            mod_v = float(row["modularity"]) if row["modularity"] else float("nan")
            sc_v  = float(row["composite_score"]) if row["composite_score"] else float("nan")
            print(
                f"  {rank:<5} {row['phase']:<7} {int(row['k_neighbors']):>4} "
                f"{float(row['resolution']):>11.1f} {row['algorithm']:<9} "
                f"{nmi_v:>7.4f} {ari_v:>7.4f} {acc_v:>7.4f} {pur_v:>7.4f} {mod_v:>7.4f} "
                f"{sc_v:>7.4f} {row['n_clusters']:>9}"
            )

    # ── Top theo Accuracy thuần (mục tiêu chính của project) ─────────────────
    print_top_block(
        f"TOP {top_n} BY ACCURACY (mục tiêu chính)",
        all_rows,
        key_fn=safe_acc,
        reverse=True,
    )

    # ── Top theo |n_clusters - N_TARGET_CLASSES| ─────────────────────────────
    print_top_block(
        f"TOP {top_n} GẦN N_TARGET_CLASSES = {N_TARGET_CLASSES} (sắp xếp tăng theo gap)",
        all_rows,
        key_fn=n_clusters_gap,
        reverse=False,
    )

    # ── Recommendation: ưu tiên ACCURACY ──────────────────────────────────────
    leiden_rows  = [r for r in all_rows if r["algorithm"] == "Leiden"]
    louvain_rows = [r for r in all_rows if r["algorithm"] == "Louvain"]

    best_acc          = max(all_rows,     key=safe_acc)
    best_leiden_acc   = max(leiden_rows,  key=safe_acc) if leiden_rows  else None
    best_louvain_acc  = max(louvain_rows, key=safe_acc) if louvain_rows else None

    print("\n  🏆 RECOMMENDATION (ưu tiên Accuracy):")
    print(f"    Best k (KNN)              : {int(best_acc['k_neighbors'])}  "
          f"(Accuracy={safe_acc(best_acc):.4f}, n_clusters={best_acc['n_clusters']})")
    if best_leiden_acc:
        print(f"    Best Resolution (Leiden)  : {float(best_leiden_acc['resolution']):.1f}  "
              f"→ Acc={safe_acc(best_leiden_acc):.4f}, "
              f"NMI={float(best_leiden_acc['nmi']):.4f}, "
              f"n_clusters={best_leiden_acc['n_clusters']}")
    if best_louvain_acc:
        print(f"    Best Resolution (Louvain) : {float(best_louvain_acc['resolution']):.1f}  "
              f"→ Acc={safe_acc(best_louvain_acc):.4f}, "
              f"NMI={float(best_louvain_acc['nmi']):.4f}, "
              f"n_clusters={best_louvain_acc['n_clusters']}")

    print(f"\n  Để áp dụng, cập nhật src/config.py:")
    print(f"    K_NEIGHBORS        = {int(best_acc['k_neighbors'])}")
    if best_leiden_acc:
        print(f"    LEIDEN_RESOLUTION  = {float(best_leiden_acc['resolution']):.1f}")
    if best_louvain_acc:
        print(f"    LOUVAIN_RESOLUTION = {float(best_louvain_acc['resolution']):.1f}")
    print(sep)


def load_existing_csv(csv_path: str) -> list[dict]:
    """Đọc kết quả đã có trong CSV để dùng lại (tránh chạy lại)."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Grid Search tối ưu tham số k và Resolution cho Leiden/Louvain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["efficientnet_v2_l", "efficientnet_v2_m", "efficientnet_v2_s",
                 "efficientnet_b7", "resnet50", "dinov2_vits14"],
        help="Tên model để tự resolve cache path và output CSV. "
             "Nếu cung cấp, --cache-path và --output sẽ được tự động đặt theo model.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Đường dẫn feature cache .npz (mặc định: results/cache/features_<model>_full.npz)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="File CSV lưu kết quả tuning (mặc định: results/tuning_<model>.csv hoặc results/tuning_results.csv)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help="Chỉ chạy Phase 1 (coarse) hoặc Phase 2 (fine-tuning). Mặc định: cả hai.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Số lượng top configurations in ra báo cáo (mặc định: 5)",
    )

    args = parser.parse_args()

    # ── Xác định model name (ưu tiên --model, fallback về config) ─────────────
    model_name = args.model if args.model else config.MODEL_NAME

    # ── Xác định đường dẫn cache ──────────────────────────────────────────────
    if args.cache_path is not None:
        cache_path = args.cache_path
    else:
        cache_path = os.path.join(
            config.RESULTS_PATH, "cache", f"features_{model_name}_full.npz"
        )

    # ── Xác định output CSV ───────────────────────────────────────────────────
    if args.output is not None:
        output_csv = args.output
    elif args.model:
        output_csv = f"results/tuning_{args.model}.csv"
    else:
        output_csv = "results/tuning_results.csv"

    # ── Kiểm tra cache ────────────────────────────────────────────────────────
    if not os.path.exists(cache_path):
        print(f"[ERROR] Không tìm thấy feature cache: {cache_path}")
        print("  → Hãy chạy trước: python main.py --save-feature-cache")
        sys.exit(1)

    print("=" * 70)
    print("  HYPERPARAMETER TUNING: k (k-NN) & Resolution (Leiden / Louvain)")
    print("  Dataset: ImageNet-Hard (full)")
    print(f"  Model        : {model_name}")
    print(f"  Feature cache: {cache_path}")
    print(f"  Output CSV   : {output_csv}")
    print("=" * 70)

    # ── Load features ─────────────────────────────────────────────────────────
    print("\n[STEP 1] Loading features from cache...")
    cache_data      = load_feature_cache(cache_path)
    features        = cache_data["features"]
    true_label_sets = cache_data["true_label_sets"]
    print(f"  ✓ Features shape: {features.shape}")
    print(f"  ✓ N samples     : {len(true_label_sets)}")

    # ── PCA whitening (giống main.py để kết quả tuning khớp với main pipeline) ─
    if getattr(config, "USE_PCA_WHITEN", False):
        print("\n[STEP 1.5] Áp dụng PCA whitening trên features...")
        features = pca_whiten(
            features,
            n_components=config.PCA_DIM,
            random_state=config.RANDOM_STATE,
        )
        print(f"  ✓ Features sau whitening: {features.shape}")

    # ── Chuẩn bị CSV ─────────────────────────────────────────────────────────
    ensure_csv_header(output_csv)

    all_rows: list[dict] = []

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if args.phase is None or args.phase == 1:
        coarse_rows = run_coarse_search(features, true_label_sets, output_csv)
        all_rows.extend(coarse_rows)
    else:
        # Phase 2 only: load Phase 1 results từ CSV
        print("\n[INFO] Đọc kết quả Phase 1 từ CSV để thực hiện Fine-Tuning...")
        existing = load_existing_csv(output_csv)
        coarse_rows = [r for r in existing if r.get("phase") == "coarse"]
        if not coarse_rows:
            print("[ERROR] Không tìm thấy kết quả Phase 1 (coarse) trong CSV.")
            print("  → Hãy chạy Phase 1 trước: python tune_params.py --phase 1")
            sys.exit(1)
        print(f"  ✓ Đã đọc {len(coarse_rows)} dòng Phase 1 từ {output_csv}")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if args.phase is None or args.phase == 2:
        fine_rows = run_fine_search(features, true_label_sets, coarse_rows, output_csv)
        all_rows.extend(fine_rows)

    # ── Nếu chỉ đọc CSV (phase 2 only), gộp tất cả để report ─────────────────
    if not all_rows:
        all_rows = load_existing_csv(output_csv)

    # ── Báo cáo ───────────────────────────────────────────────────────────────
    print_report(all_rows, top_n=args.top)

    print(f"\n✓ Toàn bộ kết quả đã lưu tại: {output_csv}")
    print("✓ Tuning hoàn thành!\n")


if __name__ == "__main__":
    main()
