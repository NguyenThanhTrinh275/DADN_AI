<!-- @format -->

# Hệ thống logging kết quả (CSV)

## Giới thiệu

Mỗi lần chạy [`main.py`](../main.py) thành công (trừ khi `--no-log`), pipeline ghi một **dòng cho mỗi thuật toán** phân cụm vào file CSV (mặc định `results/results.csv`).

Module: [`src/utils/result_logger.py`](../src/utils/result_logger.py).

## Cấu trúc cột CSV

| Cột | Ý nghĩa |
| --- | ------- |
| `timestamp` | Thời gian chạy |
| `sample_size` | Số mẫu hoặc `full` |
| `model_name` | Backbone trong `config.MODEL_NAME` |
| `k_neighbors` | `K_NEIGHBORS` |
| `leiden_resolution` | `LEIDEN_RESOLUTION` |
| `louvain_resolution` | `LOUVAIN_RESOLUTION` |
| `algorithm` | `Infomap`, `Leiden`, `Louvain`, `LPA` |
| `n_clusters` | Số cluster |
| `avg_cluster_size` | Trung bình kích thước cluster |
| `nmi` | Normalized Mutual Information |
| `accuracy` | Clustering accuracy (Hungarian) |
| `purity` | Purity |
| `ari` | Adjusted Rand Index |
| `modularity` | Modularity trên đồ thị |
| `notes` | Ghi chú (ví dụ thuật toán tốt nhất theo NMI) |

Thứ tự cột khớp `ResultLogger.fieldnames` trong code.

## Cách dùng

### Logging mặc định

```bash
python main.py
```

### Có sample

```bash
python main.py --sample 1000
```

### Cache + log

```bash
python main.py --use-feature-cache
```

### Tắt CSV

```bash
python main.py --no-log
```

### Đổi đường dẫn file log

```bash
python main.py --log-path results/my_experiments.csv
```

## Ví dụ header (rút gọn)

```text
timestamp,sample_size,model_name,k_neighbors,leiden_resolution,louvain_resolution,algorithm,n_clusters,avg_cluster_size,nmi,accuracy,purity,ari,modularity,notes
```

## Phân tích bằng Pandas

```python
import pandas as pd

df = pd.read_csv("results/results.csv")

# Lọc một thuật toán
df[df["algorithm"] == "Leiden"]

# So sánh trung bình metric
df.groupby("algorithm")[["nmi", "accuracy", "purity", "ari", "modularity"]].mean()

# Run có NMI cao nhất
df.loc[df["nmi"].idxmax()]
```

## Thay đổi tham số được ghi

Sửa [`src/config.py`](../src/config.py) rồi chạy lại `main.py`. Các giá trị `k_neighbors`, `resolution`, `model_name`, v.v. được đọc từ `config` tại thời điểm chạy.

## Ghi chú trong code

Có thể đổi tham số `notes=` trong lời gọi `logger.log_run(...)` trong [`main.py`](../main.py) nếu cần chuỗi cố định khác.

## Xử lý sự cố thường gặp

- **Không tạo file / không thêm dòng**: kiểm tra có dùng `--no-log` không; kiểm tra quyền ghi thư mục `results/`.
- **File CSV đang mở (Excel)**: đóng file rồi chạy lại để tránh lỗi ghi trên Windows.
