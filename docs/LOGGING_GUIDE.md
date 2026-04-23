<!-- @format -->

# 📊 Hệ Thống Logging Kết Quả Chạy (Results Logging System)

## Giới Thiệu

Hệ thống này tự động ghi **tất cả kết quả chạy** của pipeline vào file **CSV** để bạn dễ dàng theo dõi, so sánh các lần chạy khác nhau.

## 📋 Cấu Trúc CSV

File CSV sẽ có các cột sau:

| Cột                  | Ý Nghĩa                       | Ví Dụ                                 |
| -------------------- | ----------------------------- | ------------------------------------- |
| `timestamp`          | Thời gian chạy                | `2024-03-28 14:30:45`                 |
| `sample_size`        | Số sample được dùng           | `5000`, `full`                        |
| `model_name`         | Model extraction đặc trưng    | `dinov2_vits14`                       |
| `k_neighbors`        | Số k trong k-NN graph         | `10`                                  |
| `leiden_resolution`  | Resolution của Leiden         | `30.0`                                |
| `louvain_resolution` | Resolution của Louvain        | `30.0`                                |
| `algorithm`          | Thuật toán phân cụm           | `Leiden`, `Louvain`, `Infomap`, `LPA` |
| `n_clusters`         | Số cluster được phát hiện     | `42`                                  |
| `avg_cluster_size`   | Kích thước trung bình cluster | `119.2`                               |
| `nmi`                | Normalized Mutual Information | `0.524389`                            |
| `purity`             | Purity Score                  | `0.723456`                            |
| `ari`                | Adjusted Rand Index           | `0.615234`                            |
| `modularity`         | Modularity Score              | `0.832145`                            |
| `notes`              | Ghi chú thêm                  | `Best algorithm: Leiden`              |

## 🚀 Cách Sử Dụng

### 1️⃣ Chạy bình thường (logging tự động bật)

```bash
python main.py
```

✓ Kết quả sẽ được ghi vào `results/results.csv`

### 2️⃣ Chạy với sample nhỏ

```bash
python main.py --sample 1000
```

✓ Chạy với 1000 sample, kết quả vẫn được lưu vào CSV

### 3️⃣ Chạy với parameters khác

```bash
python main.py --sample 5000 --device gpu --save-feature-cache
```

### 4️⃣ Chạy mà không logging

```bash
python main.py --no-log
```

### 5️⃣ Chạy với đường dẫn CSV tùy chỉnh

```bash
python main.py --log-path custom_results.csv
```

## 📊 Ví Dụ File CSV

```
timestamp,sample_size,model_name,k_neighbors,leiden_resolution,louvain_resolution,algorithm,n_clusters,avg_cluster_size,nmi,purity,ari,modularity,notes
2024-03-28 14:30:45,5000,dinov2_vits14,10,30.0,30.0,Leiden,42,119.047619,0.524389,0.723456,0.615234,0.832145,Best algorithm: Leiden
2024-03-28 14:30:45,5000,dinov2_vits14,10,30.0,30.0,Louvain,38,131.578947,0.512345,0.710234,0.594123,0.815234,Best algorithm: Leiden
2024-03-28 14:30:45,5000,dinov2_vits14,10,30.0,30.0,Infomap,45,111.111111,0.498765,0.695234,0.578123,0.798234,Best algorithm: Leiden
2024-03-28 14:30:45,5000,dinov2_vits14,10,30.0,30.0,LPA,40,125.000000,0.508234,0.705234,0.589123,0.808234,Best algorithm: Leiden
2024-03-28 15:15:22,full,dinov2_vits14,10,30.0,30.0,Leiden,56,128.571429,0.534567,0.734567,0.625345,0.842345,Best algorithm: Leiden
```

## 🔧 Thay Đổi Parameters Động

Để thay đổi parameters (k, resolution, model, v.v.), hãy sửa trong **`src/config.py`**:

```python
# src/config.py
class Config:
    MODEL_NAME = 'dinov2_vits14'  # Thay đổi model
    K_NEIGHBORS = 10              # Thay đổi k cho k-NN
    LEIDEN_RESOLUTION = 30.0      # Thay đổi resolution Leiden
    LOUVAIN_RESOLUTION = 30.0     # Thay đổi resolution Louvain
    SAMPLE_SIZE = None            # Thay đổi sample size
```

Sau khi thay đổi, khi chạy `python main.py`, các parameters mới sẽ được tự động lưu vào CSV.

## 📈 Phân Tích Kết Quả

Bạn có thể dùng **Pandas** hoặc **Excel** để phân tích CSV:

### Python (Pandas)

```python
import pandas as pd

# Đọc CSV
df = pd.read_csv('results/results.csv')

# Xem toàn bộ kết quả
print(df)

# Xem kết quả của Leiden
leiden_results = df[df['algorithm'] == 'Leiden']
print(leiden_results[['timestamp', 'sample_size', 'nmi', 'purity', 'ari']])

# Tìm lần chạy tốt nhất (theo NMI)
best_run = df.loc[df['nmi'].idxmax()]
print(f"Best run: {best_run['algorithm']} with NMI = {best_run['nmi']}")

# So sánh trung bình các metric giữa các thuật toán
print(df.groupby('algorithm')[['nmi', 'purity', 'ari', 'modularity']].mean())
```

### Excel

- Mở file `results/results.csv` bằng Excel/Google Sheets
- Dùng Pivot Table để so sánh kết quả
- Tạo chart để visualize xu hướng

## 🛠️ Lỗi & Khắc Phục

### Lỗi: "CSV file not found"

→ Kiểm tra thư mục `results/` tồn tại chưa. Hệ thống sẽ tự tạo nếu chưa có.

### Lỗi: "Permission denied" khi ghi CSV

→ Đóng file CSV nếu đang mở trong Excel/Sheets, rồi chạy lại.

### CSV không có dữ liệu mới

→ Kiểm tra bạn đã chạy với `--no-log` flag chưa? Loại bỏ flag đó để bật logging.

## 💡 Thêm Ghi Chú Tùy Chỉnh

Khi muốn thêm ghi chú riêng, bạn có thể sửa trực tiếp trong `main.py`:

```python
logger.log_run(
    sample_size=sample_size,
    model_name=config.MODEL_NAME,
    k_neighbors=config.K_NEIGHBORS,
    leiden_resolution=config.LEIDEN_RESOLUTION,
    louvain_resolution=config.LOUVAIN_RESOLUTION,
    clustering_results=clustering_results_with_stats,
    notes=f'Best algorithm: {best_algo} - Run with GPU on RTX 3090'  # Thêm ghi chú
)
```

## 📝 Tóm Tắt

| Tính Năng              | Mô Tả                                                      |
| ---------------------- | ---------------------------------------------------------- |
| ✅ Tự động logging     | Mỗi lần chạy tự động lưu vào CSV                           |
| ✅ Theo dõi parameters | Ghi nhận tất cả config (k, resolution, model, sample_size) |
| ✅ So sánh thuật toán  | Dễ dàng so sánh NMI, Purity, ARI giữa các thuật toán       |
| ✅ Phân tích xu hướng  | Xem thường tìm thấy parameter tốt nhất                     |
| ✅ CSV chuẩn           | Dùng được trên Excel, Google Sheets, Pandas                |

---

**Chúc bạn thiết lập thành công! 🎉**
