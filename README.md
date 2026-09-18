# Hệ Thống AI Hỗ Trợ Nhận Diện Và Phân Loại Rác Thải (YOLO11s + CBAM)

> **Hệ thống ứng dụng mô hình thị giác máy tính YOLO11s kết hợp cơ chế chú ý CBAM nhằm hỗ trợ phát hiện, khoanh vùng và phân loại rác thải thành 7 nhóm trên nền tảng Web và luồng Camera.**

[![YOLO11s-CBAM](https://img.shields.io/badge/Model-YOLO11s--CBAM-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![mAP@50](https://img.shields.io/badge/mAP%4050-91.18%25-brightgreen.svg)](#5-kết-quả-đánh-giá-mô-hình-evaluation-metrics)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)
[![Deploy](https://img.shields.io/badge/Deploy-Render-46e3b7.svg)](#12-triển-khai-hệ-thống-cloud-deployment)

---

## 1. Giới thiệu tổng quan

Ô nhiễm rác thải là một trong những thách thức môi trường cấp bách hiện nay. Việc hỗ trợ phân loại rác ngay tại nguồn giúp tăng tỷ lệ tái chế, giảm tải cho các bãi chôn lấp và tiết kiệm tài nguyên.

Dự án xây dựng một giải pháp thử nghiệm từ **huấn luyện mô hình Deep Learning** đến **triển khai ứng dụng Web Full-stack**:

- **Nhận diện qua Webcam:** Bắt khung hình liên tục với độ trễ xử lý khoảng 28 - 32 ms trên GPU.
- **Phân loại ảnh tĩnh:** Người dùng có thể tải ảnh chụp từ thiết bị cá nhân để mô hình phân loại tự động.
- **Gợi ý phân loại thùng rác:** Hiển thị thông tin quy chuẩn màu thùng rác tương ứng (cam, xanh dương, xanh lá, vàng) để người dùng tham khảo.
- **Lưu trữ lịch sử & Thống kê:** Ghi nhận số lượt nhận diện, tỷ lệ các nhóm rác và xem lại các ảnh đã phân tích.
- **Hiển thị trực quan thông số:** Trình diễn biểu đồ hàm mất mát (Loss Curve), ma trận nhầm lẫn (Confusion Matrix) và thông số kỹ thuật trên giao diện Web.

---

## 2. Các loại rác trong phạm vi phân loại (7 Nhóm)

Mô hình được huấn luyện để nhận diện 7 nhóm rác thải sinh hoạt phổ biến:

| STT | Nhãn Class | Tên tiếng Việt | Nhóm phân loại | Quy chuẩn màu thùng rác | Chỉ dẫn xử lý tham khảo |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | `battery` | **Rác pin** | Rác nguy hại | 🟠 Thùng Cam / Đỏ | Thu gom riêng, tuyệt đối không đốt hoặc chôn chung với rác thường |
| 2 | `cardboard` | **Bìa carton** | Rác tái chế | 🟡 Thùng Vàng / Xanh dương | Gấp phẳng, giữ khô ráo để chuyển đến điểm tái chế giấy |
| 3 | `paper` | **Rác giấy** | Rác tái chế | 🟡 Thùng Vàng / Xanh dương | Thu gom sách báo cũ, hạn chế dính dầu mỡ thực phẩm |
| 4 | `glass` | **Thủy tinh** | Rác tái chế | 🔵 Thùng Xanh dương | Rửa sạch chai lọ, để riêng đồ vỡ nhằm đảm bảo an toàn |
| 5 | `metal` | **Kim loại** | Rác tái chế | ⚪ Thùng Xám / Bạc | Vỏ lon nhôm, đồ hộp kim loại nên được làm sạch và bóp xẹp |
| 6 | `plastic` | **Rác nhựa** | Rác tái chế | 🟡 Thùng Vàng / Xanh dương | Chai nhựa PET, can nhựa, bóp xẹp trước khi bỏ thùng |
| 7 | `organic` | **Rác hữu cơ** | Rác phân hủy | 🟢 Thùng Xanh lá cây | Cuống rau, vỏ củ quả, thức ăn thừa thích hợp ủ làm phân compost |

---

## 3. Bộ Dữ Liệu Huấn Luyện & Phương Pháp Chia K-Fold (Dataset & 3-Fold Cross-Validation)

Hệ thống sử dụng toàn bộ **26.048 ảnh** với tổng cộng **59.080 Bounding Boxes** phân bố trên 7 nhóm rác thải sinh hoạt (`battery`, `cardboard`, `paper`, `glass`, `metal`, `plastic`, `organic`).

### 🔄 Phương pháp kiểm chứng chéo 3-Fold (3-Fold Cross-Validation):
Để đảm bảo tính khách quan khoa học cao nhất và đánh giá chính xác năng lực tổng quát hóa của mô hình, toàn bộ 26.048 ảnh được gộp lại, xáo trộn ngẫu nhiên có kiểm soát (`random_seed = 42`) và chia đều thành **3 Fold độc lập** (K=3):

```text
                     TỔNG TẬP DỮ LIỆU (26.048 ẢNH / 59.080 BOXES)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
      FOLD 1 (8.682 ảnh)         FOLD 2 (8.682 ảnh)         FOLD 3 (8.684 ảnh)
      19.386 Bounding Boxes      20.061 Bounding Boxes      19.633 Bounding Boxes
```

### 📋 Cơ chế huấn luyện và kiểm định xoay vòng (K=3):
* **Lần 1 (Fold 1):** Huấn luyện trên Fold 2 + Fold 3 (17.366 ảnh ~ 66.7%), Kiểm định trên **Fold 1 (8.682 ảnh ~ 33.3%)**.
* **Lần 2 (Fold 2):** Huấn luyện trên Fold 1 + Fold 3 (17.366 ảnh ~ 66.7%), Kiểm định trên **Fold 2 (8.682 ảnh ~ 33.3%)**.
* **Lần 3 (Fold 3):** Huấn luyện trên Fold 1 + Fold 2 (17.364 ảnh ~ 66.7%), Kiểm định trên **Fold 3 (8.684 ảnh ~ 33.3%)**.

> 💡 **Ưu điểm học thuật vượt trội:**  
> - **100% toàn bộ 26.048 ảnh** đều lần lượt đóng vai trò làm tập kiểm định độc lập (Out-of-Fold Validation) ở một vòng lặp nhất định.
> - Kết quả đánh giá không phụ thuộc vào may rủi của một lần phân chia tập test cố định.
> - Cho phép tính toán độ ổn định của kiến trúc thông qua giá trị trung bình và độ lệch chuẩn (**Mean ± Std**).

### 📊 Thống kê chi tiết các Fold kiểm định:

| Phân vùng (Fold) | Số lượng ảnh | Tỷ lệ % | Số Bounding Boxes | Vai trò trong quá trình nghiệm thu |
| :--- | :---: | :---: | :---: | :--- |
| **Fold 1** | **8.682 ảnh** | **33.33%** | **19.386 boxes** | Tập kiểm định Fold 1 *(Mô hình đạt đỉnh cao nhất: mAP@50 đạt 91.18%)* |
| **Fold 2** | **8.682 ảnh** | **33.33%** | **20.061 boxes** | Tập kiểm định Fold 2 *(mAP@50 đạt 90.50%)* |
| **Fold 3** | **8.684 ảnh** | **33.34%** | **19.633 boxes** | Tập kiểm định Fold 3 *(mAP@50 đạt 90.90%)* |
| **TỔNG CỘNG** | **26.048 ảnh** | **100%** | **59.080 boxes** | Chuẩn hóa kích thước `640 x 640`, phân bố đồng đều 7 lớp rác |

### 📦 Phân bố số lượng Bounding Boxes theo 7 lớp rác thải:

| STT | Nhãn Class | Ý nghĩa phân loại | Số lượng Bounding Boxes | Tỷ lệ phân bố |
| :-: | :--- | :--- | :---: | :---: |
| 1 | `battery` | Pin / Pin điện tử (Rác nguy hại) | 8.059 boxes | 13.64% |
| 2 | `cardboard` | Thùng / Bìa carton (Rác tái chế) | 11.551 boxes | 19.55% |
| 3 | `paper` | Giấy / Báo / Tập vở (Rác tái chế) | 6.157 boxes | 10.42% |
| 4 | `glass` | Chai lọ / Mảnh vỡ thủy tinh (Rác tái chế) | 7.217 boxes | 12.22% |
| 5 | `metal` | Lon nhôm / Đồ hộp kim loại (Rác tái chế) | 8.999 boxes | 15.23% |
| 6 | `plastic` | Chai nhựa / Cốc nhựa / Túi nilon (Rác tái chế) | 10.173 boxes | 17.22% |
| 7 | `organic` | Rác thực phẩm / Vỏ rau củ quả (Rác hữu cơ) | 6.924 boxes | 11.72% |
| | **TỔNG CỘNG** | **7 nhóm rác thải chuẩn sinh hoạt** | **59.080 boxes** | **100%** |

---

## 4. Kiến Trúc Mô Hình & Quá Trình Huấn Luyện

Dự án phát triển trên kiến trúc nâng tiến **YOLO11s + CBAM** (tích hợp cơ chế chú ý kênh và không gian Convolutional Block Attention Module):

- **Số tầng (Layers):** 97 layers
- **Số lượng tham số (Parameters):** 8.951.735 tham số (8.95M)
- **Độ phức tạp tính toán:** 20.5 GFLOPs
- **Các phiên bản trọng số sử dụng:**
  - `bbest.pt` (PyTorch FP32 - 17.4 MB): Trọng số tối ưu nhất đạt được từ Fold 1 (thuộc kiểm định 3-Fold Cross-Validation), phục vụ chạy trên GPU CUDA hoặc inference backend.
  - `bbest.onnx` (ONNX Runtime - 34.4 MB): Định dạng ONNX tối ưu phục vụ việc nạp trực tiếp vào trình duyệt qua ONNX Runtime Web (WebGPU / WASM) và chạy mượt mà trên nền tảng Cloud Render CPU.

### ⚙️ Siêu tham số huấn luyện:

```text
- Model Architecture: YOLO11s + CBAM (Ultralytics + Attention Neck)
- Image Size: 640 x 640
- Môi trường phần cứng: Google Colab (GPU Tesla T4 16GB VRAM)
- Batch Size: 32
- Optimizer: AdamW (lr0 = 0.001, weight_decay = 0.0005)
- Early Stopping: Patience = 10 epochs
- Total Epochs: 100 epochs (Hội tụ toàn diện)
- Phương pháp kiểm chứng: 3-Fold Cross-Validation (K=3) trên toàn bộ 26.048 ảnh
- Online Augmentation: Mixup (0.15), Copy-Paste (0.3), Random Erasing (0.4), Dropout (0.1)
```

Quá trình huấn luyện chi tiết có thể theo dõi trong notebook: [`train_yolo11s_cbam_colab.ipynb`](train_yolo11s_cbam_colab.ipynb).

---

## 5. Kết Quả Đánh Giá Mô Hình (Evaluation Metrics)

Kết quả đánh giá định lượng trên tập **Validation Fold 1 (8.682 ảnh / 19.386 bounding boxes)**:

| Chỉ số đánh giá | Kết quả thực nghiệm (Fold 1) | Trung bình 3-Fold (Mean ± Std) | Ý nghĩa chuyên môn |
| :--- | :---: | :---: | :--- |
| **Precision (P)** | **93.05%** | **92.83% ± 0.15%** | Tỷ lệ dự đoán đúng trên tổng số dự đoán dương tính |
| **Recall (R)** | **85.63%** | **85.17% ± 0.51%** | Tỷ lệ phát hiện đối tượng trên tổng số đối tượng thực tế |
| **mAP @ 0.50** | **91.18%** | **90.87% ± 0.35%** | mean Average Precision tại ngưỡng IoU 0.50 |
| **mAP @ 0.50:0.95** | **78.82%** | **78.47% ± 0.29%** | mAP trung bình trên dải ngưỡng IoU từ 0.50 đến 0.95 |
| **Inference Latency** | **7.91 ms / frame** | **8.47 ± 1.82 ms** | Đạt **~126 FPS** trên GPU Tesla T4, đáp ứng vượt xa chuẩn Real-time (>30 FPS) |

### 📊 Hiệu năng chi tiết trên 7 nhóm rác (Fold 1):
* 🔋 **Pin (battery):** mAP@50 đạt **98.2%**, mAP@50-95 đạt **87.0%**
* 📦 **Bìa carton (cardboard):** mAP@50 đạt **98.4%**, mAP@50-95 đạt **86.4%**
* 📄 **Giấy (paper):** mAP@50 đạt **86.4%**, mAP@50-95 đạt **73.3%**
* 🍾 **Thủy tinh (glass):** mAP@50 đạt **99.2%**, mAP@50-95 đạt **93.5%**
* 🥫 **Kim loại (metal):** mAP@50 đạt **88.5%**, mAP@50-95 đạt **74.7%**
* 🧴 **Nhựa (plastic):** mAP@50 đạt **74.6%**, mAP@50-95 đạt **64.0%**
* 🍎 **Hữu cơ (organic):** mAP@50 đạt **93.0%**, mAP@50-95 đạt **72.8%**

### 🖼️ Minh chứng kết quả nhận diện thực tế:

Dưới đây là một ví dụ kết quả kiểm thử thực tế của mô hình (file minh chứng `test_output.jpg`):

![Minh chứng kết quả nhận diện rác](test_output.jpg)

---

## 6. Thực Nghiệm Đối Sánh Đa Kiến Trúc & Benchmarks (Model Comparisons)

Nhằm đảm bảo tính khách quan khoa học và làm cơ sở lựa chọn mô hình tối ưu cho hệ thống phân loại rác thải, đề tài đã triển khai thực nghiệm đối chứng giữa **3 trường phái kiến trúc Object Detection tiêu biểu**:

1. **One-Stage Detectors:** Họ mô hình YOLO thế hệ mới (**YOLO11s**, **YOLO11n**, **YOLOv8s**) và mô hình đề xuất cải tiến **YOLO11s + CBAM** (tích hợp cơ chế chú ý kênh & không gian).
2. **Two-Stage Detector (Kinh điển):** **Faster R-CNN** (Backbone MobileNetV3-Large FPN) đại diện cho trường phái trích xuất vùng ứng viên (Region Proposal).
3. **Transformer-based Detector (Hiện đại):** **RT-DETR-R18** (Real-Time Detection Transformer) đại diện cho trường phái Vision Transformer.

### 📋 Bảng tổng hợp đối sánh hiệu năng thực nghiệm (GPU Tesla T4 16GB, FP16, Batch=1, 200 ảnh test cố định):

| Trường phái | Mô hình | Số tham số (Params) | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50-95 (%) | Mean Latency | Tốc độ (FPS) | Đánh giá & Quyết định |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **One-Stage (Đề xuất)** | **YOLO11s + CBAM** | **8.95 M** | **93.05%** | **85.63%** | **91.18%** | **78.82%** | **7.91 ms** | **~126.4 FPS** | 🏆 **Tối ưu nhất: Chính xác cao vượt trội, siêu mượt Real-time** |
| One-Stage (Baseline) | YOLO11s (Gốc) | 9.43 M | 90.32% | 80.62% | 81.25% | 66.59% | 13.07 ms | 76.50 FPS | Cân bằng tốt nhưng độ chính xác thấp hơn CBAM |
| One-Stage | YOLO11n | 2.59 M | 84.10% | 71.50% | 79.20% | 58.40% | 12.64 ms | 79.10 FPS | Tốc độ nhanh nhưng mAP thấp, dễ bỏ sót rác nhỏ |
| One-Stage | YOLOv8s | 11.14 M | 82.40% | 74.10% | 81.50% | 60.30% | 10.36 ms | 96.49 FPS | Kiến trúc thế hệ cũ, độ trễ và độ chính xác kém hơn |
| **Two-Stage** | **Faster R-CNN** (MobileNetV3) | 18.96 M | 86.29% | 82.80% | 82.12% | 64.57% | 19.89 ms | 50.27 FPS | Nặng gấp đôi, độ trễ cao hơn, mAP@50-95 thấp |
| **Transformer** | **RT-DETR-R18** | 29.85 M | — | — | — | — | 42.50 ms | 23.53 FPS | Quá nặng (~30M params), FPS < 30 (không đạt chuẩn Real-time) |

### 📂 Danh mục mã nguồn huấn luyện & Benchmark đối chứng:

- [`train_yolo11s_cbam_colab.ipynb`](train_yolo11s_cbam_colab.ipynb): Huấn luyện & Đánh giá mô hình đề xuất **YOLO11s-CBAM** qua K-Fold Cross Validation.
- [`train_trash_yolo11_colab.ipynb`](train_trash_yolo11_colab.ipynb): Huấn luyện & Đánh giá mô hình cơ sở **YOLO11s** gốc.
- [`train_faster_rcnn_colab.ipynb`](train_faster_rcnn_colab.ipynb): Huấn luyện mô hình đối chứng **Faster R-CNN MobileNetV3** (Two-Stage).
- [`train_rtdetr_colab.ipynb`](train_rtdetr_colab.ipynb) / [`train_rtdetr_colab.py`](train_rtdetr_colab.py): Huấn luyện mô hình đối chứng **RT-DETR** (Vision Transformer).
- [`benchmark_latency_colab.py`](benchmark_latency_colab.py): Bộ công cụ benchmark GPU khoa học trên Colab (đo cùng lúc 4-6 mô hình trên 200 ảnh test cố định, 5 lần lặp, tính Latency, P95, P99, FPS).
- [`benchmark_speed.py`](benchmark_speed.py): Đo đạc chuyên sâu tốc độ suy luận triển khai của mô hình sản phẩm (`best.pt` / `best.onnx`) trên CPU, PyTorch FP16/FP32 và ONNX Runtime.

---

## 7. Các Kịch Bản Kiểm Thử (Testing Scripts)

Dự án cung cấp các kịch bản kiểm thử độc lập phục vụ kiểm tra và đánh giá:

- **`test_webcam.py`**: Khởi động luồng camera qua OpenCV, kết nối mô hình PyTorch CUDA trên card đồ họa, hiển thị HUD thông số FPS và bounding box theo thời gian thực.
- **`test_image.py`**: Nhận diện ảnh đơn lẻ từ đường dẫn file, trích xuất tọa độ Bounding Box, Confidence Score và lưu ảnh kết quả.
- **`test_api_endpoints.py`**: Kiểm thử tự động tính sẵn sàng (Health Check) và độ chính xác của các API endpoints (`/predict_image`, `/predict_frame`).
- **`test.py`**: Script thực thi kiểm tra và đánh giá tổng quát mô hình.

---

## 8. Kiến Trúc Hệ Thống (System Architecture)

Hệ thống được thiết kế theo mô hình **Hybrid Dual-Engine** linh hoạt:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          GIAO DIỆN NGƯỜI DÙNG                           │
│          Vue.js 3 SPA • Bootstrap 5 • Chart.js • Bootstrap Icons        │
│        (Dashboard, Realtime Camera, Tải ảnh lên, Lịch sử, Model)        │
└────────────────────┬───────────────────────────────┬────────────────────┘
                     │ Chế độ 1: In-Browser AI       │ Chế độ 2: REST API
                     ▼                               ▼
┌────────────────────────────────────────┐ ┌──────────────────────────────┐
│        ONNX RUNTIME WEB ENGINE         │ │        WEB API SERVER        │
│        (WebGPU / WASM SIMD)            │ │       Node.js Express        │
│  - Nạp bbest.onnx (34.4MB)              │ │  - Quản lý lịch sử, upload   │
│  - Xử lý trực tiếp camera trên browser │ └──────────────┬───────────────┘
└────────────────────────────────────────┘                │ Internal Proxy
                                                          ▼
                                           ┌──────────────────────────────┐
                                           │    PYTHON AI MICROSERVICE    │
                                           │   Flask + PyTorch (CUDA)     │
                                           │  - Nạp bbest.pt (17.4MB)      │
                                           │  - Xử lý ảnh tĩnh tải lên    │
                                           └──────────────────────────────┘
```

---

## 9. Cấu Trúc Thư Mục Repository

```text
.
├── client/                          # Mã nguồn Frontend Vue.js 3
│   ├── public/                      # Tài nguyên tĩnh (WASM, mô hình ONNX 34.4MB)
│   ├── src/                         # Components, Views, Routers, Services
│   │   ├── components/classify/     # WebcamClassifier & ImageClassifier
│   │   ├── views/                   # DashboardView, ModelInfoView, HistoryView...
│   │   └── services/                # api.js & yoloWebEngine.js (ONNX Runtime Web)
│   ├── package.json
│   └── vite.config.js
├── server/                          # Mã nguồn Backend Server
│   ├── config/                      # Cấu hình môi trường và đường dẫn lưu trữ
│   ├── controllers/                 # Bộ điều khiển logic (detection, history, stats)
│   ├── data/                        # File lưu trữ dữ liệu lịch sử nhận diện (JSON)
│   ├── repositories/                # Tầng truy xuất dữ liệu
│   ├── routes/                      # Định nghĩa các Route API (/api/predict, /history...)
│   ├── uploads/                     # Thư mục lưu 7 ảnh mẫu chuẩn (sample_*.jpg)
│   ├── server.js                    # Web server chính chạy Node.js Express (Port 5000)
│   └── yolo_service.py              # Dịch vụ AI Python Flask kết nối YOLO model (Port 5001)
├── modules/                         # Module mạng nơ-ron tùy biến
│   └── cbam.py                      # Module Convolutional Block Attention Module
├── models/                          # Cấu hình mạng nơ-ron
│   └── yolo11s-cbam.yaml            # Định nghĩa kiến trúc YOLO11s-CBAM
├── best.pt                          # Trọng số tối ưu nhất YOLO11s-CBAM (Best Fold 1) PyTorch (17.4 MB)
├── bestfold2.pt                     # Trọng số tối ưu YOLO11s-CBAM (Fold 2) PyTorch (17.4 MB)
├── bestfold3.pt                     # Trọng số tối ưu YOLO11s-CBAM (Fold 3) PyTorch (17.4 MB)
├── best.onnx                        # Mô hình ONNX Runtime cho Web & Render (34.4 MB)
├── data_balanced.yaml               # Cấu hình 7 nhãn và dataset
├── config.py                        # Cấu hình siêu tham số huấn luyện
├── train.py                         # Script huấn luyện YOLO11s trên Local
├── train_cbam_colab.py              # Script huấn luyện 3-Fold YOLO11s-CBAM
├── train_yolo11s_cbam_colab.ipynb   # Notebook huấn luyện 3-Fold trên Google Colab
├── train_faster_rcnn_colab.ipynb    # Notebook huấn luyện mô hình đối chứng Faster R-CNN
├── train_rtdetr_colab.py            # Script huấn luyện mô hình RT-DETR đối chứng
├── train_rtdetr_colab.ipynb         # Notebook huấn luyện RT-DETR trên Google Colab
├── train_trash_yolo11_colab.ipynb   # Notebook huấn luyện YOLO11 cơ bản
├── benchmark_latency_colab.py       # Bộ benchmark GPU đo đạc Latency & FPS 4 dòng mô hình
├── benchmark_speed.py               # Script benchmark tốc độ suy luận triển khai (PyTorch/CPU/ONNX)
├── test_webcam.py                   # Kiểm thử nhận diện webcam qua Python CUDA
├── test_image.py                    # Kiểm thử nhận diện trên ảnh tĩnh
├── test_api_endpoints.py            # Kiểm thử tự động các đầu API
├── evaluate_cbam.py                 # Đánh giá độ chính xác cải tiến mô hình CBAM
├── test_output.jpg                  # Ảnh mẫu minh chứng kết quả kiểm thử
├── requirements.txt                 # Danh sách thư viện Python cần thiết
├── Dockerfile                       # Cấu hình đóng gói Docker Container
├── docker-compose.yml               # Cấu hình chạy cụm Docker Compose (Port 7860)
├── start.sh                         # Kịch bản khởi động song song AI & Web trên Linux
├── DEPLOY_GUIDE.md                  # Hướng dẫn chi tiết triển khai hệ thống
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 10. Hướng Dẫn Cài Đặt & Chạy Localhost

### 📋 Yêu cầu môi trường:

- **Hệ điều hành:** Windows 10/11 hoặc Linux Ubuntu
- **Python:** 3.10 hoặc 3.11
- **Node.js:** 18.x hoặc 20.x LTS
- _(Khuyến nghị):_ Card đồ họa NVIDIA hỗ trợ CUDA để đạt tốc độ xử lý tốt nhất

### 🚀 Các bước cài đặt:

1. **Clone repository về máy:**

```bash
git clone https://github.com/Vzuong/trash-ai.git
cd trash-ai
```

2. **Cài đặt thư viện Python:**

```bash
pip install -r requirements.txt
```

3. **Cài đặt thư viện Node.js cho Server:**

```bash
cd server
npm install
cd ..
```

4. **Khởi động hệ thống:**
   - **Bước 1 (Khởi động Python AI Service trước tại Cửa sổ 1):**

```bash
python server/yolo_service.py
```

   *(Dịch vụ AI lắng nghe tại cổng `http://127.0.0.1:5001`)*

   - **Bước 2 (Khởi động Web Server tại Cửa sổ 2):**

```bash
node server/server.js
```

   *(Web Server lắng nghe tại cổng `http://localhost:5000`)*

5. **Truy cập ứng dụng:**
   Mở trình duyệt truy cập: **`http://localhost:5000`**

---

## 11. Chạy Ứng Dụng Bằng Docker

Hệ thống hỗ trợ đóng gói và chạy thông qua Docker Container:

```bash
# Xây dựng và khởi chạy container bằng Docker Compose
docker compose up --build -d

# Xem log hoạt động
docker compose logs -f
```

Sau khi khởi động thành công, truy cập ứng dụng tại cổng được cấu hình trong `docker-compose.yml`: **`http://localhost:7860`**.

---

## 12. Các Script Kiểm Thử Nhanh

```bash
# 1. Kiểm thử trên webcam máy tính (yêu cầu webcam)
python test_webcam.py

# 2. Kiểm thử trên 1 tấm ảnh mẫu
python test_image.py

# 3. Kiểm tra tính sẵn sàng của các API Endpoint
python test_api_endpoints.py
```

---

## 13. Triển Khai Hệ Thống (Cloud Deployment)

Hệ thống được triển khai thử nghiệm trên nền tảng đám mây **Render** thông qua Docker container:

- **Link Demo Trực Tuyến:** **[https://trash-ai-68ei.onrender.com](https://trash-ai-68ei.onrender.com)**
- **Môi trường Cloud:** Docker Container (Debian Linux, Python 3.11, Node.js 20, ONNX Runtime).

---

## 14. Hạn Chế Thực Tế & Hướng Phát Triển

### ⚠️ Hạn chế hiện tại của mô hình:

1. **Điều kiện ánh sáng:** Độ tin cậy nhận diện có thể giảm trong môi trường ánh sáng quá yếu hoặc bị ngược sáng mạnh.
2. **Vật thể biến dạng:** Rác bị dập nát nhiều, cháy xém hoặc bị che khuất phần lớn diện tích có thể dẫn đến việc phân loại chưa chính xác.
3. **Phạm vi phân loại:** Hiện tại mô hình chỉ giới hạn trong phạm vi 7 nhóm rác sinh hoạt phổ biến; các loại rác thải đặc thù (rác y tế, thiết bị điện tử cỡ lớn) chưa nằm trong tập nhãn huấn luyện.

### 🌟 Hướng phát triển tiếp theo:

- Tiếp tục thu thập thêm dữ liệu hình ảnh thực tế đa dạng tại các điểm thu gom rác để mở rộng tập dữ liệu.
- Bổ sung thêm các nhóm rác thải công nghệ (E-waste) và rác thải nguy hại y tế.
- Nghiên cứu tối ưu hóa mô hình để triển khai thử nghiệm trên các thiết bị phần cứng nhúng (như Raspberry Pi 5 hoặc NVIDIA Jetson Nano) nhằm ứng dụng vào mô hình thùng rác phân loại bán tự động.
