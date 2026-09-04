# Hệ Thống AI Hỗ Trợ Nhận Diện Và Phân Loại Rác Thải (YOLO11s)

> **Hệ thống ứng dụng mô hình thị giác máy tính YOLO11s nhằm hỗ trợ phát hiện, khoanh vùng và phân loại rác thải thành 7 nhóm trên nền tảng Web và luồng Camera.**

[![YOLO11s](https://img.shields.io/badge/Model-YOLO11s-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![mAP@50](https://img.shields.io/badge/mAP%4050-81.25%25-blue.svg)](#5-kết-quả-đánh-giá-mô-hình-evaluation-metrics)
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

## 3. Bộ Dữ Liệu Huấn Luyện (Dataset)

Bộ dữ liệu được tổng hợp từ nguồn dữ liệu thực tế kết hợp bổ sung các tập dữ liệu gán nhãn từ **Roboflow Universe** nhằm mở rộng quy mô và tăng mức độ đa dạng mẫu cho các nhóm rác đặc thù (`battery`, `glass`, `metal`, `plastic`).

### 🔄 Trình tự tiền xử lý và cam kết ngăn ngừa rò rỉ dữ liệu (Zero Data Leakage):
Quy trình được thực hiện nghiêm ngặt theo đúng phương pháp luận khoa học:
1. **Thu thập dữ liệu thô ban đầu (13.548 ảnh):** Kết hợp các nguồn ảnh gốc và tập dữ liệu bổ sung từ Roboflow Universe.
2. **Phân chia độc lập trước (Split first):** Bộ dữ liệu được chia độc lập thành Train gốc (5.820 ảnh), Validation gốc (4.636 ảnh) và Test độc lập (3.092 ảnh).
3. **Offline Augmentation chỉ trên tập Train:** Kịch bản `balance_dataset.py` chỉ can thiệp trên 5.820 ảnh của tập Train để cân đối tỷ lệ giữa các lớp, nâng tập Train lên 18.320 ảnh.
4. **Bảo toàn tính khách quan của Val & Test:** Toàn bộ ảnh của tập **Validation** và **Test** được giữ nguyên bản, hoàn toàn không áp dụng bất kỳ kỹ thuật tăng cường nhân tạo nào.

```text
Dữ liệu thô ban đầu (13.548 ảnh)
        │
        ├── Phân chia độc lập (Split)
        │     ├── Train gốc: 5.820 ảnh
        │     ├── Validation: 4.636 ảnh (Giữ nguyên bản)
        │     └── Test: 3.092 ảnh (Giữ nguyên bản)
        │
        └── Offline Augmentation (CHỈ áp dụng trên tập Train qua balance_dataset.py)
              └── Train sau cân đối: 18.320 ảnh
  
➔ TỔNG DỮ LIỆU CUỐI CÙNG: 18.320 (Train) + 4.636 (Val) + 3.092 (Test) = 26.048 ảnh
```

### 📊 Thống kê phân chia dữ liệu:
* **Tổng số lượng ảnh:** **26.048 ảnh** gán nhãn chuẩn YOLO (Bounding Box: `class x_center y_center width height`).
* **Kích thước chuẩn hóa:** `640 x 640 pixels`.

| Tập dữ liệu | Số lượng ảnh | Tỷ lệ % | Bản chất dữ liệu | Mục đích sử dụng |
| :--- | :---: | :---: | :--- | :--- |
| **Train Set** | **18.320 ảnh** | **70.33%** | Ảnh gốc + ảnh biến thể từ Offline Augmentation | Cập nhật trọng số mạng nơ-ron qua lan truyền ngược |
| **Validation Set** | **4.636 ảnh** | **17.80%** | Ảnh nguyên bản, không augmentation | Giám sát hiện tượng Overfitting & lưu checkpoint `best.pt` |
| **Test Set (Độc lập)** | **3.092 ảnh** | **11.87%** | Ảnh nguyên bản, không augmentation | Đánh giá khách quan năng lực tổng quát hóa của mô hình |
| **TỔNG CỘNG** | **26.048 ảnh** | **100%** | **59.080 Bounding Boxes** | Phân bố trên 7 nhóm rác thải |

### 📦 Phân bố chi tiết số lượng Bounding Boxes theo từng lớp:

| STT | Nhãn Class | Tập TRAIN (Huấn luyện) | Tập VAL (Kiểm định) | Tập TEST (Kiểm thử độc lập) | Tổng Bounding Boxes |
| :-: | :--- | :---: | :---: | :---: | :---: |
| 1 | `battery` (Pin) | 6.137 boxes | 1.112 boxes | **810 boxes** | 8.059 |
| 2 | `cardboard` (Bìa carton) | 7.150 boxes | 2.404 boxes | **1.997 boxes** | 11.551 |
| 3 | `paper` (Giấy) | 3.800 boxes | 1.370 boxes | **987 boxes** | 6.157 |
| 4 | `glass` (Thủy tinh) | 6.036 boxes | 738 boxes | **443 boxes** | 7.217 |
| 5 | `metal` (Kim loại) | 6.249 boxes | 1.760 boxes | **990 boxes** | 8.999 |
| 6 | `plastic` (Nhựa) | 8.434 boxes | 1.026 boxes | **713 boxes** | 10.173 |
| 7 | `organic` (Rác hữu cơ) | 3.825 boxes | 1.960 boxes | **1.139 boxes** | 6.924 |
| | **TỔNG CỘNG** | **41.631 boxes** | **10.370 boxes** | **7.079 boxes** | **59.080 boxes** |

---

## 4. Kiến Trúc Mô Hình & Quá Trình Huấn Luyện

Dự án thử nghiệm trên kiến trúc **YOLO11s** (phiên bản thuộc dòng YOLO do Ultralytics phát triển):

- **Số tầng (Layers):** 101 layers
- **Số lượng tham số (Parameters):** 9.415.509 tham số
- **Độ phức tạp tính toán:** 21.4 GFLOPs
- **Các phiên bản trọng số sử dụng:**
  - `best.pt` (PyTorch FP32 - 72.5 MB): Lưu trữ trọng số mô hình cùng trạng thái Optimizer (AdamW) phục vụ cho việc fine-tuning tiếp hoặc chạy trên máy chủ GPU CUDA. *(Kích thước trọng số mạng nơ-ron thuần túy tương đương ~18.4 – 19.2 MB như file base `yolo11s.pt`)*.
  - `best.onnx` (ONNX INT8 Quantized - 9.4 MB): Định dạng ONNX tối ưu qua kỹ thuật lượng tử hóa động (Dynamic Quantization INT8), phục vụ việc nạp trực tiếp vào trình duyệt qua ONNX Runtime Web.

### ⚙️ Siêu tham số huấn luyện:

```text
- Model Architecture: YOLO11s (Ultralytics)
- Image Size: 640 x 640
- Môi trường phần cứng: Google Colab & Local Machine (NVIDIA GeForce RTX 4060, 12GB RAM)
- Batch Size: 16 - 32 (Phù hợp với môi trường Google Colab GPU & RTX 4060 12GB RAM)
- Optimizer: AdamW (lr0 = 0.001, weight_decay = 0.0005)
- Learning Rate Scheduler: Cosine Annealing (cos_lr = True)
- Maximum Epochs thiết lập: 100 epochs (Early stopping patience = 30)
- Actual Training Stop: Epoch 68 (Chủ động dừng sau khi các chỉ số đánh giá có xu hướng ổn định / plateau)
- Offline Augmentation: Áp dụng balance_dataset.py chỉ trên tập Train
- Online Augmentation (Ultralytics): Mosaic (1.0), Mixup (0.15), Copy-Paste (0.3), Random Erasing (0.4)
- Close Mosaic: 15 Epochs cuối tắt Mosaic để tinh chỉnh viền Bounding Box
```

Quá trình huấn luyện chi tiết có thể theo dõi trong notebook: [`train_trash_yolo11_colab.ipynb`](train_trash_yolo11_colab.ipynb) hoặc file chạy mã nguồn [`train.py`](train.py).

---

## 5. Kết Quả Đánh Giá Mô Hình (Evaluation Metrics)

Kết quả đánh giá định lượng trên tập **Test Set độc lập (3.092 ảnh / 7.079 bounding boxes chưa từng xuất hiện lúc huấn luyện)**:

| Chỉ số đánh giá | Kết quả thực nghiệm | Ý nghĩa chuyên môn |
| :--- | :---: | :--- |
| **Precision** | **90.32%** | Tỷ lệ dự đoán đúng trên tổng số dự đoán dương tính |
| **Recall** | **80.62%** | Tỷ lệ phát hiện đối tượng trên tổng số đối tượng thực tế |
| **mAP @ 0.50** | **81.25%** | mean Average Precision tại ngưỡng IoU tiêu chuẩn 0.50 trên tập Test |
| **mAP @ 0.50:0.95** | **66.59%** | mean Average Precision trung bình trên dải ngưỡng IoU từ 0.50 đến 0.95 |
| **Inference Latency** | **~28 - 32 ms / frame** | Đạt khoảng **30 - 35 FPS** trong điều kiện kiểm thử thực nghiệm, phù hợp xử lý webcam tiệm cận thời gian thực |

*(Lưu ý: Trong quá trình khảo sát điều chỉnh siêu tham số trên tập Validation, chỉ số peak mAP@50 từng đạt mức 86.30% tại Epoch 28).*

### 🖼️ Minh chứng kết quả nhận diện thực tế:

Dưới đây là một ví dụ kết quả kiểm thử thực tế của mô hình (file minh chứng `test_output.jpg`):

![Minh chứng kết quả nhận diện rác](test_output.jpg)

---

## 6. Các Kịch Bản Kiểm Thử (Testing Scripts)

Dự án cung cấp các kịch bản kiểm thử độc lập phục vụ kiểm tra và đánh giá:

- **`test_webcam.py`**: Khởi động luồng camera qua OpenCV, kết nối mô hình PyTorch CUDA trên card đồ họa, hiển thị HUD thông số FPS và bounding box theo thời gian thực.
- **`test_image.py`**: Nhận diện ảnh đơn lẻ từ đường dẫn file, trích xuất tọa độ Bounding Box, Confidence Score và lưu ảnh kết quả.
- **`test_api_endpoints.py`**: Kiểm thử tự động tính sẵn sàng (Health Check) và độ chính xác của các API endpoints (`/predict_image`, `/predict_frame`).
- **`test.py`**: Script thực thi kiểm tra và đánh giá tổng quát mô hình.

---

## 7. Kiến Trúc Hệ Thống (System Architecture)

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
│  - Nạp best.onnx (9.4MB) vào RAM client│ │  - Quản lý lịch sử, upload   │
│  - Xử lý trực tiếp camera trên browser │ └──────────────┬───────────────┘
└────────────────────────────────────────┘                │ Internal Proxy
                                                          ▼
                                           ┌──────────────────────────────┐
                                           │    PYTHON AI MICROSERVICE    │
                                           │   Flask + PyTorch (CUDA)     │
                                           │  - Nạp best.pt (72.5MB)      │
                                           │  - Xử lý ảnh tĩnh tải lên    │
                                           └──────────────────────────────┘
```

---

## 8. Cấu Trúc Thư Mục Repository

```text
.
├── client/                          # Mã nguồn Frontend Vue.js 3
│   ├── dist/                        # Bản build tĩnh production
│   ├── public/                      # Tài nguyên tĩnh (WASM, Mô hình ONNX 9.4MB)
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
│   ├── uploads/                     # Thư mục lưu ảnh đã phân loại
│   ├── server.js                    # Web server chính chạy Node.js Express (Port 5000)
│   └── yolo_service.py              # Dịch vụ AI Python Flask kết nối YOLO model (Port 5001)
├── best.pt                          # Trọng số mô hình PyTorch (72.5 MB)
├── best.onnx                        # Mô hình ONNX INT8 cho web (9.4 MB)
├── data_balanced.yaml               # Cấu hình 7 nhãn và dataset
├── config.py                        # Cấu hình siêu tham số huấn luyện
├── balance_dataset.py               # Script cân bằng tỷ lệ mẫu dữ liệu
├── train.py                         # Script huấn luyện YOLOv11s trên Local
├── train_trash_yolo11_colab.ipynb   # Notebook huấn luyện trên Google Colab
├── test_webcam.py                   # Kiểm thử nhận diện webcam qua Python CUDA
├── test_image.py                    # Kiểm thử nhận diện trên ảnh tĩnh
├── test_api_endpoints.py            # Kiểm thử tự động các đầu API
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

## 9. Hướng Dẫn Cài Đặt & Chạy Localhost

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

## 10. Chạy Ứng Dụng Bằng Docker

Hệ thống hỗ trợ đóng gói và chạy thông qua Docker Container:

```bash
# Xây dựng và khởi chạy container bằng Docker Compose
docker compose up --build -d

# Xem log hoạt động
docker compose logs -f
```

Sau khi khởi động thành công, truy cập ứng dụng tại cổng được cấu hình trong `docker-compose.yml`: **`http://localhost:7860`**.

---

## 11. Các Script Kiểm Thử Nhanh

```bash
# 1. Kiểm thử trên webcam máy tính (yêu cầu webcam)
python test_webcam.py

# 2. Kiểm thử trên 1 tấm ảnh mẫu
python test_image.py

# 3. Kiểm tra tính sẵn sàng của các API Endpoint
python test_api_endpoints.py
```

---

## 12. Triển Khai Hệ Thống (Cloud Deployment)

Hệ thống được triển khai thử nghiệm trên nền tảng đám mây **Render** thông qua Docker container:

- **Link Demo Trực Tuyến:** **[https://trash-ai-68ei.onrender.com](https://trash-ai-68ei.onrender.com)**
- **Môi trường Cloud:** Docker Container (Debian Linux, Python 3.11, Node.js 20, ONNX Runtime).

---

## 13. Hạn Chế Thực Tế & Hướng Phát Triển

### ⚠️ Hạn chế hiện tại của mô hình:

1. **Điều kiện ánh sáng:** Độ tin cậy nhận diện có thể giảm trong môi trường ánh sáng quá yếu hoặc bị ngược sáng mạnh.
2. **Vật thể biến dạng:** Rác bị dập nát nhiều, cháy xém hoặc bị che khuất phần lớn diện tích có thể dẫn đến việc phân loại chưa chính xác.
3. **Phạm vi phân loại:** Hiện tại mô hình chỉ giới hạn trong phạm vi 7 nhóm rác sinh hoạt phổ biến; các loại rác thải đặc thù (rác y tế, thiết bị điện tử cỡ lớn) chưa nằm trong tập nhãn huấn luyện.

### 🌟 Hướng phát triển tiếp theo:

- Tiếp tục thu thập thêm dữ liệu hình ảnh thực tế đa dạng tại các điểm thu gom rác để mở rộng tập dữ liệu.
- Bổ sung thêm các nhóm rác thải công nghệ (E-waste) và rác thải nguy hại y tế.
- Nghiên cứu tối ưu hóa mô hình để triển khai thử nghiệm trên các thiết bị phần cứng nhúng (như Raspberry Pi 5 hoặc NVIDIA Jetson Nano) nhằm ứng dụng vào mô hình thùng rác phân loại bán tự động.
