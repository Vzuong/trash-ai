# 🚀 HƯỚNG DẪN ĐƯA HỆ THỐNG AI LÊN WEB ONLINE (DOCKER)

Hệ thống đã được cấu hình trọn bộ **3 trong 1** (Vue 3 + Node.js Backend + Python YOLO11 AI Service) chỉ trong **1 Docker Container duy nhất**.

---

## 🌟 CÁCH 1: Triển khai lên Hugging Face Spaces (MIỄN PHÍ 16GB RAM - KHUYÊN DÙNG ⭐⭐⭐)

Hugging Face là nền tảng số 1 thế giới cho các ứng dụng AI, miễn phí 16 GB RAM + 2 vCPU vĩnh viễn.

### Bước 1: Tạo Space trên Hugging Face
1. Truy cập: [https://huggingface.co/spaces](https://huggingface.co/spaces) (Đăng ký tài khoản miễn phí nếu chưa có).
2. Bấm nút **Create new Space**.
3. Điền các thông tin:
   * **Space name:** `trash-ai-classifier` (hoặc tên tùy thích)
   * **License:** `mit`
   * **Space SDK:** Chọn **Docker** (biểu tượng cá voi xanh 🐳)
   * **Docker template:** Chọn **Blank**
   * **Space Hardware:** Chọn **CPU Basic (Free - 2 vCPU, 16GB RAM)**
4. Bấm **Create Space**.

### Bước 2: Đẩy dự án lên Hugging Face Space
Hugging Face sẽ cung cấp cho bạn 1 đường dẫn Git của Space (Ví dụ: `https://huggingface.co/spaces/YOUR_USERNAME/trash-ai-classifier`).

Mở Terminal tại thư mục `d:\SIC\Trash` và chạy các lệnh sau:

```bash
# 1. Thêm remote của Hugging Face
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/trash-ai-classifier.git

# 2. Thêm các file cần thiết
git add Dockerfile .dockerignore start.sh best.pt client server package.json

# 3. Commit
git commit -m "Deploy Trash AI Fullstack App via Docker"

# 4. Đẩy lên Space
git push space main --force
```

*(Hoặc bạn có thể kéo thả trực tiếp các thư mục `client`, `server`, file `best.pt`, `Dockerfile`, `start.sh` lên giao diện web của Hugging Face)*.

### Bước 3: Thưởng thức thành quả!
* Hugging Face sẽ tự động kích hoạt Docker build trong ~3 phút.
* Sau khi hoàn tất, hệ thống sẽ cấp cho bạn một đường link web chính thức dạng:
  👉 **`https://YOUR_USERNAME-trash-ai-classifier.hf.space`**
* **Bất kỳ ai (kể cả trên điện thoại iPhone/Android hay máy tính)** bấm vào link đều mở camera quét rác, xem thống kê và lịch sử mượt mà 24/7!

---

## 📦 CÁCH 2: Triển khai lên Render.com

1. Đẩy mã nguồn dự án lên tài khoản GitHub của bạn (Bao gồm `Dockerfile`, `start.sh`, `best.pt`, `client/`, `server/`).
2. Truy cập [https://render.com](https://render.com) $\rightarrow$ Đăng nhập bằng GitHub.
3. Bấm **New +** $\rightarrow$ Chọn **Web Service**.
4. Chọn repository GitHub của dự án bạn vừa đẩy lên.
5. Ở mục **Environment / Runtime**, Render sẽ tự động nhận diện là **Docker**.
6. Chọn gói **Free** $\rightarrow$ Bấm **Create Web Service**.
7. Render sẽ tự động build và cấp cho bạn tên miền:
  👉 **`https://ten-du-an.onrender.com`**

---

## 🐳 CÁCH 3: Chạy thử Docker trên máy tính cục bộ (Nếu máy bạn có cài Docker Desktop)

```bash
# Build và khởi động bằng Docker Compose:
docker compose up --build

# Mở trình duyệt truy cập:
http://localhost:7860
```
