# Hướng Dẫn Triển Khai Hệ Thống (Deployment Guide)

Hệ thống được đóng gói thông qua Docker (kết hợp Vue 3 Frontend, Node.js Backend và Python AI Service chạy trên ONNX Runtime).

---

## 1. Triển khai qua Docker trên máy cục bộ (Localhost)

Yêu cầu môi trường đã cài đặt Docker Desktop.

```bash
# Xây dựng image và khởi chạy container
docker compose up --build -d

# Xem log hoạt động
docker compose logs -f

# Truy cập ứng dụng
http://localhost:7860
```

---

## 2. Triển khai lên nền tảng Render

1. Đẩy toàn bộ mã nguồn lên repository GitHub.
2. Truy cập [Render Dashboard](https://dashboard.render.com).
3. Chọn **New** -> **Web Service** -> Chọn repository tương ứng.
4. Render sẽ tự động phát hiện `Dockerfile`. Chọn môi trường **Docker**, cấu hình instance phù hợp (gói Free hoặc Starter).
5. Nhấn **Create Web Service** để bắt đầu quá trình build và deploy.

---

## 3. Triển khai lên Hugging Face Spaces

1. Tạo một Space mới trên [Hugging Face Spaces](https://huggingface.co/spaces) với SDK là **Docker**.
2. Thiết lập remote Git và đẩy mã nguồn:

```bash
git remote add space https://huggingface.co/spaces/<USERNAME>/<SPACE_NAME>.git
git push space master --force
```

Sau khi hoàn tất quá trình build, ứng dụng sẽ phục vụ tại đường dẫn của Space.
