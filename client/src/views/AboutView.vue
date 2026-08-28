<template>
  <div class="container-xl py-4">
    <!-- Header -->
    <div class="mb-4 pb-2 border-bottom">
      <h2 class="fw-bold text-dark mb-1">
        <i class="bi bi-info-circle-fill text-success me-2"></i>Giới Thiệu Hệ Thống
      </h2>
      <p class="text-muted small mb-0">
        Mục tiêu phát triển, giải pháp công nghệ và ý nghĩa môi trường của Hệ thống AI Phân Loại Rác.
      </p>
    </div>

    <div class="row g-4 mb-4">
      <!-- Left: Project Objectives & Scope -->
      <div class="col-lg-7">
        <div class="eco-card p-4 h-100 d-flex flex-column">
          <div class="d-flex align-items-center gap-2 mb-3 text-success">
            <i class="bi bi-bullseye fs-4"></i>
            <h5 class="fw-bold mb-0 text-dark">Mục Tiêu & Ý Nghĩa</h5>
          </div>

          <p class="text-muted" style="line-height: 1.7;">
            Ô nhiễm rác thải và việc phân loại rác thủ công tại nguồn đang là thách thức lớn đối với môi trường. Đề tài <strong>"Hệ thống AI Phân Loại Rác"</strong> ứng dụng các tiến bộ mới nhất của <strong>Thị giác máy tính (Computer Vision)</strong> nhằm tự động hóa quy trình nhận diện, phân loại rác thải nhanh chóng và chính xác.
          </p>

          <div class="d-flex flex-column gap-3 my-2">
            <div class="d-flex align-items-start gap-3 p-3 bg-light rounded-3 border">
              <div class="rounded-circle p-2 bg-success text-white flex-shrink-0">
                <i class="bi bi-image"></i>
              </div>
              <div>
                <h6 class="fw-bold text-dark mb-1">Nhận diện chính xác từ hình ảnh</h6>
                <p class="small text-muted mb-0">Hỗ trợ người dùng kéo thả tải ảnh từ máy tính, tự động xác định vị trí bounding box và phân loại 7 nhóm rác chính.</p>
              </div>
            </div>

            <div class="d-flex align-items-start gap-3 p-3 bg-light rounded-3 border">
              <div class="rounded-circle p-2 bg-primary text-white flex-shrink-0">
                <i class="bi bi-camera-video"></i>
              </div>
              <div>
                <h6 class="fw-bold text-dark mb-1">Giám sát Real-time qua Webcam</h6>
                <p class="small text-muted mb-0">Xử lý luồng video trực tiếp 30 FPS với độ trễ thấp (~32ms), phát hiện đồng thời nhiều vật thể và hướng dẫn bỏ đúng thùng rác.</p>
              </div>
            </div>

            <div class="d-flex align-items-start gap-3 p-3 bg-light rounded-3 border">
              <div class="rounded-circle p-2 bg-warning text-dark flex-shrink-0">
                <i class="bi bi-bar-chart"></i>
              </div>
              <div>
                <h6 class="fw-bold text-dark mb-1">Thống kê & Quản lý lịch sử</h6>
                <p class="small text-muted mb-0">Lưu trữ dữ liệu phân loại, vẽ biểu đồ phân bố rác thải giúp nâng cao ý thức phân loại rác tại nguồn.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Right: Student Project Info Card -->
      <div class="col-lg-5">
        <div class="eco-card p-4 h-100 d-flex flex-column justify-content-between">
          <div>
            <div class="d-flex align-items-center gap-2 mb-3 text-success">
              <i class="bi bi-person-badge-fill fs-4"></i>
              <h5 class="fw-bold mb-0 text-dark">Thông Tin Phát Triển</h5>
            </div>

            <div class="table-responsive">
              <table class="table table-bordered align-middle mb-3">
                <tbody>
                  <tr>
                    <td class="bg-light text-muted small fw-semibold" style="width: 38%;">Hệ thống</td>
                    <td class="fw-bold text-success">Hệ thống AI Phân Loại Rác</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted small fw-semibold">Phiên bản</td>
                    <td class="text-dark">v1.0.0 (YOLO11s Core)</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted small fw-semibold">Lĩnh vực</td>
                    <td class="text-dark">AI Computer Vision & Môi trường</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted small fw-semibold">Tác giả / Nhóm</td>
                    <td class="fw-bold text-dark">Nhóm Nghiên cứu & Phát triển AI</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted small fw-semibold">Khả năng triển khai</td>
                    <td class="text-dark">Web App, Camera thông minh, Thùng rác AI</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Environmental Quote Box -->
          <div class="p-3 bg-success-subtle rounded-3 border border-success-subtle text-center">
            <i class="bi bi-quote fs-3 text-success d-block mb-1"></i>
            <p class="small text-success-emphasis mb-0 fst-italic">
              "Mỗi hành động phân loại rác đúng cách hôm nay là một đóng góp thiết thực cho hành tinh xanh ngày mai."
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- Tech Stack Grid -->
    <div class="eco-card p-4 mb-4">
      <div class="d-flex align-items-center gap-2 mb-3 text-success">
        <i class="bi bi-stack fs-4"></i>
        <h5 class="fw-bold mb-0 text-dark">Công Nghệ Sử Dụng (Tech Stack)</h5>
      </div>

      <div class="row g-3">
        <div class="col-6 col-md-4 col-lg-2">
          <div class="p-3 bg-light rounded-3 text-center border h-100">
            <div class="display-6 mb-2">👁️</div>
            <h6 class="fw-bold mb-1">YOLO11</h6>
            <span class="text-muted small">Computer Vision</span>
          </div>
        </div>
        <div class="col-6 col-md-4 col-lg-2">
          <div class="p-3 bg-light rounded-3 text-center border h-100">
            <div class="display-6 mb-2">🟢</div>
            <h6 class="fw-bold mb-1">Vue 3 + Vite</h6>
            <span class="text-muted small">Frontend Framework</span>
          </div>
        </div>
        <div class="col-6 col-md-4 col-lg-2">
          <div class="p-3 bg-light rounded-3 text-center border h-100">
            <div class="display-6 mb-2">🌿</div>
            <h6 class="fw-bold mb-1">Node.js</h6>
            <span class="text-muted small">Runtime Engine</span>
          </div>
        </div>
        <div class="col-6 col-md-4 col-lg-2">
          <div class="p-3 bg-light rounded-3 text-center border h-100">
            <div class="display-6 mb-2">🚀</div>
            <h6 class="fw-bold mb-1">Express.js</h6>
            <span class="text-muted small">RESTful API Server</span>
          </div>
        </div>
        <div class="col-6 col-md-4 col-lg-2">
          <div class="p-3 bg-light rounded-3 text-center border h-100">
            <div class="display-6 mb-2">🎨</div>
            <h6 class="fw-bold mb-1">Bootstrap 5</h6>
            <span class="text-muted small">Responsive UI</span>
          </div>
        </div>
        <div class="col-6 col-md-4 col-lg-2">
          <div class="p-3 bg-light rounded-3 text-center border h-100">
            <div class="display-6 mb-2">🗄️</div>
            <h6 class="fw-bold mb-1">MySQL / JSON</h6>
            <span class="text-muted small">Data Persistence</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
</script>

<style scoped>
</style>
