<template>
  <nav class="navbar navbar-expand-lg sticky-top bg-white border-bottom shadow-sm py-2">
    <div class="container-xl">
      <!-- Logo & Brand Name -->
      <router-link to="/" class="navbar-brand d-flex align-items-center gap-2 text-decoration-none">
        <div class="brand-icon-wrapper d-flex align-items-center justify-content-center">
          <i class="bi bi-recycle fs-4 text-white"></i>
        </div>
        <div class="d-flex flex-column">
          <span class="fw-bold brand-title text-success">Hệ thống AI Phân Loại Rác</span>
          <span class="brand-subtitle text-muted">Nhận diện và phân loại rác bằng trí tuệ nhân tạo</span>
        </div>
      </router-link>

      <!-- Mobile Toggle Button -->
      <button 
        class="navbar-toggler border-0 shadow-none" 
        type="button" 
        data-bs-toggle="collapse" 
        data-bs-target="#navbarMain"
        aria-controls="navbarMain" 
        aria-expanded="false" 
        aria-label="Toggle navigation"
      >
        <i class="bi bi-list fs-2 text-success"></i>
      </button>

      <!-- Navigation Links -->
      <div class="collapse navbar-collapse" id="navbarMain">
        <ul class="navbar-nav ms-auto mb-2 mb-lg-0 gap-1 align-items-lg-center">
          <li class="nav-item">
            <router-link to="/" class="nav-link nav-link-custom" active-class="active">
              <i class="bi bi-grid-1x2-fill"></i>
              <span>Dashboard</span>
            </router-link>
          </li>

          <li class="nav-item">
            <router-link to="/classify" class="nav-link nav-link-custom" active-class="active">
              <i class="bi bi-camera-fill"></i>
              <span>Phân loại rác</span>
            </router-link>
          </li>

          <li class="nav-item">
            <router-link to="/history" class="nav-link nav-link-custom" active-class="active">
              <i class="bi bi-clock-history"></i>
              <span>Lịch sử</span>
            </router-link>
          </li>

          <li class="nav-item">
            <router-link to="/model" class="nav-link nav-link-custom" active-class="active">
              <i class="bi bi-cpu-fill"></i>
              <span>Mô hình AI</span>
            </router-link>
          </li>

          <li class="nav-item">
            <router-link to="/about" class="nav-link nav-link-custom" active-class="active">
              <i class="bi bi-info-circle-fill"></i>
              <span>Giới thiệu</span>
            </router-link>
          </li>

          <!-- Direct CTA Button -->
          <li class="nav-item ms-lg-2 mt-2 mt-lg-0">
            <router-link to="/classify" class="btn btn-eco-primary btn-sm d-flex align-items-center gap-1 shadow-sm">
              <i class="bi bi-lightning-charge-fill"></i>
              <span>Phân loại ngay</span>
            </router-link>
          </li>
        </ul>
      </div>
    </div>
  </nav>
</template>

<script setup>
</script>

<style scoped>
.brand-icon-wrapper {
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #2e7d32 0%, #10b981 100%);
  border-radius: 10px;
  box-shadow: 0 4px 10px rgba(46, 125, 50, 0.25);
}

.brand-title {
  font-size: 1.15rem;
  letter-spacing: -0.01em;
  line-height: 1.2;
}

.brand-subtitle {
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.02em;
}
</style>
