<template>
  <footer class="bg-white border-top py-4 mt-auto">
    <div class="container-xl">
      <div class="row align-items-center gy-3">
        <div class="col-md-6 text-center text-md-start">
          <p class="mb-0 text-muted small">
            © 2026 <strong>Hệ thống AI Phân Loại Rác</strong>. Phát triển vì một môi trường xanh - sạch - đẹp.
          </p>
        </div>

        <div class="col-md-6 text-center text-md-end">
          <div class="d-flex align-items-center justify-content-center justify-content-md-end gap-3 small text-muted">
            <router-link to="/about" class="text-decoration-none text-muted hover-green">
              <i class="bi bi-info-circle me-1"></i> Giới thiệu hệ thống
            </router-link>
            <span>•</span>
            <router-link to="/model" class="text-decoration-none text-muted hover-green">
              <i class="bi bi-cpu me-1"></i> Thông số mô hình
            </router-link>
            <span>•</span>
            <router-link to="/history" class="text-decoration-none text-muted hover-green">
              <i class="bi bi-clock-history me-1"></i> Lịch sử nhận diện
            </router-link>
          </div>
        </div>
      </div>
    </div>
  </footer>
</template>

<style scoped>
.hover-green:hover {
  color: var(--eco-primary) !important;
}
</style>
