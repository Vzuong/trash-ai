<template>
  <div class="d-flex flex-column align-items-center justify-content-center py-5">
    <div class="spinner-border text-success mb-3" style="width: 2.5rem; height: 2.5rem;" role="status">
      <span class="visually-hidden">Đang xử lý...</span>
    </div>
    <p class="text-muted mb-0 fw-medium">{{ message }}</p>
  </div>
</template>

<script setup>
defineProps({
  message: { type: String, default: 'Đang tải dữ liệu, vui lòng đợi trong giây lát...' }
});
</script>
