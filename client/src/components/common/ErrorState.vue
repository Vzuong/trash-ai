<template>
  <div class="alert alert-danger border-danger-subtle d-flex align-items-center gap-3 p-3 rounded-3" role="alert">
    <i class="bi bi-exclamation-triangle-fill fs-3 text-danger flex-shrink-0"></i>
    <div class="flex-grow-1">
      <h6 class="fw-bold mb-1">{{ title }}</h6>
      <p class="mb-0 small text-danger-emphasis">{{ message }}</p>
    </div>
    <button v-if="retryable" @click="$emit('retry')" class="btn btn-outline-danger btn-sm text-nowrap">
      <i class="bi bi-arrow-clockwise me-1"></i> Thử lại
    </button>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, default: 'Đã xảy ra lỗi' },
  message: { type: String, default: 'Không thể kết nối tới máy chủ hoặc dữ liệu không hợp lệ.' },
  retryable: { type: Boolean, default: true }
});

defineEmits(['retry']);
</script>
