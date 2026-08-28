<template>
  <div class="text-center py-5 px-3">
    <div class="empty-icon-wrapper mx-auto mb-3 d-flex align-items-center justify-content-center">
      <i :class="['bi', icon, 'text-muted fs-1']"></i>
    </div>
    <h5 class="fw-bold text-dark mb-2">{{ title }}</h5>
    <p class="text-muted small mb-3 max-w-sm mx-auto">{{ message }}</p>
    <slot name="action"></slot>
  </div>
</template>

<script setup>
defineProps({
  icon: { type: String, default: 'bi-inbox' },
  title: { type: String, default: 'Chưa có dữ liệu' },
  message: { type: String, default: 'Dữ liệu hiện tại đang trống hoặc chưa được cập nhật.' }
});
</script>

<style scoped>
.empty-icon-wrapper {
  width: 72px;
  height: 72px;
  background-color: #f1f5f9;
  border-radius: 50%;
}

.max-w-sm {
  max-width: 380px;
}
</style>
