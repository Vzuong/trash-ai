<template>
  <div 
    v-if="item" 
    class="modal fade show d-block" 
    tabindex="-1" 
    style="background-color: rgba(15, 23, 42, 0.6); backdrop-filter: blur(4px);"
    @click.self="$emit('close')"
  >
    <div class="modal-dialog modal-dialog-centered modal-lg">
      <div class="modal-content border-0 rounded-4 shadow-lg overflow-hidden">
        <!-- Modal Header -->
        <div class="modal-header bg-light border-bottom px-4 py-3">
          <div class="d-flex align-items-center gap-2">
            <span class="badge bg-success-subtle text-success border border-success-subtle px-2 py-1">
              <i class="bi bi-tag-fill me-1"></i> ID: {{ item.id }}
            </span>
            <h5 class="modal-title fw-bold text-dark mb-0">Chi tiết phân loại</h5>
          </div>
          <button type="button" class="btn-close shadow-none" @click="$emit('close')" aria-label="Close"></button>
        </div>

        <!-- Modal Body -->
        <div class="modal-body p-4">
          <div class="row g-4">
            <!-- Left: Image Preview with BBox overlay if available -->
            <div class="col-md-6">
              <div class="rounded-3 overflow-hidden bg-dark text-center position-relative d-flex align-items-center justify-content-center" style="min-height: 280px; max-height: 380px;">
                <img 
                  v-if="item.imageUrl" 
                  :src="item.imageUrl" 
                  :alt="item.imageName"
                  class="img-fluid rounded-3"
                  style="max-height: 360px; object-fit: contain;"
                  @error="onImageError"
                />
                <div v-else class="text-white-50 p-4">
                  <i class="bi bi-image display-4 d-block mb-2"></i>
                  <span>Không có ảnh xem trước</span>
                </div>
              </div>
              <div class="text-muted small text-center mt-2">
                <i class="bi bi-file-earmark me-1"></i> {{ item.imageName }}
              </div>
            </div>

            <!-- Right: Detailed Classification Information -->
            <div class="col-md-6 d-flex flex-column justify-content-between">
              <div>
                <!-- Primary Classification Badge -->
                <div class="p-3 bg-light rounded-3 border mb-3">
                  <div class="d-flex align-items-center justify-content-between mb-1">
                    <span class="badge bg-white text-dark border px-2 py-1 small">
                      {{ item.category || 'Rác tái chế' }}
                    </span>
                    <span class="display-6 fw-bold text-success">{{ item.confidencePercent }}%</span>
                  </div>
                  <h4 class="fw-bold mb-0 text-dark">{{ item.primaryClass }}</h4>
                </div>

                <!-- Metadata Grid -->
                <div class="row g-2 mb-3">
                  <div class="col-6">
                    <div class="p-2 bg-light rounded-2 border">
                      <span class="text-muted small d-block">Phương thức</span>
                      <strong class="text-dark small">
                        <i :class="['bi', item.method === 'webcam' ? 'bi-camera-video' : 'bi-upload', 'me-1']"></i>
                        {{ item.methodName }}
                      </strong>
                    </div>
                  </div>
                  <div class="col-6">
                    <div class="p-2 bg-light rounded-2 border">
                      <span class="text-muted small d-block">Thời gian suy luận</span>
                      <strong class="text-dark small">{{ item.inferenceTime }} ms</strong>
                    </div>
                  </div>
                  <div class="col-6">
                    <div class="p-2 bg-light rounded-2 border">
                      <span class="text-muted small d-block">Số đối tượng</span>
                      <strong class="text-dark small">{{ item.totalObjects || 1 }} vật thể</strong>
                    </div>
                  </div>
                  <div class="col-6">
                    <div class="p-2 bg-light rounded-2 border">
                      <span class="text-muted small d-block">Thời gian thực hiện</span>
                      <strong class="text-dark small">{{ formatTime(item.createdAt) }}</strong>
                    </div>
                  </div>
                </div>

                <!-- BBoxes List if present -->
                <div v-if="item.detections && item.detections.length > 0">
                  <h6 class="small fw-bold text-muted text-uppercase mb-2">Các vùng nhận diện:</h6>
                  <div class="d-flex flex-column gap-1 max-h-40 overflow-y-auto">
                    <div 
                      v-for="(det, i) in item.detections" 
                      :key="i"
                      class="p-2 bg-light border rounded-2 small d-flex align-items-center justify-content-between"
                    >
                      <span class="fw-medium">{{ i + 1 }}. {{ det.className || det.classCode }}</span>
                      <span class="badge bg-success-subtle text-success border border-success-subtle">
                        {{ Math.round((det.confidence || 0.9) * 100) }}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Footer Actions -->
              <div class="d-flex justify-content-end gap-2 mt-4 pt-3 border-top">
                <button type="button" class="btn btn-outline-danger btn-sm" @click="$emit('delete', item.id)">
                  <i class="bi bi-trash me-1"></i> Xóa bản ghi này
                </button>
                <button type="button" class="btn btn-secondary btn-sm" @click="$emit('close')">
                  Đóng
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  item: { type: Object, default: null }
});

defineEmits(['close', 'delete']);

function formatTime(isoStr) {
  if (!isoStr) return '--:--';
  const date = new Date(isoStr);
  return date.toLocaleString('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function onImageError(e) {
  e.target.style.display = 'none';
}
</script>

<style scoped>
.max-h-40 {
  max-height: 120px;
}
</style>
