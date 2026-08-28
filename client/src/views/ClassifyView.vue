<template>
  <div class="container-xl py-4">
    <!-- Page Header & Tab Navigation -->
    <div class="d-flex flex-column flex-md-row align-items-md-center justify-content-between gap-3 mb-4 pb-2 border-bottom">
      <div>
        <h2 class="fw-bold text-dark mb-1">
          <i class="bi bi-camera-fill text-success me-2"></i>AI phân loại rác
        </h2>
        <p class="text-muted small mb-0">
          Nhận diện và phân loại rác từ hình ảnh hoặc webcam theo thời gian thực.
        </p>
      </div>

      <!-- Tab Buttons -->
      <div class="p-1 bg-light rounded-pill border d-inline-flex align-items-center">
        <button 
          class="btn btn-sm rounded-pill px-3 py-2 fw-semibold transition"
          :class="activeTab === 'image' ? 'btn-success text-white shadow-sm' : 'btn-light text-muted'"
          @click="activeTab = 'image'"
        >
          <i class="bi bi-image me-1"></i> Phân loại từ ảnh
        </button>
        <button 
          class="btn btn-sm rounded-pill px-3 py-2 fw-semibold transition ms-1"
          :class="activeTab === 'webcam' ? 'btn-success text-white shadow-sm' : 'btn-light text-muted'"
          @click="activeTab = 'webcam'"
        >
          <i class="bi bi-camera-video me-1"></i> Nhận diện bằng webcam
        </button>
      </div>
    </div>

    <!-- Tab 1: Image Classifier Component -->
    <div v-show="activeTab === 'image'">
      <ImageClassifier />
    </div>

    <!-- Tab 2: Webcam Classifier Component -->
    <div v-show="activeTab === 'webcam'">
      <WebcamClassifier />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import ImageClassifier from '../components/classify/ImageClassifier.vue';
import WebcamClassifier from '../components/classify/WebcamClassifier.vue';

const activeTab = ref('image');
</script>

<style scoped>
.transition {
  transition: all 0.2s ease;
}
</style>
