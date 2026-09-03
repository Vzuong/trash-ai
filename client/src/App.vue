<template>
  <div class="app-layout d-flex flex-column min-vh-100">
    <Navbar />
    <main class="flex-grow-1">
      <router-view />
    </main>
    <Footer />
  </div>
</template>

<script setup>
import { onMounted } from 'vue';
import Navbar from './components/layout/Navbar.vue';
import Footer from './components/layout/Footer.vue';
import yoloWebEngine from './services/yoloWebEngine';

// Tải ngầm mô hình AI 38MB ngay khi người dùng vừa mở web
onMounted(() => {
  yoloWebEngine.loadModel('/models/best.onnx')
    .then(() => console.log('✅ [App] Mô hình AI đã tải sẵn sàng trong nền!'))
    .catch((err) => console.warn('[App] Tải trước mô hình:', err.message));
});
</script>

<style>
.app-layout {
  min-height: 100vh;
}
</style>
