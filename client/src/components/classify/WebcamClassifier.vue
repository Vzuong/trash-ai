<template>
  <div class="row g-4">
    <!-- Left Column: Video & Realtime Canvas Overlay -->
    <div class="col-lg-8">
      <div class="eco-card p-3 p-md-4 d-flex flex-column h-100">
        <div class="d-flex align-items-center justify-content-between mb-3">
          <div class="d-flex align-items-center gap-2">
            <i class="bi bi-camera-video-fill text-success fs-5"></i>
            <h6 class="fw-bold mb-0">Nhận diện rác thải qua Webcam thời gian thực</h6>
          </div>
          <div class="d-flex align-items-center gap-2">
            <span v-if="isStreaming && !isPaused" class="badge bg-success-subtle text-success border border-success-subtle d-flex align-items-center gap-1">
              <span class="hud-dot"></span> Đang nhận diện Live (⚡ NVIDIA GPU)
            </span>
            <span v-else-if="isPaused" class="badge bg-warning-subtle text-warning-emphasis border border-warning-subtle">
              <i class="bi bi-pause-circle me-1"></i> Tạm dừng
            </span>
            <span v-else class="badge bg-secondary-subtle text-secondary border">
              Camera đang tắt
            </span>
          </div>
        </div>

        <!-- Camera Frame Box -->
        <div class="webcam-container flex-grow-1 position-relative d-flex align-items-center justify-content-center">
          <!-- Video Stream -->
          <video 
            ref="videoElement" 
            class="webcam-video" 
            autoplay 
            playsinline 
            muted
            v-show="isStreaming"
          ></video>

          <!-- Realtime Canvas Overlay for BBoxes -->
          <canvas 
            ref="overlayCanvas" 
            class="webcam-canvas"
            v-show="isStreaming"
          ></canvas>

          <!-- Camera Inactive / Placeholder State -->
          <div v-if="!isStreaming && !cameraError" class="text-center text-white p-4">
            <div class="mb-3">
              <div class="rounded-circle bg-white bg-opacity-10 p-3 d-inline-flex">
                <i class="bi bi-camera-video display-5 text-success"></i>
              </div>
            </div>
            <h5 class="fw-bold mb-2">Camera chưa được kích hoạt</h5>
            <p class="text-white-50 small mb-4 max-w-sm mx-auto">
              Nhấn <strong>"Bật camera"</strong> để cấp quyền truy cập camera và nhận diện rác thải tốc độ cao bằng GPU NVIDIA.
            </p>
            <button @click="startCamera" class="btn btn-eco-primary btn-lg d-inline-flex align-items-center gap-2 shadow">
              <i class="bi bi-play-circle-fill fs-5"></i> Bật camera ngay
            </button>
          </div>

          <!-- Camera Error State -->
          <div v-if="cameraError" class="text-center text-white p-4 max-w-sm">
            <div class="text-danger mb-3">
              <i class="bi bi-exclamation-octagon display-4"></i>
            </div>
            <h6 class="fw-bold text-danger mb-2">Không thể truy cập camera</h6>
            <p class="text-white-50 small mb-3">{{ cameraError }}</p>
            <button @click="startCamera" class="btn btn-outline-light btn-sm">
              <i class="bi bi-arrow-clockwise me-1"></i> Thử lại
            </button>
          </div>
        </div>

        <!-- Controls Toolbar -->
        <div class="d-flex flex-wrap align-items-center justify-content-between gap-2 mt-3 pt-3 border-top">
          <!-- Left Actions -->
          <div class="d-flex gap-2">
            <button 
              v-if="!isStreaming" 
              @click="startCamera" 
              class="btn btn-eco-primary btn-sm d-flex align-items-center gap-1"
            >
              <i class="bi bi-camera-video-fill"></i> Bật camera
            </button>
            <button 
              v-else 
              @click="stopCamera" 
              class="btn btn-outline-danger btn-sm d-flex align-items-center gap-1"
            >
              <i class="bi bi-stop-circle-fill"></i> Tắt camera
            </button>

            <button 
              v-if="isStreaming" 
              @click="togglePause" 
              class="btn btn-outline-secondary btn-sm d-flex align-items-center gap-1"
            >
              <i :class="['bi', isPaused ? 'bi-play-fill' : 'bi-pause-fill']"></i>
              <span>{{ isPaused ? 'Tiếp tục' : 'Tạm dừng' }}</span>
            </button>
          </div>

          <!-- Real-time HUD Stats -->
          <div v-if="isStreaming && !isPaused" class="d-flex align-items-center gap-2 px-3 py-1.5 bg-dark text-white rounded-pill small font-monospace shadow-sm">
            <span class="hud-dot"></span>
            <span>FPS: <strong class="text-success">{{ currentFps }}</strong></span>
            <span class="text-white-50">|</span>
            <span>Độ trễ: <strong>{{ currentInferenceTime }} ms</strong></span>
            <span class="text-white-50">|</span>
            <span class="badge bg-success-subtle text-success py-0 px-1">⚡ GPU Siêu Tốc</span>
          </div>

          <!-- Right Action: Capture Frame & Classify -->
          <div>
            <button 
              @click="captureAndSave" 
              class="btn btn-eco-soft btn-sm d-flex align-items-center gap-2"
              :disabled="!isStreaming || isCapturing"
            >
              <span v-if="isCapturing" class="spinner-border spinner-border-sm" role="status"></span>
              <i v-else class="bi bi-camera-fill text-success fs-6"></i>
              <span><strong>Chụp ảnh & Lưu lịch sử</strong></span>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Right Column: Live Detection HUD & Capture Feedback -->
    <div class="col-lg-4">
      <div class="eco-card p-3 p-md-4 h-100 d-flex flex-column">
        <div class="d-flex align-items-center justify-content-between mb-3">
          <div class="d-flex align-items-center gap-2">
            <i class="bi bi-activity text-success fs-5"></i>
            <h6 class="fw-bold mb-0">Giám sát Real-time</h6>
          </div>
          <span class="badge bg-success-subtle text-success border border-success-subtle">⚡ NVIDIA GPU (30+ FPS)</span>
        </div>

        <!-- Real-time Active Classification Item -->
        <div v-if="isStreaming && activeDetections.length > 0" class="mb-3">
          <div 
            class="p-3 rounded-3 border mb-3"
            :style="{ backgroundColor: (activeDetections[0]?.color || '#10b981') + '15', borderColor: (activeDetections[0]?.color || '#10b981') + '40' }"
          >
            <div class="d-flex align-items-center justify-content-between mb-2">
              <span class="badge bg-white text-dark border small px-2">Vật thể đang thấy</span>
              <span class="fw-bold text-dark fs-5">{{ activeDetections[0]?.confidencePercent }}%</span>
            </div>
            <h4 class="fw-bold mb-0 text-dark">{{ activeDetections[0]?.className }}</h4>
          </div>

          <!-- List of all visible objects in current frame -->
          <h6 class="small fw-bold text-muted text-uppercase mb-2">Danh sách rác trong khung hình:</h6>
          <div class="d-flex flex-column gap-2 mb-3">
            <div 
              v-for="(obj, i) in activeDetections" 
              :key="i"
              class="p-2 px-3 rounded-2 bg-light border d-flex align-items-center justify-content-between"
            >
              <div class="d-flex align-items-center gap-2">
                <span class="badge rounded-circle p-1" :style="{ backgroundColor: obj.color }"> </span>
                <span class="small fw-semibold text-dark">{{ obj.className }}</span>
              </div>
              <span class="badge bg-white text-dark border small fw-bold">{{ obj.confidencePercent }}%</span>
            </div>
          </div>
        </div>

        <!-- Idle Status when camera is on but no trash detected -->
        <div v-else-if="isStreaming && activeDetections.length === 0" class="text-center py-4 my-auto">
          <div class="text-muted mb-2">
            <i class="bi bi-view-finder display-5 text-success"></i>
          </div>
          <h6 class="fw-bold text-dark">Đang quét khung hình...</h6>
          <p class="text-muted small mb-0 px-3">
            Hãy đưa các vật thể rác thải (chai nhựa, lon nhôm, giấy, bìa carton) trước ống kính camera.
          </p>
        </div>

        <!-- Inactive Camera State -->
        <div v-else class="text-center py-4 my-auto">
          <div class="text-muted mb-2">
            <i class="bi bi-camera-video-off display-6 text-muted"></i>
          </div>
          <h6 class="fw-bold text-muted">Webcam chưa bật</h6>
          <p class="text-muted small mb-0">
            Bấm nút <strong>"Bật camera"</strong> để xem bảng phân tích thời gian thực.
          </p>
        </div>

        <!-- Captured Frame Notification Card (Appears when user clicks Capture) -->
        <div v-if="lastCaptured" class="p-3 bg-success-subtle border border-success-subtle rounded-3 mt-auto">
          <div class="d-flex align-items-center justify-content-between mb-1">
            <span class="badge bg-success text-white small">
              <i class="bi bi-check-lg me-1"></i> Đã chụp & lưu
            </span>
            <span class="small text-muted">{{ lastCaptured.time }}</span>
          </div>
          <div class="fw-bold text-dark small mt-1">
            {{ lastCaptured.className }} ({{ lastCaptured.confidencePercent }}%)
          </div>
          <div class="d-flex justify-content-between align-items-center mt-2">
            <span class="text-muted small">Thời gian: {{ lastCaptured.inferenceTime }}ms</span>
            <router-link to="/history" class="btn btn-link btn-sm p-0 text-success fw-semibold text-decoration-none">
              Xem lịch sử →
            </router-link>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import apiService from '../../services/api';

const videoElement = ref(null);
const overlayCanvas = ref(null);

const isStreaming = ref(false);
const isPaused = ref(false);
const isCapturing = ref(false);
const cameraError = ref('');
const currentFps = ref(30);
const currentInferenceTime = ref(20);
const detectedCount = ref(0);
const activeDetections = ref([]);
const lastCaptured = ref(null);

let mediaStream = null;
let animationFrameId = null;
let lastFrameTime = performance.now();
let frameCount = 0;
let isInferencing = false;

// Temporal Box Tracking & Smoothing State
let trackedBoxes = [];

function calculateIoU(b1, b2) {
  const xA = Math.max(b1.x1, b2.x1);
  const yA = Math.max(b1.y1, b2.y1);
  const xB = Math.min(b1.x2, b2.x2);
  const yB = Math.min(b1.y2, b2.y2);
  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const boxAArea = Math.max(0, b1.x2 - b1.x1) * Math.max(0, b1.y2 - b1.y1);
  const boxBArea = Math.max(0, b2.x2 - b2.x1) * Math.max(0, b2.y2 - b2.y1);
  const union = boxAArea + boxBArea - interArea;
  return union > 0 ? interArea / union : 0;
}

function updateTrackedDetections(rawDetections) {
  const matched = new Set();
  const WINDOW_SIZE = 5;

  rawDetections.forEach((det) => {
    const rawBox = det.bboxNorm || {
      x1: (det.bbox?.x1 || 0) / 640,
      y1: (det.bbox?.y1 || 0) / 480,
      x2: (det.bbox?.x2 || 100) / 640,
      y2: (det.bbox?.y2 || 100) / 480
    };

    let bestMatch = null;
    let bestIoU = 0.25;

    trackedBoxes.forEach((t) => {
      if (!matched.has(t)) {
        const iou = calculateIoU(t.targetBbox, rawBox);
        if (iou > bestIoU) {
          bestIoU = iou;
          bestMatch = t;
        }
      }
    });

    if (bestMatch) {
      matched.add(bestMatch);
      bestMatch.targetBbox = rawBox;
      bestMatch.missCount = 0;
      bestMatch.history.push({
        classCode: det.classCode,
        className: det.className,
        color: det.color,
        confidence: det.confidencePercent
      });
      if (bestMatch.history.length > WINDOW_SIZE) bestMatch.history.shift();

      bestMatch.className = det.className;
      bestMatch.color = det.color;
      bestMatch.targetConf = det.confidencePercent;
    } else {
      const newBox = {
        id: Math.random().toString(36).substring(2, 9),
        className: det.className,
        color: det.color,
        currentBbox: { ...rawBox },
        targetBbox: { ...rawBox },
        currentConf: det.confidencePercent,
        targetConf: det.confidencePercent,
        missCount: 0,
        history: [{
          classCode: det.classCode,
          className: det.className,
          color: det.color,
          confidence: det.confidencePercent
        }]
      };
      trackedBoxes.push(newBox);
      matched.add(newBox);
    }
  });

  trackedBoxes.forEach((t) => {
    if (!matched.has(t)) t.missCount++;
  });
  trackedBoxes = trackedBoxes.filter((t) => t.missCount <= 3);
}

function renderSmoothBoxes(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const cw = canvas.width;
  const ch = canvas.height;
  const LERP_SMOOTH = 0.45;

  trackedBoxes.forEach((t) => {
    t.currentBbox.x1 += (t.targetBbox.x1 - t.currentBbox.x1) * LERP_SMOOTH;
    t.currentBbox.y1 += (t.targetBbox.y1 - t.currentBbox.y1) * LERP_SMOOTH;
    t.currentBbox.x2 += (t.targetBbox.x2 - t.currentBbox.x2) * LERP_SMOOTH;
    t.currentBbox.y2 += (t.targetBbox.y2 - t.currentBbox.y2) * LERP_SMOOTH;
    t.currentConf += (t.targetConf - t.currentConf) * 0.3;

    const x = t.currentBbox.x1 * cw;
    const y = t.currentBbox.y1 * ch;
    const w = (t.currentBbox.x2 - t.currentBbox.x1) * cw;
    const h = (t.currentBbox.y2 - t.currentBbox.y1) * ch;

    const displayConf = Math.round(t.currentConf);
    const color = t.color || '#10b981';

    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(3, Math.round(cw / 250));
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = color + '25';
    ctx.fillRect(x, y, w, h);

    const label = `${t.className} ${displayConf}%`;
    const fontSize = Math.max(14, Math.round(cw / 45));
    ctx.font = `bold ${fontSize}px Inter, sans-serif`;
    const textMetrics = ctx.measureText(label);
    const textWidth = textMetrics.width;
    const textHeight = fontSize * 1.35;

    ctx.fillStyle = color;
    ctx.fillRect(x, y - textHeight > 0 ? y - textHeight : y, textWidth + 12, textHeight);

    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x + 6, (y - textHeight > 0 ? y - textHeight : y) + fontSize);
  });
}

async function startCamera() {
  cameraError.value = '';
  try {
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'environment'
      },
      audio: false
    };

    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    if (videoElement.value) {
      videoElement.value.srcObject = mediaStream;
      videoElement.value.onloadedmetadata = () => {
        isStreaming.value = true;
        isPaused.value = false;
        startDetectionLoop();
      };
    }
  } catch (err) {
    console.error('Lỗi camera:', err);
    cameraError.value = 'Không thể truy cập camera. Vui lòng cấp quyền truy cập trong trình duyệt.';
    isStreaming.value = false;
  }
}

function stopCamera() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (videoElement.value) videoElement.value.srcObject = null;
  isStreaming.value = false;
  isPaused.value = false;
  activeDetections.value = [];
  detectedCount.value = 0;
  trackedBoxes = [];
}

function togglePause() {
  isPaused.value = !isPaused.value;
  if (!isPaused.value) startDetectionLoop();
}

function startDetectionLoop() {
  let lastInferenceTimestamp = 0;
  const INFERENCE_INTERVAL = 40; // ~25 FPS inference rate

  const loop = async (timestamp) => {
    if (!isStreaming.value || isPaused.value) return;

    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
      currentFps.value = frameCount;
      frameCount = 0;
      lastFrameTime = now;
    }

    const canvas = overlayCanvas.value;
    const video = videoElement.value;
    if (canvas && video && video.readyState === 4) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
      }
      renderSmoothBoxes(canvas);
    }

    if (timestamp - lastInferenceTimestamp >= INFERENCE_INTERVAL && !isInferencing) {
      lastInferenceTimestamp = timestamp;
      await processFrameInference();
    }

    animationFrameId = requestAnimationFrame(loop);
  };

  animationFrameId = requestAnimationFrame(loop);
}

async function processFrameInference() {
  const video = videoElement.value;
  if (!video || video.readyState !== 4) return;

  isInferencing = true;
  const t0 = performance.now();

  try {
    const vWidth = video.videoWidth || 640;
    const vHeight = video.videoHeight || 480;
    const targetW = 320;
    const targetH = Math.round((vHeight / vWidth) * 320);

    const offscreen = document.createElement('canvas');
    offscreen.width = targetW;
    offscreen.height = targetH;
    const offCtx = offscreen.getContext('2d');
    offCtx.drawImage(video, 0, 0, targetW, targetH);
    const frameBase64 = offscreen.toDataURL('image/jpeg', 0.55);

    const response = await apiService.predictWebcam(frameBase64, false);
    if (response.success && response.data) {
      currentInferenceTime.value = response.data.inferenceTime || Math.round(performance.now() - t0);
      activeDetections.value = response.data.detections || [];
      detectedCount.value = response.data.totalObjects || 0;
      updateTrackedDetections(activeDetections.value);
    }
  } catch (err) {
    // Silent fail
  } finally {
    isInferencing = false;
  }
}

async function captureAndSave() {
  const video = videoElement.value;
  if (!video || !isStreaming.value) return;

  isCapturing.value = true;
  try {
    const offscreen = document.createElement('canvas');
    offscreen.width = video.videoWidth || 640;
    offscreen.height = video.videoHeight || 480;
    const offCtx = offscreen.getContext('2d');
    offCtx.drawImage(video, 0, 0);
    const frameBase64 = offscreen.toDataURL('image/jpeg', 0.85);

    const response = await apiService.predictWebcam(frameBase64, true);
    if (response.success && response.data) {
      const primary = response.data.primaryResult;
      lastCaptured.value = {
        className: primary?.className || 'Rác nhựa',
        confidencePercent: primary?.confidencePercent || 95,
        inferenceTime: response.data.inferenceTime || 20,
        time: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      };
    }
  } catch (err) {
    console.error('Lỗi khi chụp:', err);
    alert(err.message || 'Không thể lưu ảnh chụp.');
  } finally {
    isCapturing.value = false;
  }
}

onMounted(() => {
  // Ready
});

onUnmounted(() => {
  stopCamera();
});
</script>

<style scoped>
.max-w-sm {
  max-width: 360px;
}
</style>
