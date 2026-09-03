<template>
  <div class="webcam-classifier">
    <!-- AI Backend Status Banner (Nếu cần) -->
    <div v-if="modelStatus === 'error'" class="alert alert-danger d-flex align-items-center mb-3 shadow-sm border-0">
      <i class="bi bi-exclamation-triangle-fill fs-4 me-3"></i>
      <div>
        <strong>Không thể kết nối dịch vụ AI GPU:</strong>
        <div class="small">{{ loadErrorMessage || 'Vui lòng kiểm tra server backend và tiến trình AI.' }}</div>
        <button class="btn btn-sm btn-outline-danger mt-2" @click="retryLoadModel">
          <i class="bi bi-arrow-clockwise me-1"></i> Thử kết nối lại
        </button>
      </div>
    </div>

    <div class="row g-4">
      <!-- Camera Preview Column -->
      <div class="col-lg-8">
        <div class="card border-0 shadow-sm overflow-hidden position-relative bg-dark rounded-4">
          <!-- Video Frame & Overlay Canvas Container -->
          <div class="ratio ratio-4x3 position-relative video-container">
            <video
              ref="videoElement"
              autoplay
              playsinline
              muted
              class="w-100 h-100 object-fit-cover"
            ></video>

            <!-- Bounding Box Canvas Overlay -->
            <canvas
              ref="overlayCanvas"
              class="position-absolute top-0 start-0 w-100 h-100 pointer-events-none"
            ></canvas>

            <!-- Video Inactive Placeholder -->
            <div
              v-if="!isStreaming"
              class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column align-items-center justify-content-center text-white bg-dark bg-opacity-75"
            >
              <i class="bi bi-camera-video display-1 text-muted mb-3"></i>
              <h5 class="fw-bold">Camera chưa bật</h5>
              <p class="text-white-50 small mb-3 text-center px-4">
                Nhấn <strong>"Bật camera"</strong> để quét và phân loại rác thải tự động trực tiếp trên thiết bị của bạn.
              </p>
              <button
                class="btn btn-success btn-lg px-4 rounded-pill shadow"
                :disabled="modelStatus !== 'ready'"
                @click="startCamera"
              >
                <i class="bi bi-play-fill me-1"></i> Bật camera
              </button>
            </div>
          </div>

          <!-- Bottom Realtime Stats Bar -->
          <div class="card-footer bg-dark border-top border-secondary py-2 px-3 d-flex flex-wrap align-items-center justify-content-between text-white small">
            <div class="d-flex align-items-center gap-3">
              <span class="d-flex align-items-center gap-1">
                <span class="badge rounded-circle p-1" :class="isStreaming ? 'bg-success' : 'bg-secondary'"> </span>
                <span class="fw-semibold">{{ isStreaming ? (isPaused ? 'Tạm dừng' : 'Đang nhận diện') : 'Chờ bật camera' }}</span>
              </span>
              <span v-if="isStreaming" class="text-white-50">|</span>
              <span v-if="isStreaming" class="text-white-50">
                FPS: <strong class="text-white">{{ currentFps }}</strong>
              </span>
              <span v-if="isStreaming" class="text-white-50">|</span>
              <span v-if="isStreaming" class="text-white-50">
                Độ trễ: <strong class="text-success">{{ currentInferenceTime }} ms</strong>
              </span>
            </div>

            <div class="d-flex align-items-center gap-2">
              <span v-if="detectedCount > 0" class="badge bg-primary">
                {{ detectedCount }} vật thể
              </span>
            </div>
          </div>
        </div>

        <!-- Camera Controls -->
        <div class="d-flex flex-wrap align-items-center justify-content-between gap-2 mt-3">
          <div class="d-flex gap-2">
            <button
              v-if="!isStreaming"
              class="btn btn-success px-4"
              :disabled="modelStatus !== 'ready'"
              @click="startCamera"
            >
              <i class="bi bi-camera-video me-1"></i> Bật camera
            </button>
            <template v-else>
              <button class="btn btn-outline-danger px-3" @click="stopCamera">
                <i class="bi bi-stop-fill me-1"></i> Tắt camera
              </button>
              <button class="btn btn-outline-secondary px-3" @click="togglePause">
                <i :class="isPaused ? 'bi bi-play-fill' : 'bi bi-pause-fill'" class="me-1"></i>
                {{ isPaused ? 'Tiếp tục' : 'Tạm dừng' }}
              </button>
              <button
                v-if="hasMultipleCameras"
                class="btn btn-outline-dark px-3"
                @click="switchCamera"
                title="Đổi camera trước/sau"
              >
                <i class="bi bi-arrow-repeat me-1"></i> Đổi camera
              </button>
            </template>
          </div>

          <button
            v-if="isStreaming"
            class="btn btn-primary px-4 shadow-sm"
            :disabled="isCapturing"
            @click="captureAndSave"
          >
            <span v-if="isCapturing" class="spinner-border spinner-border-sm me-1"></span>
            <i v-else class="bi bi-camera me-1"></i>
            Chụp ảnh & Lưu lịch sử
          </button>
        </div>
      </div>

      <!-- Real-Time Classification Results Column -->
      <div class="col-lg-4">
        <div class="card border-0 shadow-sm rounded-4 h-100 p-4 d-flex flex-column">
          <h5 class="fw-bold text-dark mb-3 d-flex align-items-center gap-2">
            <i class="bi bi-stars text-success"></i>
            Kết quả phân loại
          </h5>

          <!-- When Trash Objects Detected -->
          <div v-if="activeDetections.length > 0" class="d-flex flex-column gap-3">
            <!-- Primary Detection Card -->
            <div class="p-3 rounded-3 border" :style="{ backgroundColor: activeDetections[0].color + '15', borderColor: activeDetections[0].color + '40' }">
              <div class="d-flex align-items-center justify-content-between mb-2">
                <span class="badge text-white small" :style="{ backgroundColor: activeDetections[0].color }">
                  <i :class="activeDetections[0].icon" class="me-1"></i> {{ activeDetections[0].category }}
                </span>
                <span class="fs-5 fw-bold" :style="{ color: activeDetections[0].color }">
                  {{ activeDetections[0].confidencePercent }}%
                </span>
              </div>
              <h4 class="fw-bold mb-1" :style="{ color: activeDetections[0].color }">
                {{ activeDetections[0].className }}
              </h4>
              <p class="small text-dark mb-2">
                <strong>Thùng rác:</strong> {{ activeDetections[0].binColor }}
              </p>
              <div class="small text-muted bg-white p-2 rounded-2 border">
                <i class="bi bi-info-circle text-primary me-1"></i>
                {{ activeDetections[0].instruction }}
              </div>
            </div>

            <!-- List of other detected objects -->
            <div v-if="activeDetections.length > 1">
              <h6 class="small fw-bold text-muted text-uppercase mb-2">Các vật thể khác trong khung hình:</h6>
              <div class="d-flex flex-column gap-2">
                <div
                  v-for="(obj, i) in activeDetections.slice(1)"
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
          </div>

          <!-- When Camera Active But No Trash Detected -->
          <div v-else-if="isStreaming" class="text-center py-5 my-auto">
            <div class="text-muted mb-3">
              <i class="bi bi-view-finder display-4 text-success opacity-75"></i>
            </div>
            <h6 class="fw-bold text-dark">Đang quét khung hình...</h6>
            <p class="text-muted small mb-0 px-3">
              Hãy đưa các vật thể rác thải (chai nhựa, lon nhôm, giấy, bìa carton) trước ống kính camera.
            </p>
          </div>

          <!-- When Camera Inactive -->
          <div v-else class="text-center py-5 my-auto">
            <div class="text-muted mb-3">
              <i class="bi bi-camera-video-off display-4 text-muted"></i>
            </div>
            <h6 class="fw-bold text-muted">Webcam chưa bật</h6>
            <p class="text-muted small mb-0 px-3">
              Nhấn nút <strong>"Bật camera"</strong> để xem kết quả phân loại thời gian thực trực tiếp trên thiết bị.
            </p>
          </div>

          <!-- Last Captured Snapshot Card -->
          <div v-if="lastCaptured" class="p-3 bg-success-subtle border border-success-subtle rounded-3 mt-auto pt-3">
            <div class="d-flex align-items-center justify-content-between mb-1">
              <span class="badge bg-success text-white small">
                <i class="bi bi-check-lg me-1"></i> Đã lưu lịch sử
              </span>
              <span class="small text-muted">{{ lastCaptured.time }}</span>
            </div>
            <div class="fw-bold text-dark small mt-1">
              {{ lastCaptured.className }} ({{ lastCaptured.confidencePercent }}%)
            </div>
            <div class="d-flex justify-content-between align-items-center mt-2">
              <span class="text-muted small">Thời gian xử lý: {{ lastCaptured.inferenceTime }}ms</span>
              <router-link to="/history" class="btn btn-link btn-sm p-0 text-success fw-semibold text-decoration-none">
                Xem lịch sử →
              </router-link>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import yoloWebEngine, { CONF_THRESHOLD, IOU_THRESHOLD } from '../../services/yoloWebEngine';
import transformBBoxToVideoOverlay from '../../utils/coordinateTransform';
import apiService from '../../services/api';

const videoElement = ref(null);
const overlayCanvas = ref(null);

// Synchronously detect hostname:
// - Localhost / Local IP -> 'server_gpu' (NVIDIA GPU on presentation PC)
// - Render / Public Internet -> ALWAYS 'client_webgpu' (Run on viewer device GPU, never send to Render CPU)
const isRunningLocally = typeof window !== 'undefined' && 
  (window.location.hostname === 'localhost' || 
   window.location.hostname === '127.0.0.1' || 
   window.location.hostname.includes('192.168.'));

const engineMode = ref(isRunningLocally ? 'server_gpu' : 'client_webgpu');
const clientModelLoading = ref(false);
const clientBackend = ref('WebGPU');

const modelStatus = ref('ready'); // 'ready' | 'error'
const modelBackend = ref(isRunningLocally ? 'NVIDIA GPU (Local)' : 'WebGPU (Thiết bị của bạn)');
const loadErrorMessage = ref('');
let serverHasGPUState = false;

const isStreaming = ref(false);
const isPaused = ref(false);
const isCapturing = ref(false);
const hasMultipleCameras = ref(false);
const currentFacingMode = ref('environment');

const currentFps = ref(0);
const currentInferenceTime = ref(0);
const detectedCount = ref(0);
const activeDetections = ref([]);
const lastCaptured = ref(null);

let mediaStream = null;
let animationFrameId = null;
let lastFpsTimestamp = performance.now();
let frameCount = 0;
let isInferencing = false;
let resizeObserver = null;

// Reusable offscreen canvas for high-speed lightweight frame capture (max 640px)
let captureCanvas = null;
let captureCtx = null;

async function setEngineMode(mode) {
  if (mode === 'server_gpu' && !isRunningLocally && !serverHasGPUState) {
    const ok = confirm('⚠️ Chú ý: Máy chủ đám mây Render là gói miễn phí KHÔNG CÓ GPU (chỉ có CPU yếu).\n\nNếu bạn chọn chế độ này, frame camera phải bay sang Mỹ xử lý trên CPU nên độ trễ sẽ bị chậm ~2 giây.\n\nBạn có muốn giữ chế độ "WebGPU Trình Duyệt" để dùng GPU của chính máy bạn không?');
    if (ok) return;
  }

  engineMode.value = mode;
  if (mode === 'client_webgpu') {
    if (yoloWebEngine.status !== 'ready') {
      clientModelLoading.value = true;
      try {
        await yoloWebEngine.loadModel('/models/best.onnx');
        clientBackend.value = yoloWebEngine.activeProvider || 'WebGPU';
      } catch (e) {
        console.warn('Lỗi kích hoạt WebGPU Client:', e);
      } finally {
        clientModelLoading.value = false;
      }
    } else {
      clientBackend.value = yoloWebEngine.activeProvider || 'WebGPU';
    }
  }
}

async function initModel() {
  try {
    const healthRes = await apiService.checkHealth();
    if (healthRes && healthRes.ai) {
      const dev = (healthRes.ai.device || healthRes.ai.backend || '').toLowerCase();
      if (dev.includes('cuda') || dev.includes('gpu') || dev.includes('nvidia') || dev.includes('1650') || dev.includes('rtx')) {
        serverHasGPUState = true;
      }
      if (isRunningLocally || serverHasGPUState) {
        modelBackend.value = healthRes.ai.device || 'NVIDIA GPU (Local)';
      } else {
        modelBackend.value = 'Render Cloud (CPU)';
      }
    }
  } catch (err) {
    console.warn('[WebcamClassifier] Kiểm tra AI Health ban đầu:', err);
  }

  // Tự động nhận diện môi trường:
  // - Nếu chạy trên localhost hoặc server có GPU NVIDIA -> ưu tiên Server GPU cho thuyết trình siêu mượt (~20ms)
  // - Nếu chạy trên Render (cloud công khai không GPU) -> tự động kích hoạt WebGPU để người xem dùng GPU máy họ!
  if (isRunningLocally || serverHasGPUState) {
    engineMode.value = 'server_gpu';
  } else {
    engineMode.value = 'client_webgpu';
    setEngineMode('client_webgpu');
  }

  modelStatus.value = 'ready';
}

function retryLoadModel() {
  initModel();
}

async function checkCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter((d) => d.kind === 'videoinput');
    hasMultipleCameras.value = videoInputs.length > 1;
  } catch (e) {
    hasMultipleCameras.value = false;
  }
}

async function startCamera() {

  try {
    const constraints = {
      video: {
        facingMode: { ideal: currentFacingMode.value },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      // Fallback to default video camera
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }

    if (videoElement.value) {
      videoElement.value.srcObject = mediaStream;
      videoElement.value.onloadedmetadata = () => {
        isStreaming.value = true;
        isPaused.value = false;
        syncCanvasSize();
        startDetectionLoop();
      };
    }
  } catch (err) {
    console.error('[WebcamClassifier] Lỗi truy cập camera:', err);
    alert('Không thể truy cập camera. Vui lòng cấp quyền truy cập máy ảnh trong trình duyệt.');
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
  if (videoElement.value) {
    videoElement.value.srcObject = null;
  }

  isStreaming.value = false;
  isPaused.value = false;
  isInferencing = false;
  activeDetections.value = [];
  detectedCount.value = 0;
  clearCanvas();
}

function togglePause() {
  isPaused.value = !isPaused.value;
  if (!isPaused.value) {
    startDetectionLoop();
  }
}

async function switchCamera() {
  currentFacingMode.value = currentFacingMode.value === 'environment' ? 'user' : 'environment';
  stopCamera();
  await startCamera();
}

function syncCanvasSize() {
  const video = videoElement.value;
  const canvas = overlayCanvas.value;
  if (!video || !canvas) return;

  const rect = video.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    canvas.width = Math.round(rect.width);
    canvas.height = Math.round(rect.height);
  }
}

function clearCanvas() {
  const canvas = overlayCanvas.value;
  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function startDetectionLoop() {
  const loop = async () => {
    if (!isStreaming.value || isPaused.value) return;

    // Smooth FPS Counter
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTimestamp >= 1000) {
      currentFps.value = frameCount;
      frameCount = 0;
      lastFpsTimestamp = now;
    }

    const video = videoElement.value;

    // Draw active bounding boxes continuously at video screen refresh rate
    renderOverlayBoxes(activeDetections.value);

    // Request-gated inference
    if (video && video.readyState >= 2 && !isInferencing) {
      isInferencing = true;
      const t0 = performance.now();

      try {
        let ranClient = false;
        if (yoloWebEngine && yoloWebEngine.status === 'ready') {
          try {
            const result = await yoloWebEngine.detect(video, CONF_THRESHOLD, IOU_THRESHOLD);
            currentInferenceTime.value = result.inferenceTime;
            activeDetections.value = result.detections || [];
            detectedCount.value = result.totalObjects || 0;
            clientBackend.value = result.backend || clientBackend.value;
            ranClient = true;
          } catch (e) {
            console.warn('[WebcamClassifier] WebGPU frame error, fallback server:', e.message);
          }
        }

        if (!ranClient) {
          // Gửi frame sang server (/api/predict/webcam) - đảm bảo luôn khoanh rác 100% như localhost
          const maxDim = 640;
          const vW = video.videoWidth || 640;
          const vH = video.videoHeight || 480;
          const scale = Math.min(maxDim / vW, maxDim / vH, 1.0);
          const capW = Math.round(vW * scale);
          const capH = Math.round(vH * scale);

          if (!captureCanvas) {
            captureCanvas = document.createElement('canvas');
          }
          if (captureCanvas.width !== capW || captureCanvas.height !== capH) {
            captureCanvas.width = capW;
            captureCanvas.height = capH;
            captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });
          }

          // Render vào buffer 640px
          captureCtx.drawImage(video, 0, 0, capW, capH);
          // Nén JPEG nhẹ ~25KB (encode trong ~3ms)
          const frameBase64 = captureCanvas.toDataURL('image/jpeg', 0.65);

          const res = await apiService.predictWebcam(frameBase64, false);
          const data = res?.data || res;

          if (data && data.success !== false) {
            currentInferenceTime.value = data.inferenceTime || Math.round(performance.now() - t0);
            activeDetections.value = data.detections || [];
            detectedCount.value = data.totalObjects || 0;
            if (data.model) {
              modelBackend.value = data.model;
            }
          }
        }
      } catch (err) {
        console.warn('[WebcamClassifier] Lỗi frame inference:', err);
      } finally {
        isInferencing = false;
      }
    }

    animationFrameId = requestAnimationFrame(loop);
  };

  animationFrameId = requestAnimationFrame(loop);
}

function renderOverlayBoxes(detections) {
  const canvas = overlayCanvas.value;
  const video = videoElement.value;
  if (!canvas || !video) return;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!detections || detections.length === 0) return;

  detections.forEach((det) => {
    const coords = transformBBoxToVideoOverlay(det.bboxNorm, video, canvas);
    if (!coords.visible) return;

    const { x, y, w, h } = coords;
    const color = det.color || '#10b981';

    // Draw box outline
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(3, Math.round(canvas.width / 220));
    ctx.strokeRect(x, y, w, h);

    // Semi-transparent fill
    ctx.fillStyle = color + '22';
    ctx.fillRect(x, y, w, h);

    // Label tag
    const label = `${det.className} ${det.confidencePercent}%`;
    const fontSize = Math.max(13, Math.round(canvas.width / 42));
    ctx.font = `bold ${fontSize}px Inter, sans-serif`;
    const textMetrics = ctx.measureText(label);
    const textWidth = textMetrics.width;
    const textHeight = fontSize * 1.35;

    ctx.fillStyle = color;
    const labelY = y - textHeight > 0 ? y - textHeight : y;
    ctx.fillRect(x, labelY, textWidth + 12, textHeight);

    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x + 6, labelY + fontSize);
  });
}

async function captureAndSave() {
  const video = videoElement.value;
  if (!video || !isStreaming.value) return;

  isCapturing.value = true;
  try {
    const fullCanvas = document.createElement('canvas');
    fullCanvas.width = video.videoWidth || 640;
    fullCanvas.height = video.videoHeight || 480;
    const fullCtx = fullCanvas.getContext('2d');
    fullCtx.drawImage(video, 0, 0);

    const frameBase64 = fullCanvas.toDataURL('image/jpeg', 0.85);

    if (engineMode.value === 'server_gpu') {
      const response = await apiService.predictWebcam(frameBase64, true);
      const data = response?.data || response;

      if (data) {
        const primary = data.primaryResult;
        lastCaptured.value = {
          className: primary ? primary.className : 'Không phát hiện rác',
          confidencePercent: primary ? primary.confidencePercent : 0,
          inferenceTime: data.inferenceTime || 0,
          time: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
        };
      }
    } else {
      // Trong chế độ WebGPU client: nhận diện trực tiếp và lưu vào lịch sử
      const result = await yoloWebEngine.detect(fullCanvas, CONF_THRESHOLD, IOU_THRESHOLD);
      const primary = result.primaryResult;

      await apiService.saveWebcamHistory({
        image: frameBase64,
        method: 'webcam',
        primaryResult: primary,
        totalObjects: result.totalObjects,
        inferenceTime: result.inferenceTime,
        detections: result.detections
      });

      lastCaptured.value = {
        className: primary ? primary.className : 'Không phát hiện rác',
        confidencePercent: primary ? primary.confidencePercent : 0,
        inferenceTime: result.inferenceTime,
        time: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      };
    }
  } catch (err) {
    console.error('[WebcamClassifier] Lỗi khi chụp & lưu ảnh:', err);
    alert('Không thể lưu ảnh lịch sử: ' + (err.message || 'Lỗi kết nối'));
  } finally {
    isCapturing.value = false;
  }
}

onMounted(async () => {
  await initModel();
  await checkCameras();

  if (videoElement.value) {
    resizeObserver = new ResizeObserver(() => {
      syncCanvasSize();
      renderOverlayBoxes(activeDetections.value);
    });
    resizeObserver.observe(videoElement.value);
  }
});

onUnmounted(() => {
  stopCamera();
  if (resizeObserver) {
    resizeObserver.disconnect();
  }
});
</script>

<style scoped>
.video-container {
  background-color: #111827;
  border-radius: 1rem;
  overflow: hidden;
}

.pointer-events-none {
  pointer-events: none;
}

.transition-all {
  transition: all 0.2s ease-in-out;
}
</style>
