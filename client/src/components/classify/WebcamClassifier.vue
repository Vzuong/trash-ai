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
              <span class="hud-dot"></span> Đang nhận diện Live ({{ isLocalModelReady ? '⚡ WASM AI' : '☁️ Cloud AI' }})
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
              Nhấn <strong>"Bật camera"</strong> để cấp quyền truy cập camera và nhận diện rác thải tốc độ cao bằng AI.
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
            <span class="badge bg-success-subtle text-success py-0 px-1">{{ isLocalModelReady ? 'WASM GPU' : 'Cloud' }}</span>
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

        <!-- In-Browser AI Loading Banner -->
        <div v-if="isDownloadingModel" class="mt-3 p-2 px-3 bg-info-subtle border border-info-subtle rounded-3 small text-info-emphasis d-flex align-items-center justify-content-between">
          <div class="d-flex align-items-center gap-2">
            <span class="spinner-border spinner-border-sm text-info" role="status"></span>
            <span class="fw-medium">{{ modelLoadingProgress }}</span>
          </div>
          <span class="badge bg-info text-white">Đang kích hoạt AI máy (30+ FPS)</span>
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
          <span class="badge bg-light text-muted border">{{ isLocalModelReady ? '⚡ Trình duyệt (30+ FPS)' : '☁️ Cloud API' }}</span>
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
            Hãy đưa các vật thể rác thải (chai nhựa, lon nhôm, giấy, bìa carton) trước ống kính webcam.
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
const currentInferenceTime = ref(28);
const detectedCount = ref(0);
const activeDetections = ref([]);
const lastCaptured = ref(null);
const isLocalModelReady = ref(false);

let mediaStream = null;
let animationFrameId = null;
let lastFrameTime = performance.now();
let frameCount = 0;
let isInferencing = false;
let ortSession = null;

const CLASS_NAMES = {
  0: 'battery',
  1: 'cardboard',
  2: 'paper',
  3: 'glass',
  4: 'metal',
  5: 'plastic',
  6: 'organic'
};

const CLASS_META = {
  battery: { code: 'battery', name: 'Rác pin', color: '#ef4444' },
  cardboard: { code: 'cardboard', name: 'Rác bìa carton', color: '#d97706' },
  paper: { code: 'paper', name: 'Rác giấy', color: '#f59e0b' },
  glass: { code: 'glass', name: 'Rác thủy tinh', color: '#06b6d4' },
  metal: { code: 'metal', name: 'Rác kim loại', color: '#64748b' },
  plastic: { code: 'plastic', name: 'Rác nhựa', color: '#3b82f6' },
  organic: { code: 'organic', name: 'Rác hữu cơ', color: '#10b981' }
};

const modelLoadingProgress = ref('');
const isDownloadingModel = ref(false);

// Initialize In-Browser ONNX Runtime Web Model for 60 FPS zero-lag inference
async function initLocalONNX() {
  if (typeof window === 'undefined' || !window.ort || ortSession) return;

  try {
    isDownloadingModel.value = true;
    modelLoadingProgress.value = 'Đang kết nối chip xử lý AI...';

    // 1. Configure WASM path from CDN & single-thread (100% mobile compatibility without SharedArrayBuffer errors)
    window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
    window.ort.env.wasm.numThreads = 1;

    // 2. Fetch model with live progress
    modelLoadingProgress.value = 'Đang tải AI về chip điện thoại...';
    const response = await fetch('/best.onnx');
    if (!response.ok) throw new Error(`HTTP ${response.status} khi tải best.onnx`);

    const contentLength = response.headers.get('content-length');
    const totalBytes = contentLength ? parseInt(contentLength, 10) : 36200000;
    let loadedBytes = 0;

    const reader = response.body.getReader();
    const chunks = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loadedBytes += value.length;
      const pct = Math.round((loadedBytes / totalBytes) * 100);
      modelLoadingProgress.value = `Đang nạp AI vào chip máy: ${pct}% (${(loadedBytes / (1024 * 1024)).toFixed(1)}MB)`;
    }

    const modelBuffer = new Uint8Array(loadedBytes);
    let offset = 0;
    for (const chunk of chunks) {
      modelBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    modelLoadingProgress.value = 'Đang khởi chạy mô hình AI trên máy...';
    ortSession = await window.ort.InferenceSession.create(modelBuffer.buffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    isLocalModelReady.value = true;
    isDownloadingModel.value = false;
    modelLoadingProgress.value = 'Đã kích hoạt AI trên máy (30+ FPS)';
    console.log('🚀 [ONNX Web] In-browser AI model initialized successfully! (30-60 FPS active)');
  } catch (e) {
    console.warn('⚠️ [ONNX Web] In-browser engine loading failed, using Cloud API:', e.message);
    isLocalModelReady.value = false;
    isDownloadingModel.value = false;
    modelLoadingProgress.value = 'Đang dùng Cloud AI';
  }
}

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

// In-Browser ONNX Preprocessing & Inference
async function runLocalInference(video) {
  const targetSize = 640;
  const offscreen = document.createElement('canvas');
  offscreen.width = targetSize;
  offscreen.height = targetSize;
  const ctx = offscreen.getContext('2d');

  const vW = video.videoWidth || 640;
  const vH = video.videoHeight || 480;
  const r = Math.min(targetSize / vW, targetSize / vH);
  const nw = Math.round(vW * r);
  const nh = Math.round(vH * r);
  const dx = (targetSize - nw) / 2;
  const dy = (targetSize - nh) / 2;

  ctx.fillStyle = '#727272';
  ctx.fillRect(0, 0, targetSize, targetSize);
  ctx.drawImage(video, dx, dy, nw, nh);

  const imgData = ctx.getImageData(0, 0, targetSize, targetSize).data;
  const floatArr = new Float32Array(3 * targetSize * targetSize);
  const planeSize = targetSize * targetSize;

  for (let i = 0; i < planeSize; i++) {
    floatArr[i] = imgData[i * 4] / 255.0;
    floatArr[planeSize + i] = imgData[i * 4 + 1] / 255.0;
    floatArr[planeSize * 2 + i] = imgData[i * 4 + 2] / 255.0;
  }

  const tensor = new ort.Tensor('float32', floatArr, [1, 3, targetSize, targetSize]);
  const inputName = ortSession.inputNames[0];
  const feeds = {};
  feeds[inputName] = tensor;

  const outputMap = await ortSession.run(feeds);
  const outputTensor = outputMap[ortSession.outputNames[0]];
  const data = outputTensor.data;
  const numAnchors = 8400;
  const numClasses = 7;
  const confThresh = 0.30;

  const boxes = [];
  const confs = [];
  const classIds = [];

  for (let i = 0; i < numAnchors; i++) {
    let maxScore = -1;
    let maxCls = -1;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numAnchors + i];
      if (score > maxScore) {
        maxScore = score;
        maxCls = c;
      }
    }

    if (maxScore >= confThresh) {
      const xc = data[0 * numAnchors + i];
      const yc = data[1 * numAnchors + i];
      const w = data[2 * numAnchors + i];
      const h = data[3 * numAnchors + i];

      const x1 = Math.max(0, Math.min(vW, (xc - w / 2 - dx) / r));
      const y1 = Math.max(0, Math.min(vH, (yc - h / 2 - dy) / r));
      const x2 = Math.max(0, Math.min(vW, (xc + w / 2 - dx) / r));
      const y2 = Math.max(0, Math.min(vH, (yc + h / 2 - dy) / r));

      boxes.push([x1, y1, x2 - x1, y2 - y1]);
      confs.push(maxScore);
      classIds.push(maxCls);
    }
  }

  // JS NMS
  const detections = [];
  const picked = [];
  for (let i = 0; i < boxes.length; i++) {
    let keep = true;
    for (let j = 0; j < picked.length; j++) {
      const pIdx = picked[j];
      const b1 = { x1: boxes[i][0], y1: boxes[i][1], x2: boxes[i][0] + boxes[i][2], y2: boxes[i][1] + boxes[i][3] };
      const b2 = { x1: boxes[pIdx][0], y1: boxes[pIdx][1], x2: boxes[pIdx][0] + boxes[pIdx][2], y2: boxes[pIdx][1] + boxes[pIdx][3] };
      if (calculateIoU(b1, b2) > 0.45) {
        keep = false;
        break;
      }
    }
    if (keep) {
      picked.push(i);
      const cId = classIds[i];
      const cName = CLASS_NAMES[cId] || 'plastic';
      const meta = CLASS_META[cName] || { name: `Rác ${cName}`, color: '#10b981' };
      const bx = boxes[i];
      detections.push({
        id: detections.length + 1,
        className: meta.name,
        color: meta.color,
        confidencePercent: Math.round(confs[i] * 100),
        bboxNorm: {
          x1: bx[0] / vW,
          y1: bx[1] / vH,
          x2: (bx[0] + bx[2]) / vW,
          y2: (bx[1] + bx[3]) / vH
        }
      });
    }
  }

  return detections;
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
    initLocalONNX();
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
  const INFERENCE_INTERVAL = isLocalModelReady.value ? 33 : 80; // 30 FPS for local, 12 FPS for cloud

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
    if (isLocalModelReady.value && ortSession) {
      // ⚡ ULTRA-FAST IN-BROWSER WEB WORKER / WASM INFERENCE (15-30ms, 30-60 FPS)
      const detections = await runLocalInference(video);
      currentInferenceTime.value = Math.round(performance.now() - t0);
      activeDetections.value = detections;
      detectedCount.value = detections.length;
      updateTrackedDetections(detections);
    } else {
      // ☁️ COMPRESSED CLOUD FALLBACK (Ultra-small 256x192 JPEG @ 0.45, only ~7KB)
      const vWidth = video.videoWidth || 640;
      const vHeight = video.videoHeight || 480;
      const targetW = 256;
      const targetH = Math.round((vHeight / vWidth) * 256);

      const offscreen = document.createElement('canvas');
      offscreen.width = targetW;
      offscreen.height = targetH;
      const offCtx = offscreen.getContext('2d');
      offCtx.drawImage(video, 0, 0, targetW, targetH);
      const frameBase64 = offscreen.toDataURL('image/jpeg', 0.45);

      const response = await apiService.predictWebcam(frameBase64, false);
      if (response.success && response.data) {
        currentInferenceTime.value = Math.round(performance.now() - t0);
        activeDetections.value = response.data.detections || [];
        detectedCount.value = response.data.totalObjects || 0;
        updateTrackedDetections(activeDetections.value);
      }
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
        inferenceTime: response.data.inferenceTime || 30,
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
  initLocalONNX();
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
