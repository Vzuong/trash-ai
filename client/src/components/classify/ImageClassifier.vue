<template>
  <div class="row g-4">
    <!-- Left Column: Upload & Canvas Viewer -->
    <div class="col-lg-7">
      <div class="eco-card h-100 p-3 p-md-4 d-flex flex-column">
        <div class="d-flex align-items-center justify-content-between mb-3">
          <div class="d-flex align-items-center gap-2">
            <i class="bi bi-image text-success fs-5"></i>
            <h6 class="fw-bold mb-0">Tải ảnh rác thải để phân loại</h6>
          </div>
          <button 
            v-if="selectedImage" 
            @click="resetImage" 
            class="btn btn-outline-secondary btn-sm d-flex align-items-center gap-1"
          >
            <i class="bi bi-trash"></i> Xóa ảnh
          </button>
        </div>

        <!-- Persistent Hidden File Input -->
        <input 
          type="file" 
          ref="fileInput" 
          class="d-none" 
          accept="image/jpeg,image/png,image/webp,image/jpg" 
          @change="handleFileSelect"
        />

        <!-- Dropzone (Shown when no image selected) -->
        <div 
          v-if="!selectedImage"
          class="eco-dropzone flex-grow-1 d-flex flex-column align-items-center justify-content-center"
          :class="{ 'drag-active': isDragging }"
          @dragover.prevent="isDragging = true"
          @dragleave.prevent="isDragging = false"
          @drop.prevent="handleDrop"
          @click="triggerFileInput"
        >
          <div class="mb-3">
            <i class="bi bi-cloud-arrow-up text-success display-4"></i>
          </div>
          <h6 class="fw-bold text-dark mb-1">Kéo & thả ảnh rác thải vào đây</h6>
          <p class="text-muted small mb-3">hoặc nhấp chuột để chọn file từ máy tính</p>
          <span class="badge bg-light text-secondary border px-3 py-2">
            Hỗ trợ: JPG, PNG, WEBP (Tối đa 15MB)
          </span>
        </div>

        <!-- Image & Bounding Box Canvas Container (Shown after image chosen) -->
        <div v-else class="position-relative flex-grow-1 d-flex align-items-center justify-content-center bg-dark rounded-3 overflow-hidden" style="min-height: 380px;">
          <!-- Loading Overlay -->
          <div v-if="isLoading" class="position-absolute w-100 h-100 d-flex flex-column align-items-center justify-content-center bg-dark bg-opacity-75 z-3">
            <div class="spinner-border text-success mb-2" role="status"></div>
            <span class="text-white small fw-medium">Hệ thống AI đang phân tích khung hình...</span>
          </div>

          <!-- Canvas where image + BBoxes are drawn -->
          <canvas ref="resultCanvas" class="img-fluid rounded-3" style="max-height: 480px;"></canvas>
        </div>

        <!-- Action Bar Below Canvas -->
        <div v-if="selectedImage" class="d-flex align-items-center justify-content-between mt-3 pt-3 border-top">
          <div class="small text-muted text-truncate me-2" style="max-width: 250px;">
            <i class="bi bi-file-earmark-image me-1"></i> {{ selectedFile?.name || 'Ảnh đã chọn' }}
          </div>
          <div class="d-flex gap-2">
            <button 
              @click="triggerFileInput" 
              class="btn btn-outline-secondary btn-sm"
              :disabled="isLoading"
            >
              Chọn ảnh khác
            </button>
            <button 
              @click="classifyImage" 
              class="btn btn-eco-primary btn-sm d-flex align-items-center gap-1"
              :disabled="isLoading"
            >
              <i class="bi bi-play-fill fs-5"></i>
              <span>{{ hasResult ? 'Phân loại lại' : 'Bắt đầu phân loại' }}</span>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Right Column: Detection Results Panel -->
    <div class="col-lg-5">
      <div class="eco-card h-100 p-3 p-md-4">
        <div class="d-flex align-items-center justify-content-between mb-3">
          <div class="d-flex align-items-center gap-2">
            <i class="bi bi-clipboard2-check text-success fs-5"></i>
            <h6 class="fw-bold mb-0">Kết quả phân loại</h6>
          </div>
          <span v-if="result" class="badge bg-success-subtle text-success border border-success-subtle">
            <i class="bi bi-check-circle-fill me-1"></i> Hoàn thành
          </span>
        </div>

        <!-- Initial Placeholder State -->
        <div v-if="!result && !isLoading" class="text-center py-5">
          <div class="mb-3 text-muted">
            <i class="bi bi-search display-5"></i>
          </div>
          <h6 class="fw-bold text-dark">Chưa có kết quả</h6>
          <p class="text-muted small mb-0 px-3">
            Vui lòng chọn hoặc kéo thả một bức ảnh rác thải (chai nhựa, lon nước, giấy, thủy tinh...) và bấm <strong>"Bắt đầu phân loại"</strong>.
          </p>
        </div>

        <!-- Loading State in Side Panel -->
        <div v-else-if="isLoading" class="py-5 text-center">
          <div class="spinner-grow text-success mb-3" role="status"></div>
          <h6 class="fw-bold">Mô hình AI đang suy luận...</h6>
          <p class="text-muted small">Đang trích xuất đặc trưng và xác định bounding box.</p>
        </div>

        <!-- Results Content -->
        <div v-else-if="result" class="d-flex flex-column gap-3">
          <!-- Primary Waste Classification Card -->
          <div 
            class="p-3 rounded-3 border d-flex align-items-center justify-content-between"
            :style="{ backgroundColor: (result.primaryResult?.color || '#10b981') + '15', borderColor: (result.primaryResult?.color || '#10b981') + '40' }"
          >
            <div class="d-flex align-items-center gap-3">
              <div 
                class="rounded-circle p-2 d-flex align-items-center justify-content-center text-white shadow-sm"
                :style="{ backgroundColor: result.primaryResult?.color || '#10b981', width: '48px', height: '48px' }"
              >
                <i :class="['bi', result.primaryResult?.icon || 'bi-recycle', 'fs-4']"></i>
              </div>
              <div>
                <span class="badge bg-white text-dark border px-2 py-1 small mb-1">
                  {{ result.primaryResult?.category || 'Rác tái chế' }}
                </span>
                <h5 class="fw-bold mb-0 text-dark">{{ result.primaryResult?.className }}</h5>
              </div>
            </div>
            <div class="text-end">
              <span class="display-6 fw-bold text-dark">{{ result.primaryResult?.confidencePercent }}%</span>
              <div class="small text-muted">Độ tin cậy</div>
            </div>
          </div>

          <!-- Quick Metrics Bar -->
          <div class="row g-2">
            <div class="col-6">
              <div class="bg-light p-2 rounded-2 text-center border">
                <div class="small text-muted">Thời gian suy luận</div>
                <div class="fw-bold text-dark">{{ result.inferenceTime }} ms</div>
              </div>
            </div>
            <div class="col-6">
              <div class="bg-light p-2 rounded-2 text-center border">
                <div class="small text-muted">Số đối tượng</div>
                <div class="fw-bold text-dark">{{ result.totalObjects }} vật thể</div>
              </div>
            </div>
          </div>

          <!-- Multi-object List -->
          <div>
            <h6 class="fw-bold small text-muted text-uppercase mb-2">Chi tiết các đối tượng phát hiện:</h6>
            <div class="d-flex flex-column gap-2">
              <div 
                v-for="(det, idx) in result.detections" 
                :key="idx" 
                class="p-2 px-3 rounded-2 bg-light border d-flex align-items-center justify-content-between"
              >
                <div class="d-flex align-items-center gap-2">
                  <span class="badge rounded-circle p-1" :style="{ backgroundColor: det.color }"> </span>
                  <span class="fw-semibold text-dark">{{ idx + 1 }}. {{ det.className }}</span>
                </div>
                <div class="d-flex align-items-center gap-2">
                  <div class="progress" style="width: 70px; height: 6px;">
                    <div class="progress-bar" :style="{ width: det.confidencePercent + '%', backgroundColor: det.color }"></div>
                  </div>
                  <span class="small fw-bold">{{ det.confidencePercent }}%</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Bottom Action Buttons -->
          <div class="d-flex gap-2 pt-2">
            <router-link to="/history" class="btn btn-outline-success btn-sm w-100 d-flex align-items-center justify-content-center gap-1">
              <i class="bi bi-clock-history"></i> Xem trong lịch sử
            </router-link>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue';
import yoloWebEngine, { CONF_THRESHOLD, IOU_THRESHOLD } from '../../services/yoloWebEngine';
import apiService from '../../services/api';

const fileInput = ref(null);
const resultCanvas = ref(null);
const selectedFile = ref(null);
const selectedImage = ref(null);
const isDragging = ref(false);
const isLoading = ref(false);
const hasResult = ref(false);
const result = ref(null);

function triggerFileInput() {
  if (fileInput.value) {
    fileInput.value.value = '';
    fileInput.value.click();
  }
}

function handleFileSelect(event) {
  const file = event.target.files?.[0];
  if (file) processSelectedFile(file);
}

function handleDrop(event) {
  isDragging.value = false;
  const file = event.dataTransfer.files?.[0];
  if (file) processSelectedFile(file);
}

function processSelectedFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Vui lòng chọn file hình ảnh hợp lệ (JPG, PNG, WEBP)!');
    return;
  }
  selectedFile.value = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    selectedImage.value = e.target.result;
    result.value = null;
    hasResult.value = false;
    nextTick(() => {
      drawOriginalImage(selectedImage.value);
    });
  };
  reader.readAsDataURL(file);
}

function resetImage() {
  selectedFile.value = null;
  selectedImage.value = null;
  result.value = null;
  hasResult.value = false;
  if (fileInput.value) fileInput.value.value = '';
}

function drawOriginalImage(imgSrc) {
  const canvas = resultCanvas.value;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
  };
  img.src = imgSrc;
}

async function classifyImage() {
  if (!selectedFile.value && !selectedImage.value) return;
  isLoading.value = true;

  try {
    let detectResult = null;
    let fromServer = false;

    // Ưu tiên 1: Gửi ảnh lên Server Backend API (/api/predict)
    // Hoạt động 100% ổn định trên mọi thiết bị và trình duyệt (không phụ thuộc WebGPU của máy khách)
    if (selectedFile.value) {
      try {
        const formData = new FormData();
        formData.append('image', selectedFile.value);
        const res = await apiService.predictImage(formData);
        const data = res?.data || res;
        if (data && (data.success || data.detections)) {
          detectResult = {
            primaryResult: data.primaryResult,
            detections: data.detections || [],
            totalObjects: data.totalObjects !== undefined ? data.totalObjects : (data.detections?.length || 0),
            inferenceTime: data.inferenceTime || 0
          };
          fromServer = true;
        }
      } catch (serverErr) {
        console.warn('[ImageClassifier] Server API không phản hồi, chuyển sang WebGPU fallback:', serverErr.message);
      }
    }

    // Ưu tiên 2: Fallback nhận diện trực tiếp trong trình duyệt bằng WebGPU/WASM (nếu Server tắt)
    if (!detectResult) {
      const clientInferencePromise = (async () => {
        if (yoloWebEngine.status !== 'ready') {
          await yoloWebEngine.loadModel('/models/best.onnx');
        }
        const img = new Image();
        img.src = selectedImage.value;
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
        });
        return await yoloWebEngine.detect(img, CONF_THRESHOLD, IOU_THRESHOLD);
      })();

      // Đặt timeout 12s để tránh treo vô hạn nếu card đồ họa/trình duyệt không hỗ trợ WebGPU
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Dịch vụ AI chưa sẵn sàng hoặc trình duyệt không phản hồi. Vui lòng đảm bảo backend Python yolo_service.py đang chạy.')), 12000)
      );

      detectResult = await Promise.race([clientInferencePromise, timeoutPromise]);
    }

    if (!detectResult) {
      throw new Error('Không nhận được kết quả nhận diện từ mô hình AI.');
    }

    result.value = {
      success: true,
      primaryResult: detectResult.primaryResult,
      detections: detectResult.detections || [],
      totalObjects: detectResult.totalObjects || 0,
      inferenceTime: detectResult.inferenceTime || 0,
      imageUrl: selectedImage.value
    };
    hasResult.value = true;

    nextTick(() => {
      drawDetectionOverlay(selectedImage.value, detectResult.detections || []);
    });

    // Nếu chạy từ WebGPU client, lưu lịch sử thủ công (nếu từ server thì server đã tự lưu)
    if (!fromServer) {
      apiService.saveWebcamHistory({
        image: selectedImage.value,
        method: 'upload',
        primaryResult: detectResult.primaryResult,
        totalObjects: detectResult.totalObjects,
        inferenceTime: detectResult.inferenceTime,
        detections: detectResult.detections
      }).catch((err) => console.warn('Lưu lịch sử ảnh upload:', err));
    }

  } catch (error) {
    console.error('Lỗi phân loại ảnh:', error);
    alert('Không thể thực hiện phân loại ảnh:\n' + (error.message || 'Lỗi không xác định. Hãy kiểm tra xem Python AI Service (yolo_service.py) đã được khởi động chưa.'));
  } finally {
    isLoading.value = false;
  }
}

function drawDetectionOverlay(imgSrc, detections) {
  const canvas = resultCanvas.value;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    // Draw bounding boxes and labels
    detections.forEach((det) => {
      const { bbox, bboxNorm, color, className, confidencePercent } = det;
      let x, y, w, h;
      const bx1 = bboxNorm?.x1 ?? bboxNorm?.normX1;
      const by1 = bboxNorm?.y1 ?? bboxNorm?.normY1;
      const bx2 = bboxNorm?.x2 ?? bboxNorm?.normX2;
      const by2 = bboxNorm?.y2 ?? bboxNorm?.normY2;

      if (bx1 !== undefined && bx2 !== undefined) {
        x = bx1 * canvas.width;
        y = by1 * canvas.height;
        w = (bx2 - bx1) * canvas.width;
        h = (by2 - by1) * canvas.height;
      } else if (bbox) {
        x = bbox.x1;
        y = bbox.y1;
        w = bbox.x2 - bbox.x1;
        h = bbox.y2 - bbox.y1;
      } else {
        return;
      }

      // Box outline
      ctx.strokeStyle = color || '#10b981';
      ctx.lineWidth = Math.max(3, Math.round(canvas.width / 200));
      ctx.strokeRect(x, y, w, h);

      // Semi-transparent fill
      ctx.fillStyle = (color || '#10b981') + '20';
      ctx.fillRect(x, y, w, h);

      // Label background & text
      const labelText = `${className} ${confidencePercent}%`;
      const fontSize = Math.max(14, Math.round(canvas.width / 40));
      ctx.font = `bold ${fontSize}px Inter, sans-serif`;
      const textMetrics = ctx.measureText(labelText);
      const textWidth = textMetrics.width;
      const textHeight = fontSize * 1.3;

      ctx.fillStyle = color || '#10b981';
      ctx.fillRect(x, y - textHeight > 0 ? y - textHeight : y, textWidth + 14, textHeight);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 7, (y - textHeight > 0 ? y - textHeight : y) + fontSize);
    });
  };
  img.src = imgSrc;
}
</script>

<style scoped>
</style>
