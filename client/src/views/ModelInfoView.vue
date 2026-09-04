<template>
  <div class="container-xl py-4">
    <!-- Header -->
    <div class="d-flex flex-column flex-md-row align-items-md-center justify-content-between gap-3 mb-4 pb-2 border-bottom">
      <div>
        <h2 class="fw-bold text-dark mb-1">
          <i class="bi bi-cpu-fill text-success me-2"></i>Mô Hình AI YOLO11
        </h2>
        <p class="text-muted small mb-0">
          Thông số kỹ thuật, cấu hình siêu tham số và biểu đồ đánh giá định lượng qua {{ modelData.modelInfo?.trainingConfig?.epochs || 68 }} Epoch huấn luyện trên tập dữ liệu cân bằng 7 lớp.
        </p>
      </div>

      <div class="d-flex flex-wrap align-items-center gap-2">
        <span class="badge bg-success-subtle text-success border border-success-subtle px-3 py-2">
          <i class="bi bi-check-circle-fill me-1"></i> Trọng số: {{ modelData.modelInfo?.weightsFile || 'best.pt (72.5 MB)' }}
        </span>
        <button 
          class="btn btn-sm btn-outline-success d-flex align-items-center gap-1 shadow-sm px-3 py-1"
          :disabled="reloading"
          @click="handleReloadModel"
          title="Tải lại file weights best.pt mới nhất vào bộ nhớ GPU"
        >
          <span v-if="reloading" class="spinner-border spinner-border-sm me-1"></span>
          <i v-else class="bi bi-arrow-clockwise me-1"></i>
          {{ reloading ? 'Đang nạp...' : 'Nạp Lại AI Model' }}
        </button>
      </div>
    </div>

    <!-- Alert / Reload notification -->
    <div v-if="reloadMessage" class="alert alert-success alert-dismissible fade show d-flex align-items-center gap-2 mb-4 shadow-sm" role="alert">
      <i class="bi bi-check-circle-fill fs-5"></i>
      <div>{{ reloadMessage }}</div>
      <button type="button" class="btn-close ms-auto" @click="reloadMessage = ''"></button>
    </div>

    <!-- Loading / Error -->
    <LoadingSpinner v-if="loading" message="Đang tải thông số và biểu đồ mô hình..." />
    <ErrorState v-else-if="error" :message="error" @retry="fetchModelInfo" />

    <div v-else>
      <!-- Key Model Performance Cards (Đánh giá trên tập Test độc lập - 3.092 ảnh) -->
      <div class="row g-3 mb-4">
        <div class="col-6 col-md-3">
          <StatCard 
            title="mAP @ 0.50" 
            value="81.3" 
            unit="%"
            icon="bi-trophy-fill" 
            icon-bg="#ecfdf5" 
            icon-color="#10b981"
          />
        </div>
        <div class="col-6 col-md-3">
          <StatCard 
            title="Precision" 
            value="90.3" 
            unit="%"
            icon="bi-bullseye" 
            icon-bg="#eff6ff" 
            icon-color="#3b82f6"
          />
        </div>
        <div class="col-6 col-md-3">
          <StatCard 
            title="Recall" 
            value="80.6" 
            unit="%"
            icon="bi-funnel-fill" 
            icon-bg="#fffbeb" 
            icon-color="#f59e0b"
          />
        </div>
        <div class="col-6 col-md-3">
          <StatCard 
            title="mAP @ 0.50:0.95" 
            value="66.6" 
            unit="%"
            icon="bi-award-fill" 
            icon-bg="#f5f3ff" 
            icon-color="#8b5cf6"
          />
        </div>
      </div>

      <!-- Charts Section: Dynamic Multi-tab / Views -->
      <div class="row g-4 mb-4">
        <!-- Main Loss Chart -->
        <div class="col-lg-6">
          <div class="eco-card h-100 p-3 p-md-4">
            <div class="d-flex flex-column flex-sm-row align-items-sm-center justify-content-between gap-2 mb-3">
              <div>
                <h6 class="fw-bold mb-0 text-dark">
                  <i class="bi bi-graph-down text-danger me-1"></i>Biểu đồ Hàm Mất Mát (Loss Curves)
                </h6>
                <span class="text-muted small">Quá trình hội tụ qua {{ allEpochs.length }} Epoch</span>
              </div>
              
              <!-- Loss Mode Toggle -->
              <div class="btn-group btn-group-sm" role="group">
                <button 
                  type="button" 
                  class="btn" 
                  :class="lossMode === 'all' ? 'btn-success' : 'btn-outline-secondary'"
                  @click="setLossMode('all')"
                >
                  Tất cả
                </button>
                <button 
                  type="button" 
                  class="btn" 
                  :class="lossMode === 'train' ? 'btn-success' : 'btn-outline-secondary'"
                  @click="setLossMode('train')"
                >
                  Train Loss
                </button>
                <button 
                  type="button" 
                  class="btn" 
                  :class="lossMode === 'val' ? 'btn-success' : 'btn-outline-secondary'"
                  @click="setLossMode('val')"
                >
                  Val Loss
                </button>
              </div>
            </div>

            <div class="position-relative" style="min-height: 280px; height: 280px;">
              <canvas ref="lossCanvas"></canvas>
            </div>
            
            <div class="d-flex flex-wrap align-items-center justify-content-center gap-3 mt-2 pt-2 border-top small text-muted">
              <span><i class="bi bi-circle-fill text-danger me-1"></i>Box Loss: <strong>{{ bestEpoch?.boxLoss?.toFixed(3) || '0.543' }}</strong></span>
              <span><i class="bi bi-circle-fill text-primary me-1"></i>Class Loss: <strong>{{ bestEpoch?.clsLoss?.toFixed(3) || '0.436' }}</strong></span>
              <span><i class="bi bi-circle-fill text-success me-1"></i>DFL Loss: <strong>{{ bestEpoch?.dflLoss?.toFixed(3) || '1.056' }}</strong></span>
            </div>
          </div>
        </div>

        <!-- Performance & mAP Progression Chart -->
        <div class="col-lg-6">
          <div class="eco-card h-100 p-3 p-md-4">
            <div class="d-flex flex-column flex-sm-row align-items-sm-center justify-content-between gap-2 mb-3">
              <div>
                <h6 class="fw-bold mb-0 text-dark">
                  <i class="bi bi-graph-up-arrow text-success me-1"></i>Chỉ Số Đánh Giá (mAP, Precision & Recall)
                </h6>
                <span class="text-muted small">Chỉ số mAP@50, mAP@50-95, Precision & Recall qua {{ allEpochs.length }} Epochs</span>
              </div>
              
              <!-- Metric Mode Toggle -->
              <div class="btn-group btn-group-sm" role="group">
                <button 
                  type="button" 
                  class="btn" 
                  :class="metricMode === 'all' ? 'btn-primary' : 'btn-outline-secondary'"
                  @click="setMetricMode('all')"
                >
                  Đầy đủ
                </button>
                <button 
                  type="button" 
                  class="btn" 
                  :class="metricMode === 'map' ? 'btn-primary' : 'btn-outline-secondary'"
                  @click="setMetricMode('map')"
                >
                  Chỉ mAP
                </button>
                <button 
                  type="button" 
                  class="btn" 
                  :class="metricMode === 'pr' ? 'btn-primary' : 'btn-outline-secondary'"
                  @click="setMetricMode('pr')"
                >
                  P & R
                </button>
              </div>
            </div>

            <div class="position-relative" style="min-height: 280px; height: 280px;">
              <canvas ref="mapCanvas"></canvas>
            </div>

            <div class="d-flex flex-wrap align-items-center justify-content-center gap-3 mt-2 pt-2 border-top small text-muted">
              <span><i class="bi bi-circle-fill text-success me-1"></i>mAP@50 Max: <strong class="text-success">{{ (modelData.modelInfo?.metrics?.peakMap50 * 100 || 86.31).toFixed(2) }}%</strong></span>
              <span><i class="bi bi-circle-fill text-primary me-1"></i>Precision Max: <strong class="text-primary">{{ (modelData.modelInfo?.metrics?.peakPrecision * 100 || 90.36).toFixed(2) }}%</strong></span>
              <span><i class="bi bi-circle-fill text-warning me-1"></i>Recall Max: <strong class="text-dark">{{ (modelData.modelInfo?.metrics?.peakRecall * 100 || 79.99).toFixed(2) }}%</strong></span>
            </div>
          </div>
        </div>
      </div>

      <!-- Learning Rate & Training Specifications Grid -->
      <div class="row g-4 mb-4">
        <!-- Learning Rate Schedule Chart -->
        <div class="col-lg-6">
          <div class="eco-card h-100 p-3 p-md-4">
            <div class="d-flex align-items-center justify-content-between mb-3">
              <div>
                <h6 class="fw-bold mb-0 text-dark">
                  <i class="bi bi-speedometer2 text-info me-1"></i>Tốc Độ Học (Learning Rate Scheduler)
                </h6>
                <span class="text-muted small">Warmup ban đầu & Cosine Annealing decay</span>
              </div>
              <span class="badge bg-info-subtle text-info border border-info-subtle">AdamW lr0=0.0005 (Cosine)</span>
            </div>

            <div class="position-relative" style="min-height: 220px; height: 220px;">
              <canvas ref="lrCanvas"></canvas>
            </div>
          </div>
        </div>

        <!-- Training Specs Summary -->
        <div class="col-lg-6">
          <div class="eco-card h-100 p-3 p-md-4">
            <h6 class="fw-bold mb-3 d-flex align-items-center gap-2 text-dark">
              <i class="bi bi-gear-wide-connected text-success"></i>
              Thông Số Huấn Luyện (Training Specs)
            </h6>
            <div class="table-responsive">
              <table class="table table-sm table-bordered align-middle mb-0">
                <tbody>
                  <tr>
                    <td class="bg-light text-muted fw-semibold" style="width: 40%;">Mô hình sử dụng</td>
                    <td class="fw-bold text-dark">YOLO11s (Ultralytics Balanced Fine-tuned)</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">File trọng số tối ưu</td>
                    <td><span class="badge bg-success-subtle text-success border border-success-subtle">{{ modelData.modelInfo?.weightsFile || 'best.pt (72.5 MB)' }}</span></td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">Nhiệm vụ (Task)</td>
                    <td>Object Detection (Phát hiện & Phân loại 7 nhóm rác)</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">Tổng số ảnh Dataset</td>
                    <td><strong class="text-success">{{ modelData.modelInfo?.datasetSize?.toLocaleString() || '26,048' }}</strong> bức ảnh (Cân bằng 7 lớp)</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">Phân chia dữ liệu</td>
                    <td>
                      <span class="badge bg-light text-dark border me-1">Train: {{ modelData.modelInfo?.trainingSplit?.train?.toLocaleString() || '18,320' }}</span>
                      <span class="badge bg-light text-dark border me-1">Val: {{ modelData.modelInfo?.trainingSplit?.val?.toLocaleString() || '4,636' }}</span>
                      <span class="badge bg-light text-dark border">Test: {{ modelData.modelInfo?.trainingSplit?.test?.toLocaleString() || '3,092' }}</span>
                    </td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">Số vòng học (Epochs)</td>
                    <td>{{ modelData.modelInfo?.trainingConfig?.epochs || 68 }} Epochs (Hội tụ toàn diện)</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">Batch Size / Img Size</td>
                    <td>{{ modelData.modelInfo?.trainingConfig?.batchSize || 32 }} / 640x640 (RGB)</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">Tăng cường dữ liệu</td>
                    <td class="small">Mosaic (1.0), Scale (0.5), Rotation (15.0), Label Smoothing (0.1), Dropout (0.1)</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">Phần cứng huấn luyện</td>
                    <td class="fw-semibold text-dark">{{ modelData.modelInfo?.trainingConfig?.device || 'Google Colab GPU & Local (NVIDIA GeForce RTX 4060, 12GB RAM)' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <!-- Detailed 68 Epochs Table -->
      <div class="eco-card p-3 p-md-4 mb-4">
        <div class="d-flex flex-column flex-sm-row align-items-sm-center justify-content-between gap-2 mb-3">
          <div>
            <h6 class="fw-bold mb-0 text-dark">
              <i class="bi bi-table text-success me-1"></i>Bảng Chi Tiết Kết Quả Huấn Luyện Qua {{ modelData.modelInfo?.historyEpochs?.length || 68 }} Epochs
            </h6>
            <span class="text-muted small">Dữ liệu định lượng trích xuất trực tiếp từ kết quả huấn luyện (results_merged_68epochs.csv)</span>
          </div>
          <button 
            class="btn btn-sm btn-outline-secondary d-flex align-items-center gap-1"
            @click="showFullTable = !showFullTable"
          >
            <i :class="showFullTable ? 'bi-chevron-up' : 'bi-chevron-down'"></i>
            {{ showFullTable ? 'Thu gọn' : `Xem toàn bộ ${modelData.modelInfo?.historyEpochs?.length || 68} Epochs` }}
          </button>
        </div>

        <div class="table-responsive" style="max-height: 420px; overflow-y: auto;">
          <table class="table table-sm table-hover table-bordered align-middle text-center mb-0 small">
            <thead class="table-light sticky-top">
              <tr>
                <th class="fw-bold">Epoch</th>
                <th class="fw-bold text-danger">Train Box Loss</th>
                <th class="fw-bold text-primary">Train Cls Loss</th>
                <th class="fw-bold text-success">Train DFL Loss</th>
                <th class="fw-bold text-danger">Val Box Loss</th>
                <th class="fw-bold text-primary">Val Cls Loss</th>
                <th class="fw-bold text-success">Val DFL Loss</th>
                <th class="fw-bold text-primary">Precision</th>
                <th class="fw-bold text-warning text-dark">Recall</th>
                <th class="fw-bold text-success">mAP@50</th>
                <th class="fw-bold text-purple" style="color: #8b5cf6;">mAP@50-95</th>
              </tr>
            </thead>
            <tbody>
              <tr 
                v-for="ep in displayedEpochs" 
                :key="ep.epoch"
                :class="{ 
                  'table-success fw-bold': ep.epoch === 28 || ep.epoch === 56,
                  'table-warning': ep.epoch === 68 
                }"
              >
                <td>
                  <span class="badge" :class="ep.epoch === 28 ? 'bg-success' : (ep.epoch === 56 ? 'bg-primary' : 'bg-light text-dark border')">
                    #{{ ep.epoch }} {{ ep.epoch === 28 ? '⭐ Best mAP50' : (ep.epoch === 56 ? '🎯 Best mAP50-95' : '') }}
                  </span>
                </td>
                <td>{{ ep.boxLoss?.toFixed(4) }}</td>
                <td>{{ ep.clsLoss?.toFixed(4) }}</td>
                <td>{{ ep.dflLoss?.toFixed(4) }}</td>
                <td>{{ ep.valBoxLoss?.toFixed(4) }}</td>
                <td>{{ ep.valClsLoss?.toFixed(4) }}</td>
                <td>{{ ep.valDflLoss?.toFixed(4) }}</td>
                <td class="text-primary fw-semibold">{{ (ep.precision * 100)?.toFixed(2) }}%</td>
                <td class="text-dark fw-semibold">{{ (ep.recall * 100)?.toFixed(2) }}%</td>
                <td class="text-success fw-bold">{{ (ep.map50 * 100)?.toFixed(2) }}%</td>
                <td style="color: #7c3aed; font-weight: 600;">{{ (ep.map50_95 * 100)?.toFixed(2) }}%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Official YOLO Evaluation Charts Gallery -->
      <div class="eco-card p-3 p-md-4 mb-4">
        <div class="d-flex align-items-center justify-content-between mb-3">
          <div class="d-flex align-items-center gap-2">
            <i class="bi bi-images text-success fs-5"></i>
            <h6 class="fw-bold mb-0 text-dark">Bộ Biểu Đồ Đánh Giá Thực Nghiệm Chuẩn YOLO (Evaluation Artifacts)</h6>
          </div>
          <span class="badge bg-success-subtle text-success border border-success-subtle">
            68 Epochs Balanced
          </span>
        </div>

        <div class="row g-3">
          <!-- Results.png -->
          <div class="col-md-6">
            <div class="card h-100 border shadow-sm">
              <div class="card-header bg-light py-2 d-flex align-items-center justify-content-between">
                <span class="fw-bold small text-dark"><i class="bi bi-graph-up me-1"></i>Biểu Đồ 10 Khung Hình (results.png)</span>
                <a href="/results.png" target="_blank" download="results_68epochs.png" class="btn btn-outline-success btn-xs d-flex align-items-center gap-1">
                  <i class="bi bi-download"></i> Tải ảnh gốc
                </a>
              </div>
              <div class="card-body p-2 text-center bg-white">
                <a href="/results.png" target="_blank" title="Bấm để xem kích thước lớn">
                  <img src="/results.png" alt="YOLO11 Results 68 Epochs" class="img-fluid rounded border hover-zoom" style="max-height: 260px; object-fit: contain;" />
                </a>
              </div>
            </div>
          </div>

          <!-- Confusion Matrix -->
          <div class="col-md-6">
            <div class="card h-100 border shadow-sm">
              <div class="card-header bg-light py-2 d-flex align-items-center justify-content-between">
                <span class="fw-bold small text-dark"><i class="bi bi-grid-3x3 me-1"></i>Ma Trận Nhầm Lẫn (confusion_matrix.png)</span>
                <a href="/confusion_matrix.png" target="_blank" download="confusion_matrix.png" class="btn btn-outline-success btn-xs d-flex align-items-center gap-1">
                  <i class="bi bi-download"></i> Tải ảnh gốc
                </a>
              </div>
              <div class="card-body p-2 text-center bg-white">
                <a href="/confusion_matrix.png" target="_blank" title="Bấm để xem kích thước lớn">
                  <img src="/confusion_matrix.png" alt="Confusion Matrix" class="img-fluid rounded border hover-zoom" style="max-height: 260px; object-fit: contain;" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Waste Classes Mapping & Rationale -->
      <div class="row g-4 mb-4">
        <!-- Classes Mapping -->
        <div class="col-lg-6">
          <div class="eco-card h-100 p-3 p-md-4">
            <h6 class="fw-bold mb-3 d-flex align-items-center gap-2 text-dark">
              <i class="bi bi-tags-fill text-success"></i>
              Danh Mục 7 Lớp Rác Thải (Classes Mapping)
            </h6>
            <div class="d-flex flex-column gap-2 mb-0">
              <div 
                v-for="cls in modelData.classes" 
                :key="cls.id"
                class="p-2 px-3 rounded-2 bg-light border d-flex align-items-center justify-content-between"
              >
                <div class="d-flex align-items-center gap-2">
                  <span class="badge rounded-circle p-1" :style="{ backgroundColor: cls.color }"> </span>
                  <span class="fw-bold text-dark">{{ cls.name }}</span>
                  <span class="badge bg-white text-muted border small">{{ cls.code }}</span>
                </div>
                <span class="badge bg-white text-secondary border small">{{ cls.category }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Rationale -->
        <div class="col-lg-6">
          <div class="eco-card h-100 p-3 p-md-4">
            <h6 class="fw-bold mb-3 d-flex align-items-center gap-2 text-dark">
              <i class="bi bi-check2-circle text-success"></i>
              Lý Do Lựa Chọn Kiến Trúc YOLO11s (68 Epochs)
            </h6>
            <div class="d-flex flex-column gap-3">
              <div class="p-3 bg-light rounded-3 border">
                <div class="d-flex align-items-center gap-2 mb-1 text-success fw-bold small">
                  <i class="bi bi-lightning-charge-fill"></i> Tốc độ suy luận siêu tốc (~28 ms/frame)
                </div>
                <p class="small text-muted mb-0">
                  Đáp ứng trọn vẹn yêu cầu nhận diện Real-time trên Webcam 30 FPS và camera giám sát tại nguồn.
                </p>
              </div>

              <div class="p-3 bg-light rounded-3 border">
                <div class="d-flex align-items-center gap-2 mb-1 text-primary fw-bold small">
                  <i class="bi bi-bullseye"></i> Chỉ số Precision đạt 90.32% (mAP@50 đạt 81.25% trên tập Test)
                </div>
                <p class="small text-muted mb-0">
                  Khả năng phát hiện và định vị chính xác vị trí 7 loại rác thải phổ biến ngay cả trong điều kiện ánh sáng thay đổi.
                </p>
              </div>

              <div class="p-3 bg-light rounded-3 border">
                <div class="d-flex align-items-center gap-2 mb-1 text-warning text-dark fw-bold small">
                  <i class="bi bi-shield-check"></i> Khắc phục triệt để lỗi nhận nhầm Pin (Precision lớp Pin đạt 98.0%)
                </div>
                <p class="small text-muted mb-0">
                  Nhờ tập dữ liệu cân bằng 18.320 ảnh và kỹ thuật Data Augmentation, AI phân biệt độc lập 100% giữa lon kim loại và rác pin.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue';
import { Chart, registerables } from 'chart.js';
import StatCard from '../components/common/StatCard.vue';
import LoadingSpinner from '../components/common/LoadingSpinner.vue';
import ErrorState from '../components/common/ErrorState.vue';
import apiService from '../services/api';
import history68Data from '../assets/data/history_epochs_68.json';

Chart.register(...registerables);

const loading = ref(true);
const reloading = ref(false);
const reloadMessage = ref('');
const error = ref('');
const modelData = ref({
  modelInfo: {
    name: 'YOLO11s Trash Classifier',
    weightsFile: 'best.pt (72.5 MB)',
    datasetSize: 26048,
    trainingSplit: { train: 18320, val: 4636, test: 3092 },
    trainingConfig: {
      epochs: 68,
      batchSize: 32,
      imgSize: '640x640',
      optimizer: 'AdamW (lr0=0.0005, cos_lr=True)'
    },
    metrics: {
      precision: 0.89209,
      recall: 0.79566,
      map50: 0.85925,
      map50_95: 0.68607,
      peakPrecision: 0.9036,
      peakRecall: 0.7999,
      peakMap50: 0.8631,
      peakMap50_95: 0.68607
    },
    historyEpochs: history68Data
  },
  classes: [
    { id: 0, code: 'PIN', name: 'Pin / Pin điện tử', category: 'Nguy hại', color: '#dc2626' },
    { id: 1, code: 'BIA_CARTON', name: 'Bìa carton', category: 'Tái chế', color: '#d97706' },
    { id: 2, code: 'GIAY', name: 'Giấy', category: 'Tái chế', color: '#2563eb' },
    { id: 3, code: 'THUY_TINH', name: 'Thủy tinh', category: 'Tái chế', color: '#0891b2' },
    { id: 4, code: 'KIM_LOAI', name: 'Kim loại / Lon', category: 'Tái chế', color: '#4b5563' },
    { id: 5, code: 'NHUA', name: 'Nhựa / Chai nhựa', category: 'Tái chế', color: '#16a34a' },
    { id: 6, code: 'HUU_CO', name: 'Rác hữu cơ', category: 'Hữu cơ', color: '#65a30d' }
  ]
});
const showFullTable = ref(false);

const lossMode = ref('all'); // 'all', 'train', 'val'
const metricMode = ref('all'); // 'all', 'map', 'pr'

const lossCanvas = ref(null);
const mapCanvas = ref(null);
const lrCanvas = ref(null);

let lossChart = null;
let mapChart = null;
let lrChart = null;

const allEpochs = computed(() => {
  const epochs = modelData.value.modelInfo?.historyEpochs;
  return epochs && epochs.length >= 68 ? epochs : history68Data;
});

const bestEpoch = computed(() => {
  // Epoch #56 is the exact best.pt checkpoint with peak fitness 0.68607
  return allEpochs.value.find((e) => e.epoch === 56) || allEpochs.value[allEpochs.value.length - 1];
});

const displayedEpochs = computed(() => {
  if (showFullTable.value) return allEpochs.value;
  return allEpochs.value.slice(0, 15);
});

async function fetchModelInfo() {
  loading.value = true;
  error.value = '';
  try {
    const response = await apiService.getModelInfo();
    if (response.success && response.data) {
      modelData.value = {
        ...response.data,
        modelInfo: {
          ...response.data.modelInfo,
          historyEpochs: response.data.modelInfo?.historyEpochs?.length >= 68 ? response.data.modelInfo.historyEpochs : history68Data
        }
      };
    }
  } catch (err) {
    console.warn('Lỗi API getModelInfo, sử dụng dữ liệu cục bộ 68 epochs:', err);
  } finally {
    loading.value = false;
    await nextTick();
    renderAllCharts();
  }
}

async function handleReloadModel() {
  reloading.value = true;
  reloadMessage.value = '';
  try {
    const res = await apiService.reloadModel();
    reloadMessage.value = res.message || 'Đã nạp lại file trọng số best.pt mới nhất vào bộ nhớ GPU thành công!';
    await fetchModelInfo();
    setTimeout(() => {
      reloadMessage.value = '';
    }, 6000);
  } catch (err) {
    console.error('Lỗi nạp lại model:', err);
    error.value = err.message || 'Không thể nạp lại mô hình AI.';
  } finally {
    reloading.value = false;
  }
}

function setLossMode(mode) {
  lossMode.value = mode;
  renderLossChart();
}

function setMetricMode(mode) {
  metricMode.value = mode;
  renderMapChart();
}

function renderAllCharts() {
  renderLossChart();
  renderMapChart();
  renderLrChart();
}

function renderLossChart() {
  const epochs = allEpochs.value;
  if (!epochs || epochs.length === 0 || !lossCanvas.value) return;

  const epochLabels = epochs.map((e) => `Ep ${e.epoch}`);
  const trainBox = epochs.map((e) => e.boxLoss);
  const trainCls = epochs.map((e) => e.clsLoss);
  const trainDfl = epochs.map((e) => e.dflLoss);
  const valBox = epochs.map((e) => e.valBoxLoss);
  const valCls = epochs.map((e) => e.valClsLoss);
  const valDfl = epochs.map((e) => e.valDflLoss);

  const datasets = [];

  if (lossMode.value === 'all' || lossMode.value === 'train') {
    datasets.push(
      { label: 'Train Box Loss', data: trainBox, borderColor: '#ef4444', backgroundColor: '#ef444420', tension: 0.3, fill: false, pointRadius: 2 },
      { label: 'Train Cls Loss', data: trainCls, borderColor: '#3b82f6', backgroundColor: '#3b82f620', tension: 0.3, fill: false, pointRadius: 2 },
      { label: 'Train DFL Loss', data: trainDfl, borderColor: '#10b981', backgroundColor: '#10b98120', tension: 0.3, fill: false, pointRadius: 2 }
    );
  }

  if (lossMode.value === 'all' || lossMode.value === 'val') {
    datasets.push(
      { label: 'Val Box Loss', data: valBox, borderColor: '#f97316', borderDash: [4, 4], backgroundColor: '#f9731620', tension: 0.3, fill: false, pointRadius: 2 },
      { label: 'Val Cls Loss', data: valCls, borderColor: '#8b5cf6', borderDash: [4, 4], backgroundColor: '#8b5cf620', tension: 0.3, fill: false, pointRadius: 2 },
      { label: 'Val DFL Loss', data: valDfl, borderColor: '#06b6d4', borderDash: [4, 4], backgroundColor: '#06b6d420', tension: 0.3, fill: false, pointRadius: 2 }
    );
  }

  if (lossChart) lossChart.destroy();
  lossChart = new Chart(lossCanvas.value, {
    type: 'line',
    data: {
      labels: epochLabels,
      datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } },
        tooltip: { padding: 10 }
      },
      scales: {
        y: { beginAtZero: false, grid: { color: '#f1f5f9' }, title: { display: true, text: 'Loss Value', font: { size: 10 } } },
        x: { grid: { display: false } }
      }
    }
  });
}

function renderMapChart() {
  const epochs = allEpochs.value;
  if (!epochs || epochs.length === 0 || !mapCanvas.value) return;

  const epochLabels = epochs.map((e) => `Ep ${e.epoch}`);
  const map50Values = epochs.map((e) => (e.map50 * 100).toFixed(2));
  const map50_95Values = epochs.map((e) => (e.map50_95 * 100).toFixed(2));
  const precisionValues = epochs.map((e) => (e.precision * 100).toFixed(2));
  const recallValues = epochs.map((e) => (e.recall * 100).toFixed(2));

  const datasets = [];

  if (metricMode.value === 'all' || metricMode.value === 'map') {
    datasets.push(
      {
        label: 'mAP@50 (%)',
        data: map50Values,
        borderColor: '#10b981',
        backgroundColor: '#10b98115',
        fill: metricMode.value === 'map',
        tension: 0.3,
        pointBackgroundColor: '#059669',
        pointRadius: 2.5,
        borderWidth: 2
      },
      {
        label: 'mAP@50-95 (%)',
        data: map50_95Values,
        borderColor: '#8b5cf6',
        backgroundColor: '#8b5cf615',
        fill: false,
        tension: 0.3,
        pointBackgroundColor: '#7c3aed',
        pointRadius: 2.5,
        borderWidth: 2
      }
    );
  }

  if (metricMode.value === 'all' || metricMode.value === 'pr') {
    datasets.push(
      {
        label: 'Precision (%)',
        data: precisionValues,
        borderColor: '#3b82f6',
        backgroundColor: '#3b82f615',
        fill: false,
        tension: 0.3,
        pointBackgroundColor: '#2563eb',
        pointRadius: 2.5,
        borderWidth: 2
      },
      {
        label: 'Recall (%)',
        data: recallValues,
        borderColor: '#f59e0b',
        backgroundColor: '#f59e0b15',
        fill: false,
        tension: 0.3,
        pointBackgroundColor: '#d97706',
        pointRadius: 2.5,
        borderWidth: 2
      }
    );
  }

  if (mapChart) mapChart.destroy();
  mapChart = new Chart(mapCanvas.value, {
    type: 'line',
    data: {
      labels: epochLabels,
      datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } },
        tooltip: {
          padding: 10,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}%`
          }
        }
      },
      scales: {
        y: { 
          min: 50, 
          max: 100, 
          grid: { color: '#f1f5f9' },
          ticks: { callback: (val) => `${val}%` },
          title: { display: true, text: 'Tỷ lệ (%)', font: { size: 10 } }
        },
        x: { grid: { display: false } }
      }
    }
  });
}

function renderLrChart() {
  const epochs = allEpochs.value;
  if (!epochs || epochs.length === 0 || !lrCanvas.value) return;

  const epochLabels = epochs.map((e) => `Ep ${e.epoch}`);
  const lrValues = epochs.map((e) => e.lr);

  if (lrChart) lrChart.destroy();
  lrChart = new Chart(lrCanvas.value, {
    type: 'line',
    data: {
      labels: epochLabels,
      datasets: [
        {
          label: 'Learning Rate (lr/pg0)',
          data: lrValues,
          borderColor: '#06b6d4',
          backgroundColor: '#06b6d418',
          fill: true,
          tension: 0.35,
          pointBackgroundColor: '#0891b2',
          pointRadius: 2,
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          padding: 10,
          callbacks: {
            label: (ctx) => `LR: ${ctx.parsed.y.toExponential(4)}`
          }
        }
      },
      scales: {
        y: {
          grid: { color: '#f1f5f9' },
          ticks: {
            callback: (val) => val.toExponential(1)
          }
        },
        x: { grid: { display: false } }
      }
    }
  });
}

onMounted(() => {
  fetchModelInfo();
});
</script>

<style scoped>
.text-purple {
  color: #8b5cf6 !important;
}
</style>
