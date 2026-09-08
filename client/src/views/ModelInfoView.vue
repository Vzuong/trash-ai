<template>
  <div class="container-xl py-4">
    <!-- Header -->
    <div class="d-flex flex-column flex-md-row align-items-md-center justify-content-between gap-3 mb-4 pb-2 border-bottom">
      <div>
        <h2 class="fw-bold text-dark mb-1">
          <i class="bi bi-cpu-fill text-success me-2"></i>Mô Hình AI YOLO11s-CBAM (Fold 1)
        </h2>
        <p class="text-muted small mb-0">
          Thông số kỹ thuật, cấu hình siêu tham số và biểu đồ đánh giá định lượng qua {{ modelData.modelInfo?.trainingConfig?.epochs || 100 }} Epoch huấn luyện trên tập dữ liệu cân bằng 7 lớp.
        </p>
      </div>

      <div class="d-flex flex-wrap align-items-center gap-2">
        <span class="badge bg-success-subtle text-success border border-success-subtle px-3 py-2">
          <i class="bi bi-check-circle-fill me-1"></i> Trọng số: {{ modelData.modelInfo?.weightsFile || 'best.pt (17.4 MB)' }}
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
      <!-- Key Model Performance Cards (Đánh giá trên tập Validation Fold 1 - 8.682 ảnh) -->
      <div class="row g-3 mb-4">
        <div class="col-6 col-md-3">
          <StatCard 
            title="mAP @ 0.50" 
            :value="((modelData.modelInfo?.metrics?.map50 || 0.9118) * 100).toFixed(1)" 
            unit="%"
            icon="bi-trophy-fill" 
            icon-bg="#ecfdf5" 
            icon-color="#10b981"
          />
        </div>
        <div class="col-6 col-md-3">
          <StatCard 
            title="Precision" 
            :value="((modelData.modelInfo?.metrics?.precision || 0.9305) * 100).toFixed(1)" 
            unit="%"
            icon="bi-bullseye" 
            icon-bg="#eff6ff" 
            icon-color="#3b82f6"
          />
        </div>
        <div class="col-6 col-md-3">
          <StatCard 
            title="Recall" 
            :value="((modelData.modelInfo?.metrics?.recall || 0.8563) * 100).toFixed(1)" 
            unit="%"
            icon="bi-funnel-fill" 
            icon-bg="#fffbeb" 
            icon-color="#f59e0b"
          />
        </div>
        <div class="col-6 col-md-3">
          <StatCard 
            title="mAP @ 0.50:0.95" 
            :value="((modelData.modelInfo?.metrics?.map50_95 || 0.7882) * 100).toFixed(1)" 
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
              <span class="badge bg-info-subtle text-info border border-info-subtle">AdamW lr0=0.001 (Plateau)</span>
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
                    <td class="fw-bold text-dark">YOLO11s + CBAM Attention (Fold 1 Best Checkpoint)</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">File trọng số tối ưu</td>
                    <td><span class="badge bg-success-subtle text-success border border-success-subtle">{{ modelData.modelInfo?.weightsFile || 'best.pt (17.4 MB)' }}</span></td>
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
                      <span class="badge bg-light text-dark border me-1">Train: 17,366</span>
                      <span class="badge bg-light text-dark border me-1">Val (Fold 1): 8,682</span>
                      <span class="badge bg-light text-dark border">Phương pháp: 3-Fold CV</span>
                    </td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">Số vòng học (Epochs)</td>
                    <td>{{ modelData.modelInfo?.trainingConfig?.epochs || 100 }} Epochs (Hội tụ toàn diện)</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">Batch Size / Img Size</td>
                    <td>{{ modelData.modelInfo?.trainingConfig?.batchSize || 32 }} / 640x640 (RGB)</td>
                  </tr>
                  <tr>
                    <td class="bg-light text-muted fw-semibold">Tăng cường dữ liệu</td>
                    <td class="small">MixUp (0.15), Copy-Paste (0.3), Erasing (0.4), Dropout (0.1)</td>
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

      <!-- Detailed Epochs Table -->
      <div class="eco-card p-3 p-md-4 mb-4">
        <div class="d-flex flex-column flex-sm-row align-items-sm-center justify-content-between gap-2 mb-3">
          <div>
            <h6 class="fw-bold mb-0 text-dark">
              <i class="bi bi-table text-success me-1"></i>Bảng Chi Tiết Kết Quả Huấn Luyện Qua {{ modelData.modelInfo?.historyEpochs?.length || 100 }} Epochs
            </h6>
            <span class="text-muted small">Dữ liệu định lượng trích xuất trực tiếp từ kết quả huấn luyện (results.csv - Fold 1)</span>
          </div>
          <button 
            class="btn btn-sm btn-outline-secondary d-flex align-items-center gap-1"
            @click="showFullTable = !showFullTable"
          >
            <i :class="showFullTable ? 'bi-chevron-up' : 'bi-chevron-down'"></i>
            {{ showFullTable ? 'Thu gọn' : `Xem toàn bộ ${modelData.modelInfo?.historyEpochs?.length || 100} Epochs` }}
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
                  'table-success fw-bold': ep.epoch === 100,
                  'table-warning': ep.epoch === 99 
                }"
              >
                <td>
                  <span class="badge" :class="ep.epoch === 100 ? 'bg-success' : 'bg-light text-dark border'">
                    #{{ ep.epoch }} {{ ep.epoch === 100 ? '⭐ Best Checkpoint' : '' }}
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
        <div class="d-flex flex-column flex-md-row align-items-md-center justify-content-between gap-2 mb-3">
          <div class="d-flex align-items-center gap-2">
            <i class="bi bi-images text-success fs-5"></i>
            <div>
              <h6 class="fw-bold mb-0 text-dark">Bộ Biểu Đồ Đánh Giá Thực Nghiệm Chuẩn YOLO (Evaluation Artifacts)</h6>
              <span class="text-muted small">Kiểm thử độc lập trên tập Validation Fold 1 (8.682 bức ảnh, 100 Epochs)</span>
            </div>
          </div>
          <div class="d-flex align-items-center gap-2">
            <span class="badge bg-success-subtle text-success border border-success-subtle px-2 py-1">
              <i class="bi bi-check2-all me-1"></i>100 Epochs Balanced
            </span>
          </div>
        </div>

        <div class="row g-4">
          <!-- Card 1: Ma Trận Nhầm Lẫn Chuẩn Hóa (%) -->
          <div class="col-lg-6">
            <div class="card h-100 border shadow-sm rounded-3 overflow-hidden">
              <div class="card-header bg-light py-2 px-3 d-flex align-items-center justify-content-between">
                <div class="d-flex align-items-center gap-2">
                  <span class="badge bg-success text-white">Khuyến Nghị</span>
                  <span class="fw-bold small text-dark"><i class="bi bi-grid-3x3 me-1"></i>Ma Trận Nhầm Lẫn Chuẩn Hóa (%)</span>
                </div>
                <div class="d-flex align-items-center gap-1">
                  <button 
                    type="button"
                    class="btn btn-outline-secondary btn-xs d-flex align-items-center gap-1"
                    @click="openModalImage('/confusion_matrix_normalized.png', 'Ma Trận Nhầm Lẫn Chuẩn Hóa (Normalized Confusion Matrix)', 'Tỷ lệ nhận diện chính xác theo từng lớp (Recall). Đạt 98% ở 3 nhóm rác phổ biến và 91% rác hữu cơ.')"
                    title="Phóng to"
                  >
                    <i class="bi bi-arrows-fullscreen"></i>
                  </button>
                  <a href="/confusion_matrix_normalized.png" target="_blank" download="confusion_matrix_normalized.png" class="btn btn-outline-success btn-xs d-flex align-items-center gap-1">
                    <i class="bi bi-download"></i> Tải ảnh 3K
                  </a>
                </div>
              </div>
              <div 
                class="card-body p-2 text-center bg-white cursor-pointer" 
                @click="openModalImage('/confusion_matrix_normalized.png', 'Ma Trận Nhầm Lẫn Chuẩn Hóa (Normalized Confusion Matrix)', 'Tỷ lệ nhận diện chính xác theo từng lớp (Recall). Đạt 98% ở 3 nhóm rác phổ biến và 91% rác hữu cơ.')"
                title="Bấm để xem kích thước lớn"
              >
                <img 
                  src="/confusion_matrix_normalized.png" 
                  alt="Confusion Matrix Normalized" 
                  class="img-fluid rounded border hover-zoom" 
                  style="max-height: 280px; object-fit: contain; width: 100%;" 
                />
              </div>
              <div class="card-footer bg-light p-2 small">
                <div class="d-flex flex-wrap gap-1 justify-content-center mb-1">
                  <span class="badge bg-danger-subtle text-danger border border-danger-subtle">Pin: 98%</span>
                  <span class="badge bg-warning-subtle text-dark border border-warning-subtle">Bìa: 98%</span>
                  <span class="badge bg-info-subtle text-dark border border-info-subtle">Thủy tinh: 98%</span>
                  <span class="badge bg-success-subtle text-success border border-success-subtle">Hữu cơ: 91%</span>
                  <span class="badge bg-secondary-subtle text-secondary border border-secondary-subtle">Kim loại: 85%</span>
                  <span class="badge bg-primary-subtle text-primary border border-primary-subtle">Giấy: 83%</span>
                  <span class="badge bg-light text-dark border">Nhựa: 67%</span>
                </div>
                <p class="text-muted text-center mb-0 extra-small">
                  Chuẩn hóa theo từng lớp thực tế (True label). Độ chính xác đạt 98% trên 3 lớp rác phổ biến và 91% rác hữu cơ.
                </p>
              </div>
            </div>
          </div>

          <!-- Card 2: Biểu Đồ 10 Khung Hình Huấn Luyện (results.png) -->
          <div class="col-lg-6">
            <div class="card h-100 border shadow-sm rounded-3 overflow-hidden">
              <div class="card-header bg-light py-2 px-3 d-flex align-items-center justify-content-between">
                <div class="d-flex align-items-center gap-2">
                  <span class="badge bg-primary text-white">100 Epochs</span>
                  <span class="fw-bold small text-dark"><i class="bi bi-graph-up me-1"></i>Biểu Đồ Huấn Luyện 10 Khung Hình (results.png)</span>
                </div>
                <div class="d-flex align-items-center gap-1">
                  <button 
                    type="button"
                    class="btn btn-outline-secondary btn-xs d-flex align-items-center gap-1"
                    @click="openModalImage('/results.png', 'Biểu Đồ Tiến Trình Huấn Luyện 100 Epochs (results.png)', '10 đồ thị chuẩn của Ultralytics thể hiện Loss đào tạo/kiểm định và các chỉ số Precision, Recall, mAP50, mAP50-95 qua 100 Epochs.')"
                    title="Phóng to"
                  >
                    <i class="bi bi-arrows-fullscreen"></i>
                  </button>
                  <a href="/results.png" target="_blank" download="results_100epochs.png" class="btn btn-outline-success btn-xs d-flex align-items-center gap-1">
                    <i class="bi bi-download"></i> Tải ảnh gốc
                  </a>
                </div>
              </div>
              <div 
                class="card-body p-2 text-center bg-white cursor-pointer" 
                @click="openModalImage('/results.png', 'Biểu Đồ Tiến Trình Huấn Luyện 100 Epochs (results.png)', '10 đồ thị chuẩn của Ultralytics thể hiện Loss đào tạo/kiểm định và các chỉ số Precision, Recall, mAP50, mAP50-95 qua 100 Epochs.')"
                title="Bấm để xem kích thước lớn"
              >
                <img 
                  src="/results.png" 
                  alt="YOLO11 Results 100 Epochs" 
                  class="img-fluid rounded border hover-zoom" 
                  style="max-height: 280px; object-fit: contain; width: 100%;" 
                />
              </div>
              <div class="card-body py-1 px-2 border-top bg-light small">
                <div class="d-flex flex-wrap justify-content-around text-center gap-1">
                  <span class="small">Box Loss: <strong class="text-danger">0.468</strong></span>
                  <span class="small">Cls Loss: <strong class="text-primary">0.354</strong></span>
                  <span class="small">DFL Loss: <strong class="text-success">0.985</strong></span>
                  <span class="small">Precision: <strong class="text-primary">93.1%</strong></span>
                  <span class="small">Recall: <strong class="text-dark">85.6%</strong></span>
                  <span class="small">mAP@50: <strong class="text-success">91.2%</strong></span>
                </div>
                <p class="text-muted text-center mb-0 mt-1 extra-small">
                  Loss giảm đều và ổn định. Tại epoch 90 tắt mosaic augmentation giúp mô hình hội tụ sắc nét ở điểm tối ưu.
                </p>
              </div>
            </div>
          </div>

          <!-- Card 3: Ma Trận Nhầm Lẫn Mẫu Thô (confusion_matrix.png) -->
          <div class="col-lg-6">
            <div class="card h-100 border shadow-sm rounded-3 overflow-hidden">
              <div class="card-header bg-light py-2 px-3 d-flex align-items-center justify-content-between">
                <div class="d-flex align-items-center gap-2">
                  <span class="badge bg-secondary text-white">Số Lượng</span>
                  <span class="fw-bold small text-dark"><i class="bi bi-grid-3x3 me-1"></i>Ma Trận Số Lượng Mẫu Thô (confusion_matrix.png)</span>
                </div>
                <div class="d-flex align-items-center gap-1">
                  <button 
                    type="button"
                    class="btn btn-outline-secondary btn-xs d-flex align-items-center gap-1"
                    @click="openModalImage('/confusion_matrix.png', 'Ma Trận Nhầm Lẫn Số Lượng Mẫu Thô (confusion_matrix.png)', 'Tổng hợp số lượng mẫu dự đoán chính xác và nhầm lẫn trên 8.682 ảnh kiểm thử (Fold 1).')"
                    title="Phóng to"
                  >
                    <i class="bi bi-arrows-fullscreen"></i>
                  </button>
                  <a href="/confusion_matrix.png" target="_blank" download="confusion_matrix_raw.png" class="btn btn-outline-success btn-xs d-flex align-items-center gap-1">
                    <i class="bi bi-download"></i> Tải ảnh 3K
                  </a>
                </div>
              </div>
              <div 
                class="card-body p-2 text-center bg-white cursor-pointer" 
                @click="openModalImage('/confusion_matrix.png', 'Ma Trận Nhầm Lẫn Số Lượng Mẫu Thô (confusion_matrix.png)', 'Tổng hợp số lượng mẫu dự đoán chính xác và nhầm lẫn trên 8.682 ảnh kiểm thử (Fold 1).')"
                title="Bấm để xem kích thước lớn"
              >
                <img 
                  src="/confusion_matrix.png" 
                  alt="Raw Confusion Matrix" 
                  class="img-fluid rounded border hover-zoom" 
                  style="max-height: 280px; object-fit: contain; width: 100%;" 
                />
              </div>
              <div class="card-footer bg-light p-2 small">
                <p class="text-muted text-center mb-0 extra-small">
                  Đếm số lượng bounding box thực tế được phân loại đúng và sai trong tập kiểm thử độc lập 8.682 ảnh.
                </p>
              </div>
            </div>
          </div>

          <!-- Card 4: Đường Cong Precision-Recall & F1 Curve -->
          <div class="col-lg-6">
            <div class="card h-100 border shadow-sm rounded-3 overflow-hidden">
              <div class="card-header bg-light py-2 px-3 d-flex align-items-center justify-content-between">
                <div class="d-flex align-items-center gap-2">
                  <div class="btn-group btn-group-sm">
                    <button 
                      type="button"
                      class="btn btn-xs" 
                      :class="activeCurve === 'pr' ? 'btn-success' : 'btn-outline-secondary'"
                      @click="activeCurve = 'pr'"
                    >
                      Precision-Recall
                    </button>
                    <button 
                      type="button"
                      class="btn btn-xs" 
                      :class="activeCurve === 'f1' ? 'btn-success' : 'btn-outline-secondary'"
                      @click="activeCurve = 'f1'"
                    >
                      F1-Score
                    </button>
                  </div>
                </div>
                <div class="d-flex align-items-center gap-1">
                  <button 
                    type="button"
                    class="btn btn-outline-secondary btn-xs d-flex align-items-center gap-1"
                    @click="openModalImage(activeCurve === 'pr' ? '/BoxPR_curve.png' : '/BoxF1_curve.png', activeCurve === 'pr' ? 'Đường Cong Precision - Recall (BoxPR_curve.png)' : 'Đường Cong F1 - Confidence (BoxF1_curve.png)', activeCurve === 'pr' ? 'Đường cong Precision-Recall đánh giá độ chính xác tổng thể mAP@0.50 đạt 91.2%.' : 'Đường cong F1-Score đạt đỉnh 0.89 tại ngưỡng confidence 0.472.')"
                    title="Phóng to"
                  >
                    <i class="bi bi-arrows-fullscreen"></i>
                  </button>
                  <a :href="activeCurve === 'pr' ? '/BoxPR_curve.png' : '/BoxF1_curve.png'" target="_blank" :download="activeCurve === 'pr' ? 'BoxPR_curve.png' : 'BoxF1_curve.png'" class="btn btn-outline-success btn-xs d-flex align-items-center gap-1">
                    <i class="bi bi-download"></i> Tải ảnh
                  </a>
                </div>
              </div>
              <div 
                class="card-body p-2 text-center bg-white cursor-pointer" 
                @click="openModalImage(activeCurve === 'pr' ? '/BoxPR_curve.png' : '/BoxF1_curve.png', activeCurve === 'pr' ? 'Đường Cong Precision - Recall (BoxPR_curve.png)' : 'Đường Cong F1 - Confidence (BoxF1_curve.png)', activeCurve === 'pr' ? 'Đường cong Precision-Recall đánh giá độ chính xác tổng thể mAP@0.50 đạt 91.2%.' : 'Đường cong F1-Score đạt đỉnh 0.89 tại ngưỡng confidence 0.472.')"
                title="Bấm để xem kích thước lớn"
              >
                <img 
                  :src="activeCurve === 'pr' ? '/BoxPR_curve.png' : '/BoxF1_curve.png'" 
                  :alt="activeCurve === 'pr' ? 'Precision Recall Curve' : 'F1 Score Curve'" 
                  class="img-fluid rounded border hover-zoom" 
                  style="max-height: 280px; object-fit: contain; width: 100%;" 
                />
              </div>
              <div class="card-footer bg-light p-2 small">
                <p class="text-muted text-center mb-0 extra-small" v-if="activeCurve === 'pr'">
                  Đường cong Precision-Recall cho 7 lớp. mAP@0.50 đạt 91.2% trên toàn bộ các lớp rác thải.
                </p>
                <p class="text-muted text-center mb-0 extra-small" v-else>
                  Đường cong F1-Confidence cho thấy điểm cân bằng F1 tối ưu đạt 0.89 tại ngưỡng ngắt ~0.472.
                </p>
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
              Lý Do Lựa Chọn Kiến Trúc YOLO11s + CBAM (100 Epochs)
            </h6>
            <div class="d-flex flex-column gap-3">
              <div class="p-3 bg-light rounded-3 border">
                <div class="d-flex align-items-center gap-2 mb-1 text-success fw-bold small">
                  <i class="bi bi-lightning-charge-fill"></i> Tốc độ suy luận siêu tốc (7.91 ms/frame GPU, ~28 ms CPU ONNX)
                </div>
                <p class="small text-muted mb-0">
                  Đạt ~126 FPS trên GPU T4 và đáp ứng mượt mà Real-time trên trình duyệt / CPU của nền tảng Cloud Render.
                </p>
              </div>

              <div class="p-3 bg-light rounded-3 border">
                <div class="d-flex align-items-center gap-2 mb-1 text-primary fw-bold small">
                  <i class="bi bi-bullseye"></i> Độ chính xác vượt trội (mAP@50 đạt 91.18%, mAP@50-95 đạt 78.82%)
                </div>
                <p class="small text-muted mb-0">
                  Cơ chế chú ý kênh và không gian CBAM giúp mô hình bắt biên dạng và phân biệt chính xác 7 loại rác thải phức tạp.
                </p>
              </div>

              <div class="p-3 bg-light rounded-3 border">
                <div class="d-flex align-items-center gap-2 mb-1 text-warning text-dark fw-bold small">
                  <i class="bi bi-shield-check"></i> Độ ổn định cao qua kiểm định 3-Fold (mAP@50: 90.87% ± 0.35%)
                </div>
                <p class="small text-muted mb-0">
                  Kiểm định chéo 3-Fold trên toàn bộ 26.048 ảnh khẳng định mô hình không bị quá khớp và tổng quát hóa xuất sắc.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Lightbox Modal xem ảnh biểu đồ phóng to -->
    <div 
      v-if="modalImage.show" 
      class="modal-backdrop-custom"
      @click.self="closeModalImage"
    >
      <div class="modal-dialog-custom">
        <div class="modal-content-custom">
          <div class="modal-header d-flex align-items-center justify-content-between p-3 border-bottom">
            <h6 class="modal-title fw-bold mb-0 text-dark">
              <i class="bi bi-image text-success me-2"></i>{{ modalImage.title }}
            </h6>
            <div class="d-flex align-items-center gap-2">
              <a :href="modalImage.src" target="_blank" download class="btn btn-sm btn-outline-success d-flex align-items-center gap-1">
                <i class="bi bi-download"></i> Tải ảnh gốc
              </a>
              <button type="button" class="btn-close" @click="closeModalImage"></button>
            </div>
          </div>
          <div class="modal-body p-3 text-center bg-white" style="max-height: 75vh; overflow: auto;">
            <img :src="modalImage.src" :alt="modalImage.title" class="img-fluid rounded border shadow-sm" style="max-height: 70vh; object-fit: contain;" />
          </div>
          <div class="modal-footer p-3 bg-light border-top">
            <p class="text-muted small mb-0 w-100 text-center">{{ modalImage.desc }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue';
import { Chart, registerables } from 'chart.js';
import StatCard from '../components/common/StatCard.vue';
import LoadingSpinner from '../components/common/LoadingSpinner.vue';
import ErrorState from '../components/common/ErrorState.vue';
import apiService from '../services/api';
import history100Data from '../assets/data/history_epochs_100.json';

Chart.register(...registerables);

const loading = ref(true);
const reloading = ref(false);
const reloadMessage = ref('');
const error = ref('');
const modelData = ref({
  modelInfo: {
    name: 'YOLO11s-CBAM Trash Classifier (Fold 1)',
    weightsFile: 'best.pt (17.4 MB)',
    datasetSize: 26048,
    trainingSplit: { train: 17366, val: 8682, kfold: '3-Fold CV' },
    trainingConfig: {
      epochs: 100,
      batchSize: 32,
      imgSize: '640x640',
      optimizer: 'AdamW (lr0=0.001, weight_decay=0.0005)'
    },
    metrics: {
      precision: 0.9305,
      recall: 0.8563,
      map50: 0.9118,
      map50_95: 0.7882,
      peakPrecision: 0.93295,
      peakRecall: 0.8563,
      peakMap50: 0.9118,
      peakMap50_95: 0.7884
    },
    historyEpochs: history100Data
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
  return epochs && epochs.length > 0 ? epochs : history100Data;
});

const bestEpoch = computed(() => {
  return allEpochs.value.find((e) => e.epoch === 100) || allEpochs.value[allEpochs.value.length - 1];
});

const displayedEpochs = computed(() => {
  if (showFullTable.value) return allEpochs.value;
  return allEpochs.value.slice(allEpochs.value.length - 15);
});

const activeCurve = ref('pr');
const modalImage = ref({
  show: false,
  src: '',
  title: '',
  desc: ''
});

function openModalImage(src, title, desc) {
  modalImage.value = { show: true, src, title, desc };
}

function closeModalImage() {
  modalImage.value.show = false;
}

function onKeyDown(e) {
  if (e.key === 'Escape' && modalImage.value.show) {
    closeModalImage();
  }
}

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
          historyEpochs: response.data.modelInfo?.historyEpochs?.length >= 10 ? response.data.modelInfo.historyEpochs : history100Data
        }
      };
    }
  } catch (err) {
    console.warn('Lỗi API getModelInfo, sử dụng dữ liệu cục bộ 100 epochs:', err);
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
  window.addEventListener('keydown', onKeyDown);
});

onUnmounted(() => {
  window.removeEventListener('keydown', onKeyDown);
});
</script>

<style scoped>
.text-purple {
  color: #8b5cf6 !important;
}

.btn-xs {
  padding: 0.15rem 0.5rem;
  font-size: 0.75rem;
  border-radius: 4px;
}

.cursor-pointer {
  cursor: pointer;
}

.hover-zoom {
  transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease;
}

.hover-zoom:hover {
  transform: scale(1.02);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
}

.extra-small {
  font-size: 0.76rem;
  line-height: 1.35;
}

.modal-backdrop-custom {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(15, 23, 42, 0.78);
  backdrop-filter: blur(5px);
  z-index: 1060;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
}

.modal-dialog-custom {
  max-width: 95vw;
  width: 950px;
  max-height: 92vh;
  display: flex;
  flex-direction: column;
}

.modal-content-custom {
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.35);
  display: flex;
  flex-direction: column;
}
</style>
