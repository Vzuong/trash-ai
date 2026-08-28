<template>
  <div class="container-xl py-4">
    <!-- Hero Section -->
    <div class="eco-hero mb-4">
      <div class="row align-items-center gy-4">
        <div class="col-lg-8">
          <div class="d-inline-flex align-items-center gap-2 px-3 py-1 bg-white rounded-pill border border-success-subtle shadow-sm mb-3">
            <span class="hud-dot"></span>
            <span class="small fw-semibold text-success">Trí tuệ nhân tạo & Thị giác máy tính</span>
          </div>
          <h1 class="display-6 fw-bold text-dark mb-2">Hệ thống AI Phân Loại Rác</h1>
          <p class="lead text-muted fs-6 mb-4" style="max-width: 600px;">
            Ứng dụng trí tuệ nhân tạo trong nhận diện và phân loại rác thải sinh hoạt theo thời gian thực, hỗ trợ phân loại chính xác và nâng cao hiệu quả tái chế bảo vệ môi trường.
          </p>
          <div class="d-flex flex-wrap gap-3">
            <router-link to="/classify" class="btn btn-eco-primary btn-lg d-flex align-items-center gap-2 shadow">
              <i class="bi bi-camera-fill"></i>
              <span>Bắt đầu phân loại</span>
            </router-link>
            <router-link to="/model" class="btn btn-eco-outline btn-lg d-flex align-items-center gap-2">
              <i class="bi bi-cpu"></i>
              <span>Xem thông số AI</span>
            </router-link>
          </div>
        </div>

        <div class="col-lg-4 text-center d-none d-lg-block">
          <div class="p-4 bg-white bg-opacity-75 rounded-4 shadow-sm border border-white d-inline-block">
            <div class="display-1 text-success mb-2">🌱</div>
            <h6 class="fw-bold text-dark mb-1">Môi trường Xanh - Sạch</h6>
            <span class="text-muted small">Phân loại đúng • Tái chế nhanh</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Loading / Error States -->
    <LoadingSpinner v-if="loading" message="Đang tải dữ liệu tổng quan thống kê..." />
    <ErrorState v-else-if="error" :message="error" @retry="fetchDashboardData" />

    <div v-else>
      <!-- Key Statistics Grid -->
      <div class="row g-3 mb-4">
        <div class="col-6 col-md-4 col-xl-2">
          <StatCard 
            title="Tổng phân loại" 
            :value="stats.totalClassifications || 0" 
            unit="lượt"
            icon="bi-bar-chart-fill" 
            icon-bg="#ecfdf5" 
            icon-color="#10b981"
            subtitle="Toàn thời gian"
          />
        </div>
        <div class="col-6 col-md-4 col-xl-2">
          <StatCard 
            title="Rác nhựa" 
            :value="stats.byClass?.plastic?.count || 0" 
            unit="vật thể"
            icon="bi-droplet-half" 
            icon-bg="#eff6ff" 
            icon-color="#3b82f6"
            :subtitle="`${stats.byClass?.plastic?.percent || 0}% tổng số`"
          />
        </div>
        <div class="col-6 col-md-4 col-xl-2">
          <StatCard 
            title="Rác giấy" 
            :value="stats.byClass?.paper?.count || 0" 
            unit="vật thể"
            icon="bi-file-earmark-text" 
            icon-bg="#fffbeb" 
            icon-color="#f59e0b"
            :subtitle="`${stats.byClass?.paper?.percent || 0}% tổng số`"
          />
        </div>
        <div class="col-6 col-md-4 col-xl-2">
          <StatCard 
            title="Rác thủy tinh" 
            :value="stats.byClass?.glass?.count || 0" 
            unit="vật thể"
            icon="bi-cup-straw" 
            icon-bg="#ecfeff" 
            icon-color="#06b6d4"
            :subtitle="`${stats.byClass?.glass?.percent || 0}% tổng số`"
          />
        </div>
        <div class="col-6 col-md-4 col-xl-2">
          <StatCard 
            title="Rác kim loại" 
            :value="stats.byClass?.metal?.count || 0" 
            unit="vật thể"
            icon="bi-hammer" 
            icon-bg="#f1f5f9" 
            icon-color="#64748b"
            :subtitle="`${stats.byClass?.metal?.percent || 0}% tổng số`"
          />
        </div>
        <div class="col-6 col-md-4 col-xl-2">
          <StatCard 
            title="Rác khác" 
            :value="(stats.byClass?.cardboard?.count || 0) + (stats.byClass?.organic?.count || 0) + (stats.byClass?.battery?.count || 0)" 
            unit="vật thể"
            icon="bi-box-seam" 
            icon-bg="#fff7ed" 
            icon-color="#d97706"
            subtitle="Carton, pin, hữu cơ"
          />
        </div>
      </div>

      <!-- Charts Row -->
      <div class="row g-4 mb-4">
        <!-- Doughnut Chart: Waste Proportion -->
        <div class="col-lg-5">
          <div class="eco-card h-100 p-3 p-md-4">
            <div class="d-flex align-items-center justify-content-between mb-3">
              <h6 class="fw-bold mb-0">Tỷ lệ từng loại rác đã nhận diện</h6>
              <span class="badge bg-light text-muted border">Phân bố %</span>
            </div>
            <div class="position-relative d-flex align-items-center justify-content-center" style="min-height: 260px;">
              <canvas ref="doughnutCanvas"></canvas>
            </div>
          </div>
        </div>

        <!-- Bar Chart: Quantities -->
        <div class="col-lg-7">
          <div class="eco-card h-100 p-3 p-md-4">
            <div class="d-flex align-items-center justify-content-between mb-3">
              <h6 class="fw-bold mb-0">Số lượng phân loại theo danh mục rác</h6>
              <span class="badge bg-light text-muted border">Số lượng</span>
            </div>
            <div class="position-relative" style="min-height: 260px;">
              <canvas ref="barCanvas"></canvas>
            </div>
          </div>
        </div>
      </div>

      <!-- Section: Recent Detections -->
      <div class="eco-card p-3 p-md-4 mb-4">
        <div class="d-flex align-items-center justify-content-between mb-3">
          <div class="d-flex align-items-center gap-2">
            <i class="bi bi-clock-history text-success fs-5"></i>
            <h6 class="fw-bold mb-0">Lần phân loại gần đây</h6>
          </div>
          <router-link to="/history" class="btn btn-outline-success btn-sm d-flex align-items-center gap-1">
            <span>Xem tất cả</span>
            <i class="bi bi-arrow-right"></i>
          </router-link>
        </div>

        <div v-if="stats.recentDetections && stats.recentDetections.length > 0" class="table-responsive">
          <table class="table table-hover align-middle mb-0">
            <thead class="table-light">
              <tr>
                <th scope="col" style="width: 70px;">Ảnh</th>
                <th scope="col">Phương thức</th>
                <th scope="col">Loại rác chính</th>
                <th scope="col">Độ tin cậy</th>
                <th scope="col">Thời gian suy luận</th>
                <th scope="col">Thời điểm</th>
                <th scope="col" class="text-end">Chi tiết</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in stats.recentDetections" :key="item.id">
                <td>
                  <div class="rounded-2 bg-light border d-flex align-items-center justify-content-center overflow-hidden" style="width: 48px; height: 48px;">
                    <img v-if="item.imageUrl" :src="item.imageUrl" :alt="item.imageName" class="w-100 h-100 object-fit-cover" @error="onImgError" />
                    <i v-else class="bi bi-image text-muted"></i>
                  </div>
                </td>
                <td>
                  <span class="badge bg-light text-dark border">
                    <i :class="['bi', item.method === 'webcam' ? 'bi-camera-video' : 'bi-upload', 'me-1']"></i>
                    {{ item.methodName }}
                  </span>
                </td>
                <td>
                  <span :class="['badge-waste', `badge-${item.classCode}`]">
                    {{ item.primaryClass }}
                  </span>
                </td>
                <td>
                  <div class="d-flex align-items-center gap-2">
                    <div class="progress flex-grow-1" style="height: 6px; width: 60px;">
                      <div class="progress-bar bg-success" :style="{ width: item.confidencePercent + '%' }"></div>
                    </div>
                    <span class="small fw-bold">{{ item.confidencePercent }}%</span>
                  </div>
                </td>
                <td>
                  <span class="small text-muted">{{ item.inferenceTime }} ms</span>
                </td>
                <td>
                  <span class="small text-muted">{{ formatDate(item.createdAt) }}</span>
                </td>
                <td class="text-end">
                  <button @click="selectedModalItem = item" class="btn btn-outline-secondary btn-sm">
                    <i class="bi bi-eye"></i>
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <EmptyState v-else title="Chưa có lượt phân loại nào" message="Hãy chọn tab 'Phân loại rác' để bắt đầu trải nghiệm nhận diện bằng hình ảnh hoặc webcam." />
      </div>
    </div>

    <!-- Detail Modal -->
    <DetailModal 
      v-if="selectedModalItem" 
      :item="selectedModalItem" 
      @close="selectedModalItem = null" 
      @delete="handleDeleteRecent"
    />
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue';
import { Chart, registerables } from 'chart.js';
import StatCard from '../components/common/StatCard.vue';
import LoadingSpinner from '../components/common/LoadingSpinner.vue';
import ErrorState from '../components/common/ErrorState.vue';
import EmptyState from '../components/common/EmptyState.vue';
import DetailModal from '../components/history/DetailModal.vue';
import apiService from '../services/api';

Chart.register(...registerables);

const loading = ref(true);
const error = ref('');
const stats = ref({});
const selectedModalItem = ref(null);

const doughnutCanvas = ref(null);
const barCanvas = ref(null);
let doughnutChart = null;
let barChart = null;

async function fetchDashboardData() {
  loading.value = true;
  error.value = '';
  try {
    const response = await apiService.getStatistics();
    if (response.success) {
      stats.value = response.data;
      loading.value = false;
      await nextTick();
      renderCharts();
    }
  } catch (err) {
    console.error('Lỗi tải dashboard:', err);
    error.value = err.message || 'Không thể kết nối tới máy chủ backend.';
    loading.value = false;
  }
}

function renderCharts() {
  if (!stats.value.byClass) return;

  const labels = [];
  const dataValues = [];
  const colors = [];

  Object.values(stats.value.byClass).forEach((c) => {
    labels.push(c.name);
    dataValues.push(c.count);
    colors.push(c.color || '#10b981');
  });

  // Doughnut Chart
  if (doughnutCanvas.value) {
    if (doughnutChart) doughnutChart.destroy();
    doughnutChart = new Chart(doughnutCanvas.value, {
      type: 'doughnut',
      data: {
        labels,
        datasets: [{
          data: dataValues,
          backgroundColor: colors,
          borderWidth: 2,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, font: { family: 'Inter', size: 11 } } }
        },
        cutout: '65%'
      }
    });
  }

  // Bar Chart
  if (barCanvas.value) {
    if (barChart) barChart.destroy();
    barChart = new Chart(barCanvas.value, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Số lượng vật thể',
          data: dataValues,
          backgroundColor: colors.map((c) => c + 'cc'),
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: { beginAtZero: true, grid: { color: '#f1f5f9' } },
          x: { grid: { display: false } }
        }
      }
    });
  }
}

async function handleDeleteRecent(id) {
  try {
    await apiService.deleteHistory(id);
    selectedModalItem.value = null;
    fetchDashboardData();
  } catch (err) {
    alert(err.message || 'Không thể xóa bản ghi.');
  }
}

function formatDate(isoStr) {
  if (!isoStr) return '--:--';
  const d = new Date(isoStr);
  return d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' }) + ' ' + d.toLocaleDateString('vi-VN');
}

function onImgError(e) {
  e.target.style.display = 'none';
}

onMounted(() => {
  fetchDashboardData();
});
</script>

<style scoped>
</style>
