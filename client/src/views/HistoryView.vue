<template>
  <div class="container-xl py-4">
    <!-- Header -->
    <div class="d-flex flex-column flex-md-row align-items-md-center justify-content-between gap-3 mb-4 pb-2 border-bottom">
      <div>
        <h2 class="fw-bold text-dark mb-1">
          <i class="bi bi-clock-history text-success me-2"></i>Lịch Sử Nhận Diện
        </h2>
        <p class="text-muted small mb-0">
          Tra cứu, lọc và xem lại toàn bộ các phiên phân loại rác thải từ hình ảnh và camera.
        </p>
      </div>

      <router-link to="/classify" class="btn btn-eco-primary btn-sm d-flex align-items-center gap-2">
        <i class="bi bi-plus-lg"></i>
        <span>Phân loại lượt mới</span>
      </router-link>
    </div>

    <!-- Filters & Search Toolbar -->
    <div class="eco-card p-3 mb-4">
      <div class="row g-2 align-items-center">
        <!-- Search -->
        <div class="col-md-4">
          <div class="input-group input-group-sm">
            <span class="input-group-text bg-white border-end-0 text-muted">
              <i class="bi bi-search"></i>
            </span>
            <input 
              v-model="searchQuery" 
              type="text" 
              class="form-control border-start-0 shadow-none" 
              placeholder="Tìm kiếm theo tên file, loại rác..."
              @input="debounceFetch"
            />
          </div>
        </div>

        <!-- Filter by Waste Class -->
        <div class="col-6 col-md-3">
          <select v-model="selectedClass" class="form-select form-select-sm shadow-none" @change="fetchHistory">
            <option value="all">Tất cả loại rác</option>
            <option value="plastic">Rác nhựa (Plastic)</option>
            <option value="paper">Rác giấy (Paper)</option>
            <option value="glass">Rác thủy tinh (Glass)</option>
            <option value="metal">Rác kim loại (Metal)</option>
            <option value="cardboard">Rác bìa carton (Cardboard)</option>
            <option value="organic">Rác hữu cơ (Organic)</option>
            <option value="battery">Rác pin (Battery)</option>
          </select>
        </div>

        <!-- Filter by Method -->
        <div class="col-6 col-md-3">
          <select v-model="selectedMethod" class="form-select form-select-sm shadow-none" @change="fetchHistory">
            <option value="all">Tất cả phương thức</option>
            <option value="image">Tải ảnh lên (Image)</option>
            <option value="webcam">Webcam Realtime</option>
          </select>
        </div>

        <!-- Sort -->
        <div class="col-md-2 text-end">
          <button @click="toggleSort" class="btn btn-outline-secondary btn-sm w-100 d-flex align-items-center justify-content-center gap-1">
            <i :class="['bi', sortOrder === 'desc' ? 'bi-sort-down' : 'bi-sort-up']"></i>
            <span>{{ sortOrder === 'desc' ? 'Mới nhất' : 'Cũ nhất' }}</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Content Table -->
    <div class="eco-card overflow-hidden">
      <LoadingSpinner v-if="loading" message="Đang tải danh sách lịch sử..." />
      <ErrorState v-else-if="error" :message="error" @retry="fetchHistory" />

      <div v-else-if="historyList.length > 0" class="table-responsive">
        <table class="table table-hover align-middle mb-0">
          <thead class="table-light">
            <tr>
              <th scope="col" style="width: 70px;">Ảnh</th>
              <th scope="col">Phương thức</th>
              <th scope="col">Loại rác</th>
              <th scope="col">Độ tin cậy</th>
              <th scope="col">Số object</th>
              <th scope="col">Thời gian suy luận</th>
              <th scope="col">Thời điểm</th>
              <th scope="col" class="text-end">Hành động</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in historyList" :key="item.id">
              <td>
                <div class="rounded-2 bg-light border d-flex align-items-center justify-content-center overflow-hidden" style="width: 46px; height: 46px;">
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
                <div class="d-flex align-items-center gap-2" style="min-width: 100px;">
                  <div class="progress flex-grow-1" style="height: 6px;">
                    <div class="progress-bar bg-success" :style="{ width: item.confidencePercent + '%' }"></div>
                  </div>
                  <span class="small fw-bold">{{ item.confidencePercent }}%</span>
                </div>
              </td>
              <td>
                <span class="badge bg-light text-dark border small">{{ item.totalObjects || 1 }} vật thể</span>
              </td>
              <td>
                <span class="small text-muted">{{ item.inferenceTime }} ms</span>
              </td>
              <td>
                <span class="small text-muted">{{ formatDate(item.createdAt) }}</span>
              </td>
              <td class="text-end">
                <div class="btn-group btn-group-sm">
                  <button @click="selectedItem = item" class="btn btn-outline-secondary" title="Xem chi tiết">
                    <i class="bi bi-eye"></i>
                  </button>
                  <button @click="deleteItem(item.id)" class="btn btn-outline-danger" title="Xóa">
                    <i class="bi bi-trash"></i>
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <EmptyState 
        v-else 
        title="Không tìm thấy lịch sử phù hợp" 
        message="Hãy thử thay đổi từ khóa tìm kiếm hoặc bộ lọc loại rác."
      />
    </div>

    <!-- Detail Modal -->
    <DetailModal 
      v-if="selectedItem" 
      :item="selectedItem" 
      @close="selectedItem = null" 
      @delete="deleteItem" 
    />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import LoadingSpinner from '../components/common/LoadingSpinner.vue';
import ErrorState from '../components/common/ErrorState.vue';
import EmptyState from '../components/common/EmptyState.vue';
import DetailModal from '../components/history/DetailModal.vue';
import apiService from '../services/api';

const loading = ref(true);
const error = ref('');
const historyList = ref([]);
const selectedItem = ref(null);

const searchQuery = ref('');
const selectedClass = ref('all');
const selectedMethod = ref('all');
const sortOrder = ref('desc');

let debounceTimer = null;

function debounceFetch() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    fetchHistory();
  }, 300);
}

function toggleSort() {
  sortOrder.value = sortOrder.value === 'desc' ? 'asc' : 'desc';
  fetchHistory();
}

async function fetchHistory() {
  loading.value = true;
  error.value = '';
  try {
    const params = {
      search: searchQuery.value,
      classCode: selectedClass.value,
      method: selectedMethod.value,
      sort: sortOrder.value
    };
    const response = await apiService.getHistory(params);
    if (response.success) {
      historyList.value = response.data.items || [];
    }
  } catch (err) {
    console.error('Lỗi tải lịch sử:', err);
    error.value = err.message || 'Không thể tải dữ liệu lịch sử.';
  } finally {
    loading.value = false;
  }
}

async function deleteItem(id) {
  if (!confirm('Bạn có chắc chắn muốn xóa bản ghi này?')) return;
  try {
    await apiService.deleteHistory(id);
    selectedItem.value = null;
    fetchHistory();
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
  fetchHistory();
});
</script>

<style scoped>
</style>
