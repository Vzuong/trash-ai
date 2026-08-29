import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Accept': 'application/json',
    'ngrok-skip-browser-warning': 'true'
  }
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const customError = {
      message: error.response?.data?.message || error.message || 'Lỗi kết nối máy chủ',
      status: error.response?.status || 500,
      data: error.response?.data
    };
    return Promise.reject(customError);
  }
);

export const apiService = {
  // Classification
  predictImage(formData) {
    return apiClient.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },

  predictWebcam(frameBase64, saveToHistory = false) {
    return apiClient.post('/predict/webcam', {
      frame: frameBase64,
      saveToHistory
    });
  },

  // Statistics
  getStatistics() {
    return apiClient.get('/statistics');
  },

  // History
  getHistory(params = {}) {
    return apiClient.get('/history', { params });
  },

  getHistoryById(id) {
    return apiClient.get(`/history/${id}`);
  },

  deleteHistory(id) {
    return apiClient.delete(`/history/${id}`);
  },

  // Model Info
  getModelInfo() {
    return apiClient.get('/model-info');
  },

  reloadModel() {
    return apiClient.post('/model-info/reload');
  },

  // Health
  checkHealth() {
    return apiClient.get('/health');
  }
};

export default apiService;
