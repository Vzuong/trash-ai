const AIService = require('./aiService');
const mockAIService = require('./mockAIService');

const PYTHON_YOLO_URL = process.env.YOLO_SERVICE_URL || 'http://127.0.0.1:5001';

class RealYOLOService extends AIService {
  constructor() {
    super();
    this.pythonUrl = PYTHON_YOLO_URL;
  }

  /**
   * Predict from uploaded image via real Python YOLO service
   */
  async predict(imageInfo) {
    try {
      const response = await fetch(`${this.pythonUrl}/predict_image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: imageInfo.path
        })
      });

      if (!response.ok) {
        throw new Error(`Python service responded with status ${response.status}`);
      }

      const result = await response.json();
      if (result.success) {
        return result;
      }
      throw new Error(result.message || 'Lỗi từ YOLO service');
    } catch (err) {
      console.warn('[RealYOLOService] Python YOLO service not available or error, falling back to mock:', err.message);
      return mockAIService.predict(imageInfo);
    }
  }

  /**
   * Predict from webcam frame via real Python YOLO service
   */
  async predictFrame(base64Frame) {
    try {
      const response = await fetch(`${this.pythonUrl}/predict_frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frame: base64Frame
        })
      });

      if (!response.ok) {
        throw new Error(`Python service status ${response.status}`);
      }

      const result = await response.json();
      if (result.success) {
        return result;
      }
      throw new Error(result.message || 'Lỗi frame YOLO');
    } catch (err) {
      return mockAIService.predictFrame(base64Frame);
    }
  }

  /**
   * Get dynamic model info from Python service
   */
  async getModelInfo() {
    try {
      const response = await fetch(`${this.pythonUrl}/model_info`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        return await response.json();
      }
    } catch (err) {
      console.warn('[RealYOLOService] Failed to fetch model info from Python service:', err.message);
    }
    return null;
  }

  /**
   * Reload model in Python service
   */
  async reloadModel() {
    try {
      const response = await fetch(`${this.pythonUrl}/reload_model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        return await response.json();
      }
      throw new Error(`Python service status ${response.status}`);
    } catch (err) {
      console.error('[RealYOLOService] Failed to reload model:', err.message);
      throw err;
    }
  }
}

module.exports = new RealYOLOService();
