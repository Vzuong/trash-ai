const AIService = require('./aiService');

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
        const errorBody = await response.text();
        throw new Error(`Python AI service status ${response.status}: ${errorBody}`);
      }

      const result = await response.json();
      if (result.success) {
        return result;
      }
      throw new Error(result.message || 'Lỗi xử lý từ mô hình AI');
    } catch (err) {
      console.error('[AI ERROR] RealYOLOService predict failed:', err.message);
      throw new Error(`AI inference failed: ${err.message}`);
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
        const errorBody = await response.text();
        throw new Error(`Python AI service status ${response.status}: ${errorBody}`);
      }

      const result = await response.json();
      if (result.success) {
        return result;
      }
      throw new Error(result.message || 'Lỗi nhận diện frame từ AI');
    } catch (err) {
      console.error('[AI ERROR] RealYOLOService predictFrame failed:', err.message);
      throw new Error(`AI inference failed: ${err.message}`);
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
      throw new Error(`Python service status ${response.status}`);
    } catch (err) {
      console.error('[AI ERROR] Failed to fetch model info from Python service:', err.message);
      return null;
    }
  }

  /**
   * Check health of Python AI service
   */
  async getHealth() {
    try {
      const response = await fetch(`${this.pythonUrl}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        return await response.json();
      }
      return { status: 'error', message: `Status code ${response.status}` };
    } catch (err) {
      return { status: 'error', message: err.message };
    }
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
      console.error('[AI ERROR] Failed to reload model:', err.message);
      throw err;
    }
  }
}

module.exports = new RealYOLOService();
