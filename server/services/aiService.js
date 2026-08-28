/**
 * Abstract AI Service Interface
 * Allows seamless switching between Mock AI and real YOLO11 Python model
 */
class AIService {
  /**
   * Predict waste objects from an image buffer or file path
   * @param {Object} imageInfo - Information about the image (path, buffer, width, height)
   * @returns {Promise<Object>} Detection results
   */
  async predict(imageInfo) {
    throw new Error('Method predict() must be implemented');
  }

  /**
   * Predict waste objects from a webcam video frame (base64 string)
   * @param {string} base64Frame - Base64 encoded JPEG/PNG frame
   * @returns {Promise<Object>} Frame detection results
   */
  async predictFrame(base64Frame) {
    throw new Error('Method predictFrame() must be implemented');
  }
}

module.exports = AIService;
