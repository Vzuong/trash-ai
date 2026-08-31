/**
 * DEPRECATED - MOCK AI SERVICE REMOVED
 * All AI predictions are processed exclusively by real YOLO model.
 */
class DeprecatedMockAIService {
  async predict() {
    throw new Error('Mock AI is disabled in production. Use real YOLO service.');
  }
  async predictFrame() {
    throw new Error('Mock AI is disabled in production. Use real YOLO service.');
  }
}

module.exports = new DeprecatedMockAIService();
