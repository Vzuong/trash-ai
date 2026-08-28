const AIService = require('./aiService');
const config = require('../config/config');

class MockAIService extends AIService {
  constructor() {
    super();
    this.classes = config.classes;
  }

  /**
   * Helper to detect class intent from filename or generate diverse realistic objects
   */
  _inferClassFromFilename(filename = '') {
    const lower = filename.toLowerCase();
    if (lower.includes('plastic') || lower.includes('chai') || lower.includes('nhua') || lower.includes('bottle')) {
      return this.classes.find(c => c.code === 'plastic');
    }
    if (lower.includes('paper') || lower.includes('giay') || lower.includes('bao') || lower.includes('doc')) {
      return this.classes.find(c => c.code === 'paper');
    }
    if (lower.includes('glass') || lower.includes('thuy_tinh') || lower.includes('coc') || lower.includes('ly')) {
      return this.classes.find(c => c.code === 'glass');
    }
    if (lower.includes('metal') || lower.includes('kim_loai') || lower.includes('lon') || lower.includes('can')) {
      return this.classes.find(c => c.code === 'metal');
    }
    if (lower.includes('cardboard') || lower.includes('carton') || lower.includes('bia') || lower.includes('hop')) {
      return this.classes.find(c => c.code === 'cardboard');
    }
    if (lower.includes('battery') || lower.includes('pin') || lower.includes('sac')) {
      return this.classes.find(c => c.code === 'battery');
    }
    if (lower.includes('organic') || lower.includes('huu_co') || lower.includes('rau') || lower.includes('qua') || lower.includes('food')) {
      return this.classes.find(c => c.code === 'organic');
    }
    // Random selection
    const randomIndex = Math.floor(Math.random() * this.classes.length);
    return this.classes[randomIndex];
  }

  /**
   * Predict from uploaded image
   */
  async predict(imageInfo) {
    const startTime = Date.now();
    // Simulate real inference delay (25ms - 45ms)
    await new Promise(resolve => setTimeout(resolve, Math.floor(Math.random() * 20) + 25));

    const primaryClass = this._inferClassFromFilename(imageInfo.filename || imageInfo.originalname || '');
    const imgWidth = imageInfo.width || 640;
    const imgHeight = imageInfo.height || 480;

    const detections = [];
    
    // Main object detection with realistic bounding box
    const primaryConfidence = Number((0.88 + Math.random() * 0.10).toFixed(3)); // 88% - 98%
    const mainBox = {
      x1: Math.round(imgWidth * (0.15 + Math.random() * 0.1)),
      y1: Math.round(imgHeight * (0.12 + Math.random() * 0.1)),
      x2: Math.round(imgWidth * (0.75 + Math.random() * 0.15)),
      y2: Math.round(imgHeight * (0.80 + Math.random() * 0.12))
    };

    detections.push({
      id: 1,
      classCode: primaryClass.code,
      className: primaryClass.name,
      category: primaryClass.category,
      color: primaryClass.color,
      badgeClass: primaryClass.badgeClass,
      icon: primaryClass.icon,
      confidence: primaryConfidence,
      confidencePercent: Math.round(primaryConfidence * 100),
      bbox: mainBox,
      instruction: primaryClass.instruction,
      binColor: primaryClass.binColor
    });

    // 40% chance of detecting a secondary object (for multi-object demo)
    if (Math.random() > 0.6) {
      const otherClasses = this.classes.filter(c => c.code !== primaryClass.code);
      const secondaryClass = otherClasses[Math.floor(Math.random() * otherClasses.length)];
      const secConfidence = Number((0.78 + Math.random() * 0.12).toFixed(3));
      
      detections.push({
        id: 2,
        classCode: secondaryClass.code,
        className: secondaryClass.name,
        category: secondaryClass.category,
        color: secondaryClass.color,
        badgeClass: secondaryClass.badgeClass,
        icon: secondaryClass.icon,
        confidence: secConfidence,
        confidencePercent: Math.round(secConfidence * 100),
        bbox: {
          x1: Math.round(imgWidth * 0.65),
          y1: Math.round(imgHeight * 0.45),
          x2: Math.round(imgWidth * 0.92),
          y2: Math.round(imgHeight * 0.88)
        },
        instruction: secondaryClass.instruction,
        binColor: secondaryClass.binColor
      });
    }

    const inferenceTime = Date.now() - startTime;

    return {
      success: true,
      detections,
      primaryResult: detections[0],
      totalObjects: detections.length,
      inferenceTime,
      model: config.modelInfo.name,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Predict from webcam frame (base64)
   */
  async predictFrame(base64Frame) {
    const startTime = Date.now();
    // Ultra-fast simulated inference (15ms - 30ms)
    await new Promise(resolve => setTimeout(resolve, Math.floor(Math.random() * 15) + 15));

    // Choose class dynamically
    const randomClass = this.classes[Math.floor(Math.random() * this.classes.length)];
    const confidence = Number((0.85 + Math.random() * 0.12).toFixed(3));

    const detections = [
      {
        id: 1,
        classCode: randomClass.code,
        className: randomClass.name,
        category: randomClass.category,
        color: randomClass.color,
        badgeClass: randomClass.badgeClass,
        icon: randomClass.icon,
        confidence,
        confidencePercent: Math.round(confidence * 100),
        bbox: {
          x1: 140 + Math.floor(Math.sin(Date.now() / 1000) * 20),
          y1: 100 + Math.floor(Math.cos(Date.now() / 1000) * 15),
          x2: 480 + Math.floor(Math.sin(Date.now() / 1000) * 20),
          y2: 400 + Math.floor(Math.cos(Date.now() / 1000) * 15)
        },
        instruction: randomClass.instruction,
        binColor: randomClass.binColor
      }
    ];

    return {
      success: true,
      detections,
      primaryResult: detections[0],
      totalObjects: detections.length,
      inferenceTime: Date.now() - startTime,
      timestamp: new Date().toISOString()
    };
  }
}

module.exports = new MockAIService();
