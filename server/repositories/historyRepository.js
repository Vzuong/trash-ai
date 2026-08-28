const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const config = require('../config/config');

class HistoryRepository {
  constructor() {
    this.dataFile = path.join(config.dataDir, 'history.json');
    this.history = [];
    this._init();
  }

  _init() {
    try {
      if (!fs.existsSync(config.dataDir)) {
        fs.mkdirSync(config.dataDir, { recursive: true });
      }
      if (!fs.existsSync(config.uploadDir)) {
        fs.mkdirSync(config.uploadDir, { recursive: true });
      }

      if (fs.existsSync(this.dataFile)) {
        const raw = fs.readFileSync(this.dataFile, 'utf-8');
        this.history = JSON.parse(raw);
      } else {
        // Seed with realistic presentation items
        this.history = this._generateSeedData();
        this._save();
      }
    } catch (err) {
      console.error('[HistoryRepository] Init error:', err);
      this.history = this._generateSeedData();
    }
  }

  _generateSeedData() {
    try {
      if (fs.existsSync(this.dataFile)) {
        return JSON.parse(fs.readFileSync(this.dataFile, 'utf-8'));
      }
    } catch (e) {}
    
    const sampleTime = Date.now();
    return [
      {
        id: 'rec_plastic_101',
        method: 'image',
        methodName: 'Tải ảnh lên',
        imageUrl: '/uploads/sample_plastic.jpg',
        imageName: 'sample_plastic.jpg',
        primaryClass: 'Rác nhựa / Chai nhựa',
        classCode: 'plastic',
        category: 'Rác tái chế',
        confidence: 0.952,
        confidencePercent: 95,
        totalObjects: 1,
        inferenceTime: 28.5,
        createdAt: new Date(sampleTime - 1000 * 60 * 15).toISOString(),
        detections: [
          {
            id: 1,
            className: 'Rác nhựa / Chai nhựa',
            classCode: 'plastic',
            category: 'Rác tái chế',
            binColor: 'Thùng rác màu xanh lá/dương (Rác tái chế)',
            color: '#16a34a',
            icon: 'bi-droplet-half',
            confidence: 0.952,
            confidencePercent: 95,
            instruction: 'Rửa sạch, tháo nắp và ép xẹp chai nhựa để tiết kiệm không gian lưu trữ.',
            bbox: { x1: 94, y1: 34, x2: 163, y2: 224 }
          }
        ]
      },
      {
        id: 'rec_metal_102',
        method: 'webcam',
        methodName: 'Chụp từ Webcam',
        imageUrl: '/uploads/sample_metal.jpg',
        imageName: 'sample_metal.jpg',
        primaryClass: 'Rác kim loại / Lon',
        classCode: 'metal',
        category: 'Rác tái chế',
        confidence: 0.894,
        confidencePercent: 89,
        totalObjects: 1,
        inferenceTime: 27.8,
        createdAt: new Date(sampleTime - 1000 * 60 * 35).toISOString(),
        detections: [
          {
            id: 1,
            className: 'Rác kim loại / Lon',
            classCode: 'metal',
            category: 'Rác tái chế',
            binColor: 'Thùng rác màu xám/vàng (Rác tái chế)',
            color: '#4b5563',
            icon: 'bi-archive',
            confidence: 0.894,
            confidencePercent: 89,
            instruction: 'Rửa sạch, ép xẹp vỏ lon kim loại trước khi cho vào thùng rác tái chế.',
            bbox: { x1: 80, y1: 50, x2: 220, y2: 210 }
          }
        ]
      },
      {
        id: 'rec_battery_103',
        method: 'image',
        methodName: 'Tải ảnh lên',
        imageUrl: '/uploads/sample_battery.jpg',
        imageName: 'sample_battery.jpg',
        primaryClass: 'Pin / Pin điện tử',
        classCode: 'battery',
        category: 'Rác nguy hại',
        confidence: 0.935,
        confidencePercent: 94,
        totalObjects: 1,
        inferenceTime: 26.2,
        createdAt: new Date(sampleTime - 1000 * 60 * 60).toISOString(),
        detections: [
          {
            id: 1,
            className: 'Pin / Pin điện tử',
            classCode: 'battery',
            category: 'Rác nguy hại',
            binColor: 'Thùng rác màu cam/đỏ (Rác nguy hại)',
            color: '#dc2626',
            icon: 'bi-battery-charging',
            confidence: 0.935,
            confidencePercent: 94,
            instruction: 'Thu gom riêng trong hộp chuyên dụng, đem đến các điểm thu gom rác điện tử nguy hại.',
            bbox: { x1: 70, y1: 40, x2: 190, y2: 200 }
          }
        ]
      },
      {
        id: 'rec_glass_104',
        method: 'webcam',
        methodName: 'Chụp từ Webcam',
        imageUrl: '/uploads/sample_glass.jpg',
        imageName: 'sample_glass.jpg',
        primaryClass: 'Rác thủy tinh',
        classCode: 'glass',
        category: 'Rác tái chế',
        confidence: 0.961,
        confidencePercent: 96,
        totalObjects: 1,
        inferenceTime: 29.1,
        createdAt: new Date(sampleTime - 1000 * 60 * 95).toISOString(),
        detections: [
          {
            id: 1,
            className: 'Rác thủy tinh',
            classCode: 'glass',
            category: 'Rác tái chế',
            binColor: 'Thùng rác màu xanh dương (Rác tái chế)',
            color: '#0891b2',
            icon: 'bi-cup-straw',
            confidence: 0.961,
            confidencePercent: 96,
            instruction: 'Tráng sạch cặn bẩn, bọc cẩn thận nếu là thủy tinh vỡ để tránh gây thương tích.',
            bbox: { x1: 85, y1: 45, x2: 215, y2: 220 }
          }
        ]
      },
      {
        id: 'rec_cardboard_105',
        method: 'image',
        methodName: 'Tải ảnh lên',
        imageUrl: '/uploads/sample_cardboard.jpg',
        imageName: 'sample_cardboard.jpg',
        primaryClass: 'Bìa carton',
        classCode: 'cardboard',
        category: 'Rác tái chế',
        confidence: 0.754,
        confidencePercent: 75,
        totalObjects: 1,
        inferenceTime: 28.0,
        createdAt: new Date(sampleTime - 1000 * 60 * 140).toISOString(),
        detections: [
          {
            id: 1,
            className: 'Bìa carton',
            classCode: 'cardboard',
            category: 'Rác tái chế',
            binColor: 'Thùng rác màu vàng (Rác tái chế)',
            color: '#d97706',
            icon: 'bi-box-seam',
            confidence: 0.754,
            confidencePercent: 75,
            instruction: 'Gấp phẳng thùng carton, giữ khô ráo và bỏ vào thùng rác tái chế.',
            bbox: { x1: 60, y1: 50, x2: 240, y2: 210 }
          }
        ]
      },
      {
        id: 'rec_paper_106',
        method: 'webcam',
        methodName: 'Chụp từ Webcam',
        imageUrl: '/uploads/sample_paper.jpg',
        imageName: 'sample_paper.jpg',
        primaryClass: 'Rác giấy',
        classCode: 'paper',
        category: 'Rác tái chế',
        confidence: 0.948,
        confidencePercent: 95,
        totalObjects: 1,
        inferenceTime: 26.9,
        createdAt: new Date(sampleTime - 1000 * 60 * 200).toISOString(),
        detections: [
          {
            id: 1,
            className: 'Rác giấy',
            classCode: 'paper',
            category: 'Rác tái chế',
            binColor: 'Thùng rác màu vàng (Rác tái chế)',
            color: '#2563eb',
            icon: 'bi-newspaper',
            confidence: 0.948,
            confidencePercent: 95,
            instruction: 'Phân loại giấy sạch, không dính dầu mỡ để đưa vào tái chế.',
            bbox: { x1: 75, y1: 35, x2: 225, y2: 225 }
          }
        ]
      },
      {
        id: 'rec_organic_107',
        method: 'image',
        methodName: 'Tải ảnh lên',
        imageUrl: '/uploads/sample_organic.jpg',
        imageName: 'sample_organic.jpg',
        primaryClass: 'Rác hữu cơ',
        classCode: 'organic',
        category: 'Rác hữu cơ',
        confidence: 0.942,
        confidencePercent: 94,
        totalObjects: 1,
        inferenceTime: 27.5,
        createdAt: new Date(sampleTime - 1000 * 60 * 260).toISOString(),
        detections: [
          {
            id: 1,
            className: 'Rác hữu cơ',
            classCode: 'organic',
            category: 'Rác hữu cơ',
            binColor: 'Thùng rác màu xanh lá cây (Rác hữu cơ)',
            color: '#65a30d',
            icon: 'bi-egg-fried',
            confidence: 0.942,
            confidencePercent: 94,
            instruction: 'Bỏ vào thùng rác hữu cơ để ủ phân compost hoặc xử lý vi sinh bảo vệ môi trường.',
            bbox: { x1: 80, y1: 40, x2: 220, y2: 215 }
          }
        ]
      }
    ];
  }

  _save() {
    try {
      fs.writeFileSync(this.dataFile, JSON.stringify(this.history, null, 2), 'utf-8');
    } catch (err) {
      console.error('[HistoryRepository] Save error:', err);
    }
  }

  _readFromDisk() {
    try {
      if (fs.existsSync(this.dataFile)) {
        const raw = fs.readFileSync(this.dataFile, 'utf-8');
        this.history = JSON.parse(raw);
      }
    } catch (e) {}
  }

  /**
   * Add a new history entry
   */
  add(entry) {
    this._readFromDisk();
    const record = {
      id: entry.id || `rec_${uuidv4().substring(0, 8)}`,
      method: entry.method || 'image',
      methodName: entry.method === 'webcam' ? 'Chụp từ Webcam' : 'Tải ảnh lên',
      imageUrl: entry.imageUrl || '',
      imageName: entry.imageName || 'image.jpg',
      primaryClass: entry.primaryClass || 'Rác nhựa',
      classCode: entry.classCode || 'plastic',
      category: entry.category || 'Rác tái chế',
      confidence: entry.confidence || 0.9,
      confidencePercent: Math.round((entry.confidence || 0.9) * 100),
      totalObjects: entry.totalObjects || 1,
      inferenceTime: entry.inferenceTime || 30,
      createdAt: entry.createdAt || new Date().toISOString(),
      detections: entry.detections || []
    };

    this.history.unshift(record); // Prepend so newest is first
    this._save();
    return record;
  }

  /**
   * Query history with search, filter, and pagination
   */
  getAll({ search, classCode, method, limit = 50, offset = 0, sort = 'desc' } = {}) {
    this._readFromDisk();
    let result = [...this.history];

    // Filter by search
    if (search && search.trim()) {
      const q = search.trim().toLowerCase();
      result = result.filter(r => 
        r.primaryClass.toLowerCase().includes(q) ||
        r.imageName.toLowerCase().includes(q) ||
        r.methodName.toLowerCase().includes(q)
      );
    }

    // Filter by class
    if (classCode && classCode !== 'all') {
      result = result.filter(r => r.classCode === classCode);
    }

    // Filter by method (image / webcam)
    if (method && method !== 'all') {
      result = result.filter(r => r.method === method);
    }

    // Sort
    result.sort((a, b) => {
      const timeA = new Date(a.createdAt).getTime();
      const timeB = new Date(b.createdAt).getTime();
      return sort === 'asc' ? timeA - timeB : timeB - timeA;
    });

    const total = result.length;
    const paginated = result.slice(Number(offset), Number(offset) + Number(limit));

    return {
      total,
      items: paginated,
      limit: Number(limit),
      offset: Number(offset)
    };
  }

  /**
   * Get single history item by id
   */
  getById(id) {
    this._readFromDisk();
    return this.history.find(r => r.id === id);
  }

  /**
   * Delete item by id
   */
  delete(id) {
    this._readFromDisk();
    const index = this.history.findIndex(r => r.id === id);
    if (index !== -1) {
      const deleted = this.history.splice(index, 1)[0];
      this._save();
      return deleted;
    }
    return null;
  }

  /**
   * Calculate aggregated statistics for dashboard
   */
  getStatistics() {
    this._readFromDisk();
    const total = this.history.length;
    const byClass = {};
    const byMethod = { image: 0, webcam: 0 };
    let totalConfidence = 0;
    let totalInferenceTime = 0;

    // Initialize all classes with 0
    config.classes.forEach(c => {
      byClass[c.code] = {
        name: c.name,
        code: c.code,
        color: c.color,
        icon: c.icon,
        count: 0,
        percent: 0
      };
    });

    this.history.forEach(r => {
      if (byClass[r.classCode]) {
        byClass[r.classCode].count++;
      } else {
        byClass[r.classCode] = { name: r.primaryClass, code: r.classCode, count: 1, percent: 0 };
      }

      if (byMethod[r.method] !== undefined) {
        byMethod[r.method]++;
      }

      totalConfidence += (r.confidence || 0);
      totalInferenceTime += (r.inferenceTime || 0);
    });

    // Calculate percentages
    Object.keys(byClass).forEach(k => {
      byClass[k].percent = total > 0 ? Number(((byClass[k].count / total) * 100).toFixed(1)) : 0;
    });

    const avgConfidence = total > 0 ? Number((totalConfidence / total).toFixed(3)) : 0;
    const avgInferenceTime = total > 0 ? Math.round(totalInferenceTime / total) : 0;

    return {
      totalClassifications: total,
      avgConfidence,
      avgConfidencePercent: Math.round(avgConfidence * 100),
      avgInferenceTime,
      byClass,
      byMethod,
      recentDetections: this.history.slice(0, 5)
    };
  }
}

module.exports = new HistoryRepository();
