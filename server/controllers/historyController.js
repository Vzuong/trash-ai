const historyRepo = require('../repositories/historyRepository');

/**
 * Get all history records with filtering and pagination
 */
exports.getHistory = (req, res) => {
  try {
    const { search, classCode, method, limit, offset, sort } = req.query;
    const data = historyRepo.getAll({ search, classCode, method, limit, offset, sort });

    return res.status(200).json({
      success: true,
      data
    });
  } catch (error) {
    console.error('[HistoryController] getHistory error:', error);
    return res.status(500).json({
      success: false,
      message: 'Không thể lấy dữ liệu lịch sử.',
      error: error.message
    });
  }
};

/**
 * Get single history item by id
 */
exports.getHistoryById = (req, res) => {
  try {
    const { id } = req.params;
    const item = historyRepo.getById(id);

    if (!item) {
      return res.status(404).json({
        success: false,
        message: 'Không tìm thấy bản ghi lịch sử này.'
      });
    }

    return res.status(200).json({
      success: true,
      data: item
    });
  } catch (error) {
    console.error('[HistoryController] getHistoryById error:', error);
    return res.status(500).json({
      success: false,
      message: 'Đã xảy ra lỗi khi tìm bản ghi.',
      error: error.message
    });
  }
};

/**
 * Delete a history record
 */
exports.deleteHistory = (req, res) => {
  try {
    const { id } = req.params;
    const deleted = historyRepo.delete(id);

    if (!deleted) {
      return res.status(404).json({
        success: false,
        message: 'Không tìm thấy bản ghi để xóa.'
      });
    }

    return res.status(200).json({
      success: true,
      message: 'Đã xóa bản ghi thành công!',
      data: deleted
    });
  } catch (error) {
    console.error('[HistoryController] deleteHistory error:', error);
    return res.status(500).json({
      success: false,
      message: 'Không thể xóa bản ghi lịch sử.',
      error: error.message
    });
  }
};

/**
 * Save client-side detection result to history
 */
exports.createHistory = (req, res) => {
  try {
    const fs = require('fs');
    const path = require('path');
    const { v4: uuidv4 } = require('uuid');
    const config = require('../config/config');

    const {
      image,
      method = 'webcam',
      primaryResult,
      totalObjects = 0,
      inferenceTime = 0,
      detections = []
    } = req.body;

    let imageUrl = '/uploads/sample_default.jpg';
    let filename = `webcam_${Date.now()}_${uuidv4().substring(0, 6)}.jpg`;

    if (image && typeof image === 'string' && image.includes('base64')) {
      const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
      const filePath = path.join(config.uploadDir, filename);
      fs.writeFileSync(filePath, Buffer.from(base64Data, 'base64'));
      imageUrl = `/uploads/${filename}`;
    }

    const primary = primaryResult || (detections.length > 0 ? detections[0] : null);

    const historyEntry = historyRepo.add({
      method,
      imageUrl,
      imageName: filename,
      primaryClass: primary ? primary.className : 'Không phát hiện rác',
      classCode: primary ? primary.classCode : 'none',
      category: primary ? primary.category : 'Không xác định',
      confidence: primary ? (primary.confidence || 0) : 0,
      totalObjects: totalObjects || detections.length,
      inferenceTime: inferenceTime || 0,
      detections: detections || []
    });

    return res.status(201).json({
      success: true,
      message: 'Đã lưu lịch sử nhận diện thành công!',
      data: {
        ...historyEntry,
        historyId: historyEntry.id
      }
    });
  } catch (error) {
    console.error('[HistoryController] createHistory error:', error);
    return res.status(500).json({
      success: false,
      message: 'Không thể lưu bản ghi lịch sử.',
      error: error.message
    });
  }
};
