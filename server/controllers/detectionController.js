const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const realYoloService = require('../services/realYoloService');
const historyRepo = require('../repositories/historyRepository');
const config = require('../config/config');

/**
 * Predict image uploaded via multipart form
 */
exports.predictImage = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'Vui lòng chọn hoặc tải lên một bức ảnh hợp lệ!'
      });
    }

    const file = req.file;
    const imageUrl = `/uploads/${file.filename}`;

    // Perform AI prediction using real YOLO11
    const predictionResult = await realYoloService.predict({
      filename: file.filename,
      originalname: file.originalname,
      path: file.path,
      mimetype: file.mimetype,
      size: file.size
    });

    const primary = predictionResult.primaryResult;

    // Automatically save to history
    const historyEntry = historyRepo.add({
      method: 'image',
      imageUrl,
      imageName: file.originalname || file.filename,
      primaryClass: primary ? primary.className : 'Chưa xác định',
      classCode: primary ? primary.classCode : 'other',
      category: primary ? primary.category : 'Rác thải',
      confidence: primary ? primary.confidence : 0,
      totalObjects: predictionResult.totalObjects,
      inferenceTime: predictionResult.inferenceTime,
      detections: predictionResult.detections
    });

    return res.status(200).json({
      success: true,
      message: 'Phân loại ảnh thành công!',
      data: {
        ...predictionResult,
        imageUrl,
        historyId: historyEntry.id
      }
    });
  } catch (error) {
    console.error('[DetectionController] predictImage error:', error);
    return res.status(500).json({
      success: false,
      message: 'Đã xảy ra lỗi trong quá trình phân loại ảnh.',
      error: error.message
    });
  }
};

/**
 * Predict webcam frame sent as base64 string
 */
exports.predictWebcamFrame = async (req, res) => {
  try {
    const { frame, saveToHistory = false } = req.body;

    if (!frame) {
      return res.status(400).json({
        success: false,
        message: 'Không tìm thấy dữ liệu khung hình (Frame data missing)!'
      });
    }

    // Perform realtime AI prediction using real YOLO11
    const predictionResult = await realYoloService.predictFrame(frame);

    let imageUrl = '';
    let historyId = null;

    // If user clicked "Chụp ảnh & Lưu", save frame to disk and add to history
    if (saveToHistory) {
      const base64Data = frame.replace(/^data:image\/\w+;base64,/, '');
      const filename = `webcam_${Date.now()}_${uuidv4().substring(0, 6)}.jpg`;
      const filePath = path.join(config.uploadDir, filename);

      fs.writeFileSync(filePath, Buffer.from(base64Data, 'base64'));
      imageUrl = `/uploads/${filename}`;

      const primary = predictionResult.primaryResult;
      const historyEntry = historyRepo.add({
        method: 'webcam',
        imageUrl,
        imageName: filename,
        primaryClass: primary ? primary.className : 'Rác nhựa',
        classCode: primary ? primary.classCode : 'plastic',
        category: primary ? primary.category : 'Rác tái chế',
        confidence: primary ? primary.confidence : 0.9,
        totalObjects: predictionResult.totalObjects,
        inferenceTime: predictionResult.inferenceTime,
        detections: predictionResult.detections
      });
      historyId = historyEntry.id;
    }

    return res.status(200).json({
      success: true,
      data: {
        ...predictionResult,
        imageUrl,
        historyId
      }
    });
  } catch (error) {
    console.error('[DetectionController] predictWebcamFrame error:', error);
    return res.status(500).json({
      success: false,
      message: 'Đã xảy ra lỗi trong quá trình nhận diện webcam.',
      error: error.message
    });
  }
};
