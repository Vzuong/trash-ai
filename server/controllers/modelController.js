const config = require('../config/config');
const realYoloService = require('../services/realYoloService');

/**
 * Get AI model metadata, performance metrics, and training parameters
 */
exports.getModelInfo = async (req, res) => {
  try {
    let liveModelInfo = { ...config.modelInfo };
    let liveClasses = config.classes;

    const pythonInfo = await realYoloService.getModelInfo();
    if (pythonInfo && pythonInfo.success) {
      liveModelInfo.status = 'online';
      liveModelInfo.device = pythonInfo.device || liveModelInfo.device;
      liveModelInfo.weightsFile = `${pythonInfo.weights_file || 'best.pt'} (${((pythonInfo.metadata?.weights_size_bytes || 19154586) / (1024 * 1024)).toFixed(1)} MB)`;
      liveModelInfo.weightsPath = pythonInfo.model;
      
      if (pythonInfo.metadata?.train_metrics) {
        const tm = pythonInfo.metadata.train_metrics;
        if (tm['metrics/mAP50(B)']) liveModelInfo.metrics.map50 = tm['metrics/mAP50(B)'];
        if (tm['metrics/precision(B)']) liveModelInfo.metrics.precision = tm['metrics/precision(B)'];
        if (tm['metrics/recall(B)']) liveModelInfo.metrics.recall = tm['metrics/recall(B)'];
        if (tm['metrics/mAP50-95(B)']) liveModelInfo.metrics.map50_95 = tm['metrics/mAP50-95(B)'];
      }
    }

    return res.status(200).json({
      success: true,
      data: {
        modelInfo: liveModelInfo,
        classes: liveClasses
      }
    });
  } catch (error) {
    console.error('[ModelController] getModelInfo error:', error);
    return res.status(500).json({
      success: false,
      message: 'Không thể lấy thông tin mô hình.',
      error: error.message
    });
  }
};

/**
 * Reload AI Model dynamically
 */
exports.reloadModel = async (req, res) => {
  try {
    const result = await realYoloService.reloadModel();
    return res.status(200).json({
      success: true,
      message: result?.message || 'Đã nạp lại mô hình AI mới nhất thành công!',
      data: result
    });
  } catch (error) {
    console.error('[ModelController] reloadModel error:', error);
    return res.status(500).json({
      success: false,
      message: 'Không thể nạp lại mô hình AI.',
      error: error.message
    });
  }
};

