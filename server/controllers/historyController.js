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
