const historyRepo = require('../repositories/historyRepository');
const config = require('../config/config');

/**
 * Get aggregated statistics for the Dashboard
 */
exports.getStatistics = (req, res) => {
  try {
    const stats = historyRepo.getStatistics();

    return res.status(200).json({
      success: true,
      data: {
        ...stats,
        availableClasses: config.classes
      }
    });
  } catch (error) {
    console.error('[StatsController] getStatistics error:', error);
    return res.status(500).json({
      success: false,
      message: 'Không thể tổng hợp số liệu thống kê.',
      error: error.message
    });
  }
};
