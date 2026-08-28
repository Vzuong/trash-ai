const express = require('express');
const multer = require('multer');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const config = require('../config/config');

const detectionController = require('../controllers/detectionController');
const historyController = require('../controllers/historyController');
const statsController = require('../controllers/statsController');
const modelController = require('../controllers/modelController');

const router = express.Router();

// Configure multer storage for uploaded images
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, config.uploadDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase() || '.jpg';
    const cleanName = path.basename(file.originalname, ext).replace(/[^a-zA-Z0-9_-]/g, '_');
    cb(null, `img_${Date.now()}_${cleanName}_${uuidv4().substring(0, 6)}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 15 * 1024 * 1024 }, // 15MB
  fileFilter: (req, file, cb) => {
    const allowed = /jpeg|jpg|png|webp|bmp/i;
    const isMimeValid = allowed.test(file.mimetype);
    const isExtValid = allowed.test(path.extname(file.originalname).toLowerCase());
    if (isMimeValid || isExtValid) {
      return cb(null, true);
    }
    cb(new Error('Chỉ chấp nhận file ảnh định dạng JPG, JPEG, PNG, WEBP, BMP!'));
  }
});

// Detection Routes
router.post('/predict', upload.single('image'), detectionController.predictImage);
router.post('/predict/webcam', detectionController.predictWebcamFrame);

// History Routes
router.get('/history', historyController.getHistory);
router.get('/history/:id', historyController.getHistoryById);
router.delete('/history/:id', historyController.deleteHistory);

// Statistics Routes
router.get('/statistics', statsController.getStatistics);

// Model Info Routes
router.get('/model-info', modelController.getModelInfo);
router.post('/model-info/reload', modelController.reloadModel);

module.exports = router;
