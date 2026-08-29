const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');
const config = require('./config/config');
const apiRoutes = require('./routes/api');

const app = express();

// Ensure uploads and data directories exist
if (!fs.existsSync(config.uploadDir)) {
  fs.mkdirSync(config.uploadDir, { recursive: true });
}
if (!fs.existsSync(config.dataDir)) {
  fs.mkdirSync(config.dataDir, { recursive: true });
}

// Middlewares
app.use(cors());
app.use(morgan('dev'));
app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ extended: true, limit: '25mb' }));

// Serve static uploads & AI models
app.use('/uploads', express.static(config.uploadDir));
app.get('/best.onnx', (req, res) => {
  const candidates = [
    path.join(__dirname, '../best.onnx'),
    path.join(__dirname, 'best.onnx'),
    '/app/best.onnx',
    path.join(process.cwd(), 'best.onnx')
  ];
  const modelPath = candidates.find(p => fs.existsSync(p));
  if (modelPath) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
    return res.sendFile(path.resolve(modelPath));
  }
  res.status(404).send('Model not found');
});

// API Routes
app.use('/api', apiRoutes);

// Health Check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'online',
    systemName: 'Hệ thống AI Phân Loại Rác',
    version: '2.0.0',
    timestamp: new Date().toISOString()
  });
});

// Serve Frontend SPA in Production
const clientDistPath = path.join(__dirname, '../client/dist');
if (fs.existsSync(clientDistPath)) {
  app.use(express.static(clientDistPath));
  app.get('*', (req, res, next) => {
    if (req.path.startsWith('/api') || req.path.startsWith('/uploads')) {
      return next();
    }
    res.sendFile(path.join(clientDistPath, 'index.html'));
  });
}

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('[ServerError]', err);
  res.status(err.status || 500).json({
    success: false,
    message: err.message || 'Lỗi hệ thống máy chủ nội bộ',
    error: config.env === 'development' ? err.stack : undefined
  });
});

// Start Server
const server = app.listen(config.port, () => {
  console.log('====================================================');
  console.log(` 🌱 HỆ THỐNG AI PHÂN LOẠI RÁC - BACKEND SERVER`);
  console.log(` 🚀 Server running at: http://localhost:${config.port}`);
  console.log(` 📡 Health Check:     http://localhost:${config.port}/api/health`);
  console.log('====================================================');
});

module.exports = { app, server };
