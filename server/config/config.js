const path = require('path');

const config = {
  port: process.env.PORT || 5000,
  env: process.env.NODE_ENV || 'production',
  uploadDir: path.join(__dirname, '..', 'uploads'),
  dataDir: path.join(__dirname, '..', 'data'),
  
  // Waste classes definition
  classes: [
    {
      id: 0,
      code: 'battery',
      name: 'Rác pin',
      color: '#ef4444',
      badgeClass: 'bg-danger',
      icon: 'bi-battery-charging',
      category: 'Rác nguy hại',
      binColor: 'Thùng rác màu cam/đỏ (Rác nguy hại)',
      instruction: 'Tuyệt đối không vứt vào thùng rác chung hoặc đốt. Cần thu gom riêng gửi về điểm thu gom pin chuyên dụng.'
    },
    {
      id: 1,
      code: 'cardboard',
      name: 'Rác bìa carton',
      color: '#d97706',
      badgeClass: 'bg-warning text-dark',
      icon: 'bi-box-seam',
      category: 'Rác tái chế',
      binColor: 'Thùng rác màu vàng/xanh dương (Rác tái chế)',
      instruction: 'Gấp phẳng thùng carton, giữ sạch và khô ráo để chuyển đến các nhà máy tái chế giấy.'
    },
    {
      id: 2,
      code: 'paper',
      name: 'Rác giấy',
      color: '#f59e0b',
      badgeClass: 'bg-warning text-dark',
      icon: 'bi-file-earmark-text',
      category: 'Rác tái chế',
      binColor: 'Thùng rác màu vàng/xanh dương (Rác tái chế)',
      instruction: 'Giữ giấy khô ráo, không dính dầu mỡ thực phẩm. Có thể tái chế thành tập vở, khăn giấy.'
    },
    {
      id: 3,
      code: 'glass',
      name: 'Rác thủy tinh',
      color: '#06b6d4',
      badgeClass: 'bg-info text-dark',
      icon: 'bi-cup-straw',
      category: 'Rác tái chế',
      binColor: 'Thùng rác màu xanh dương (Rác tái chế)',
      instruction: 'Rửa sạch chai lọ thủy tinh, phân loại riêng đồ vỡ để đảm bảo an toàn cho nhân viên thu gom.'
    },
    {
      id: 4,
      code: 'metal',
      name: 'Rác kim loại',
      color: '#64748b',
      badgeClass: 'bg-secondary',
      icon: 'bi-hammer',
      category: 'Rác tái chế',
      binColor: 'Thùng rác màu xanh dương (Rác tái chế)',
      instruction: 'Ép xẹp lon nhôm/hộp kim loại sau khi đã rửa sạch đồ bên trong để tiết kiệm không gian lưu trữ.'
    },
    {
      id: 5,
      code: 'plastic',
      name: 'Rác nhựa',
      color: '#3b82f6',
      badgeClass: 'bg-primary',
      icon: 'bi-droplet-half',
      category: 'Rác tái chế',
      binColor: 'Thùng rác màu xanh dương (Rác tái chế)',
      instruction: 'Tráng sạch chất lỏng, tháo nắp và ép dẹp chai nhựa trước khi cho vào thùng rác tái chế.'
    },
    {
      id: 6,
      code: 'organic',
      name: 'Rác hữu cơ',
      color: '#10b981',
      badgeClass: 'bg-success',
      icon: 'bi-tree',
      category: 'Rác hữu cơ',
      binColor: 'Thùng rác màu xanh lá cây (Rác hữu cơ)',
      instruction: 'Bao gồm vỏ trái cây, rau củ quả, thức ăn thừa. Thích hợp ủ làm phân bón hữu cơ (Compost).'
    }
  ],

  // Model Specs & Training Information (Synced from 68 Epochs Balanced Training)
  modelInfo: {
    name: 'YOLO11s Trash Classifier',
    architecture: 'YOLO11s (Ultralytics)',
    task: 'Object Detection (Phát hiện & Phân loại đối tượng)',
    version: '2.0.0 (Balanced Dataset 7 Classes)',
    datasetSize: 26048,
    trainingSplit: {
      train: 18320,
      val: 4636,
      test: 3092
    },
    classesCount: 7,
    weightsFile: 'best.pt (72.5 MB)',
    trainingConfig: {
      epochs: 68,
      batchSize: 32,
      imgSize: '640x640',
      optimizer: 'AdamW (lr0=0.0005, cos_lr=True)',
      device: 'Google Colab GPU & Local (NVIDIA GeForce RTX 4060, 12GB RAM)',
      augmentations: 'Mosaic (1.0), Scale (0.5), Rotation (15.0), Label Smoothing (0.1), Dropout (0.1)'
    },
    metrics: {
      precision: 0.9032,
      recall: 0.8062,
      map50: 0.8125,
      map50_95: 0.6659,
      peakPrecision: 0.9036,
      peakRecall: 0.79988,
      peakMap50: 0.86306,
      peakMap50_95: 0.68607,
      avgInferenceTime: '~28 ms (GPU)'
    },
    rationale: [
      'Bộ dữ liệu được mở rộng và cân bằng hoàn hảo 7 lớp rác thải (18.320 ảnh train, 4.636 ảnh val, 3.092 ảnh test).',
      'Đạt chỉ số Precision 90.32%, Recall 80.62% và mAP@50 đạt 81.25% trên tập dữ liệu Test độc lập (3.092 ảnh).',
      'Khắc phục triệt để lỗi nhận nhầm lon kim loại thành pin với Precision lớp Pin (battery) đạt 98.0% và Thủy tinh (glass) đạt 96.4%.'
    ],
    historyEpochs: require('../data/history_epochs_68.json')
  }
};

module.exports = config;
