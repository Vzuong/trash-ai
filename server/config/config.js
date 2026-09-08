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

  // Model Specs & Training Information (YOLO11s + CBAM Attention - Fold 1 Best Checkpoint)
  modelInfo: {
    name: 'YOLO11s-CBAM Trash Classifier (Fold 1)',
    architecture: 'YOLO11s + CBAM (Channel & Spatial Attention)',
    task: 'Object Detection (Phát hiện & Phân loại 7 nhóm rác)',
    version: '3.0.0 (3-Fold Cross-Validation / K=3)',
    datasetSize: 26048,
    trainingSplit: {
      train: 17366,
      val: 8682,
      kfold: '3-Fold (K=3, Stratified Shuffle)'
    },
    classesCount: 7,
    weightsFile: 'best.pt (17.4 MB)',
    trainingConfig: {
      epochs: 100,
      batchSize: 32,
      imgSize: '640x640',
      optimizer: 'AdamW (lr0=0.001, weight_decay=0.0005)',
      device: 'Google Colab GPU (Tesla T4, 16GB VRAM)',
      patience: 10,
      augmentations: 'MixUp (0.15), Copy-Paste (0.3), Erasing (0.4), Dropout (0.1)'
    },
    metrics: {
      precision: 0.9305,
      recall: 0.8563,
      map50: 0.9118,
      map50_95: 0.7882,
      peakPrecision: 0.93295,
      peakRecall: 0.8563,
      peakMap50: 0.9118,
      peakMap50_95: 0.7884,
      avgInferenceTime: '7.91 ms (~126 FPS GPU T4)',
      kfoldMeanMap50: '90.87% ± 0.35%',
      kfoldMeanMap50_95: '78.47% ± 0.29%'
    },
    rationale: [
      'Tích hợp module chú ý kép CBAM (Channel Attention + Spatial Attention) tại cổ mạng Neck giúp trích xuất nổi bật đặc trưng rác thải.',
      'Đạt chỉ số mAP@50 ấn tượng 91.18% (0.9118) và mAP@50-95 đạt 78.82% trên tập kiểm thử 8.682 ảnh Fold 1.',
      'Kiểm định chéo 3-Fold nghiêm ngặt trên toàn bộ 26.048 ảnh đạt mAP@50 trung bình 90.87% ± 0.35% với độ lệch chuẩn cực nhỏ (0.35%).',
      'Tốc độ suy luận đạt 7.91 ms (~126 FPS trên GPU T4, ~28 ms trên CPU ONNX), đáp ứng hoàn hảo thời gian thực.'
    ],
    historyEpochs: require('../data/history_epochs_100.json')
  }
};

module.exports = config;
