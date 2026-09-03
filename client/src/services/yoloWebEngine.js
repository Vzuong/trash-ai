import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime Web paths for WASM assets
ort.env.wasm.wasmPaths = '/wasm/';

// Configure thread count for WASM SIMD (safe check for SharedArrayBuffer)
try {
  if (typeof SharedArrayBuffer === 'undefined' || !(typeof window !== 'undefined' && window.crossOriginIsolated)) {
    ort.env.wasm.numThreads = 1;
  } else {
    ort.env.wasm.numThreads = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
  }
} catch (e) {
  ort.env.wasm.numThreads = 1;
}

export const CONF_THRESHOLD = 0.35;
export const IOU_THRESHOLD = 0.45;
export const IMAGE_SIZE = 640;

export const CLASS_NAMES = [
  'battery',
  'cardboard',
  'paper',
  'glass',
  'metal',
  'plastic',
  'organic'
];

export const CLASS_META = {
  battery: {
    code: 'battery',
    name: 'Rác pin',
    category: 'Rác nguy hại',
    color: '#ef4444',
    icon: 'bi-battery-charging',
    binColor: 'Thùng rác màu cam/đỏ (Rác nguy hại)',
    instruction: 'Tuyệt đối không vứt vào thùng rác chung hoặc đốt. Cần thu gom riêng gửi về điểm thu gom pin chuyên dụng.'
  },
  cardboard: {
    code: 'cardboard',
    name: 'Rác bìa carton',
    category: 'Rác tái chế',
    color: '#d97706',
    icon: 'bi-box-seam',
    binColor: 'Thùng rác màu vàng/xanh dương (Rác tái chế)',
    instruction: 'Gấp phẳng thùng carton, giữ sạch và khô ráo để chuyển đến các nhà máy tái chế giấy.'
  },
  paper: {
    code: 'paper',
    name: 'Rác giấy',
    category: 'Rác tái chế',
    color: '#f59e0b',
    icon: 'bi-file-earmark-text',
    binColor: 'Thùng rác màu vàng/xanh dương (Rác tái chế)',
    instruction: 'Giữ giấy khô ráo, không dính dầu mỡ thực phẩm. Có thể tái chế thành tập vở, khăn giấy.'
  },
  glass: {
    code: 'glass',
    name: 'Rác thủy tinh',
    category: 'Rác tái chế',
    color: '#06b6d4',
    icon: 'bi-cup-straw',
    binColor: 'Thùng rác màu xanh dương (Rác tái chế)',
    instruction: 'Rửa sạch chai lọ thủy tinh, phân loại riêng đồ vỡ để đảm bảo an toàn cho nhân viên thu gom.'
  },
  metal: {
    code: 'metal',
    name: 'Rác kim loại',
    category: 'Rác tái chế',
    color: '#64748b',
    icon: 'bi-hammer',
    binColor: 'Thùng rác màu xanh dương (Rác tái chế)',
    instruction: 'Ép xẹp lon nhôm/hộp kim loại sau khi đã rửa sạch đồ bên trong để tiết kiệm không gian lưu trữ.'
  },
  plastic: {
    code: 'plastic',
    name: 'Rác nhựa',
    category: 'Rác tái chế',
    color: '#3b82f6',
    icon: 'bi-droplet-half',
    binColor: 'Thùng rác màu xanh dương (Rác tái chế)',
    instruction: 'Tráng sạch chất lỏng, tháo nắp và ép dẹp chai nhựa trước khi cho vào thùng rác tái chế.'
  },
  organic: {
    code: 'organic',
    name: 'Rác hữu cơ',
    category: 'Rác hữu cơ',
    color: '#10b981',
    icon: 'bi-tree',
    binColor: 'Thùng rác màu xanh lá cây (Rác hữu cơ)',
    instruction: 'Bao gồm vỏ trái cây, rau củ quả, thức ăn thừa. Thích hợp ủ làm phân bón hữu cơ (Compost).'
  }
};

class YOLOWebEngine {
  constructor() {
    this.session = null;
    this.status = 'idle'; // 'idle' | 'loading' | 'ready' | 'error'
    this.activeProvider = 'none';
    this.loadError = null;
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCanvas.width = IMAGE_SIZE;
    this.offscreenCanvas.height = IMAGE_SIZE;
    this.offscreenCtx = this.offscreenCanvas.getContext('2d', { willReadFrequently: true });
    
    // Reusable Float32Array buffer for NCHW tensor (1, 3, 640, 640)
    this.tensorBuffer = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);
  }

  /**
   * Load YOLO11s ONNX model into Browser memory
   */
  async loadModel(modelUrl = '/models/best.onnx') {
    if (this.session && this.status === 'ready') {
      return this.session;
    }

    this.status = 'loading';
    this.loadError = null;
    console.log('[YOLOWebEngine] Bắt đầu tải mô hình AI vào trình duyệt:', modelUrl);

    // Try WebGPU first with high-performance hardware acceleration, then fallback to WASM
    const providersToTry = [
      { 
        name: 'webgpu', 
        label: 'WebGPU (Hardware Accelerated)',
        options: {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all'
        }
      },
      { 
        name: 'wasm', 
        label: 'WASM SIMD (CPU)',
        options: {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        }
      }
    ];

    for (const provider of providersToTry) {
      try {
        console.log(`[YOLOWebEngine] Đang khởi tạo session với backend: ${provider.name}...`);
        this.session = await ort.InferenceSession.create(modelUrl, provider.options);
        this.activeProvider = provider.label;
        this.status = 'ready';
        console.log(`✅ [YOLOWebEngine] Mô hình đã sẵn sàng! Backend: ${this.activeProvider}`);
        return this.session;
      } catch (err) {
        console.warn(`[YOLOWebEngine] Backend ${provider.name} không khả dụng:`, err.message);
      }
    }

    this.status = 'error';
    this.loadError = 'Không thể tải mô hình AI trên trình duyệt. Vui lòng kiểm tra hỗ trợ WebAssembly/WebGPU.';
    throw new Error(this.loadError);
  }

  /**
   * Preprocess any Image/Video/Canvas source with standard Letterbox (640x640)
   */
  preprocess(sourceElement) {
    const srcW = sourceElement.videoWidth || sourceElement.naturalWidth || sourceElement.width;
    const srcH = sourceElement.videoHeight || sourceElement.naturalHeight || sourceElement.height;

    if (!srcW || !srcH) {
      throw new Error('Nguồn hình ảnh không hợp lệ hoặc chưa sẵn sàng.');
    }

    const scale = Math.min(IMAGE_SIZE / srcW, IMAGE_SIZE / srcH);
    const unpadW = Math.round(srcW * scale);
    const unpadH = Math.round(srcH * scale);
    const padX = (IMAGE_SIZE - unpadW) / 2;
    const padY = (IMAGE_SIZE - unpadH) / 2;

    const ctx = this.offscreenCtx;
    // Fill letterbox background with YOLO standard grey (114, 114, 114)
    ctx.fillStyle = '#727272';
    ctx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);

    // Draw scaled source image centered
    ctx.drawImage(sourceElement, 0, 0, srcW, srcH, padX, padY, unpadW, unpadH);

    const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const pixels = imageData.data; // RGBA uint8 array

    // Convert RGBA HWC -> RGB CHW Float32Array in range [0.0, 1.0]
    const channelSize = IMAGE_SIZE * IMAGE_SIZE;
    const rOffset = 0;
    const gOffset = channelSize;
    const bOffset = channelSize * 2;

    const buffer = this.tensorBuffer;
    for (let i = 0; i < channelSize; i++) {
      const p = i * 4;
      buffer[rOffset + i] = pixels[p] / 255.0;
      buffer[gOffset + i] = pixels[p + 1] / 255.0;
      buffer[bOffset + i] = pixels[p + 2] / 255.0;
    }

    const tensor = new ort.Tensor('float32', buffer, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);

    return {
      tensor,
      srcW,
      srcH,
      scale,
      padX,
      padY
    };
  }

  /**
   * Run real-time inference directly on the client device
   */
  async detect(sourceElement, confThresh = CONF_THRESHOLD, iouThresh = IOU_THRESHOLD) {
    if (this.status !== 'ready' || !this.session) {
      throw new Error('Mô hình AI chưa được nạp hoặc chưa sẵn sàng.');
    }

    const tStart = performance.now();
    const { tensor, srcW, srcH, scale, padX, padY } = this.preprocess(sourceElement);

    // Run inference through ONNX Runtime Web
    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    const results = await this.session.run({ [inputName]: tensor });
    const outputTensor = results[outputName];

    // Decode YOLO11 Output: Shape [1, 11, 8400]
    const data = outputTensor.data;
    const numAnchors = 8400;
    const numClasses = CLASS_NAMES.length; // 7 classes

    const candidateBoxes = [];

    for (let i = 0; i < numAnchors; i++) {
      // Find class with highest confidence score
      let maxScore = -1;
      let bestClassId = -1;

      for (let c = 0; c < numClasses; c++) {
        const score = data[(4 + c) * numAnchors + i];
        if (score > maxScore) {
          maxScore = score;
          bestClassId = c;
        }
      }

      if (maxScore >= confThresh) {
        const xc = data[0 * numAnchors + i];
        const yc = data[1 * numAnchors + i];
        const w = data[2 * numAnchors + i];
        const h = data[3 * numAnchors + i];

        // Un-letterbox back to native camera source coordinates
        const x1 = Math.max(0, Math.min(srcW, (xc - w / 2 - padX) / scale));
        const y1 = Math.max(0, Math.min(srcH, (yc - h / 2 - padY) / scale));
        const x2 = Math.max(0, Math.min(srcW, (xc + w / 2 - padX) / scale));
        const y2 = Math.max(0, Math.min(srcH, (yc + h / 2 - padY) / scale));

        const bw = x2 - x1;
        const bh = y2 - y1;

        // Discard abnormal full-screen artifacts
        if (bw > 0 && bh > 0 && (bw * bh) / (srcW * srcH) < 0.90) {
          candidateBoxes.push({
            classId: bestClassId,
            score: maxScore,
            x1,
            y1,
            x2,
            y2,
            normX1: x1 / srcW,
            normY1: y1 / srcH,
            normX2: x2 / srcW,
            normY2: y2 / srcH
          });
        }
      }
    }

    // Apply Non-Maximum Suppression (NMS)
    const nmsDetections = this.applyNMS(candidateBoxes, iouThresh);

    // Format output detections
    const detections = nmsDetections.map((det, idx) => {
      const rawName = CLASS_NAMES[det.classId] || `class_${det.classId}`;
      const meta = CLASS_META[rawName] || {
        code: rawName,
        name: `Rác ${rawName}`,
        category: 'Rác thải',
        color: '#8b5cf6',
        icon: 'bi-trash',
        binColor: 'Thùng rác phân loại thông thường',
        instruction: 'Vui lòng phân loại đúng quy định.'
      };

      return {
        id: idx + 1,
        classCode: meta.code,
        className: meta.name,
        category: meta.category,
        color: meta.color,
        icon: meta.icon,
        confidence: Number(det.score.toFixed(3)),
        confidencePercent: Math.round(det.score * 100),
        bbox: {
          x1: Math.round(det.x1),
          y1: Math.round(det.y1),
          x2: Math.round(det.x2),
          y2: Math.round(det.y2)
        },
        bboxNorm: {
          x1: det.normX1,
          y1: det.normY1,
          x2: det.normX2,
          y2: det.normY2
        },
        binColor: meta.binColor,
        instruction: meta.instruction
      };
    });

    const inferenceTime = Math.round(performance.now() - tStart);

    return {
      success: true,
      detections,
      primaryResult: detections.length > 0 ? detections[0] : null,
      totalObjects: detections.length,
      inferenceTime,
      backend: this.activeProvider
    };
  }

  /**
   * Fast Non-Maximum Suppression
   */
  applyNMS(boxes, iouThresh) {
    if (boxes.length === 0) return [];

    // Sort by confidence descending
    boxes.sort((a, b) => b.score - a.score);

    const selected = [];
    const active = new Array(boxes.length).fill(true);

    for (let i = 0; i < boxes.length; i++) {
      if (!active[i]) continue;

      const current = boxes[i];
      selected.push(current);

      const areaA = (current.x2 - current.x1) * (current.y2 - current.y1);

      for (let j = i + 1; j < boxes.length; j++) {
        if (!active[j]) continue;

        const candidate = boxes[j];

        // Intersection
        const interX1 = Math.max(current.x1, candidate.x1);
        const interY1 = Math.max(current.y1, candidate.y1);
        const interX2 = Math.min(current.x2, candidate.x2);
        const interY2 = Math.min(current.y2, candidate.y2);

        if (interX2 > interX1 && interY2 > interY1) {
          const interArea = (interX2 - interX1) * (interY2 - interY1);
          const areaB = (candidate.x2 - candidate.x1) * (candidate.y2 - candidate.y1);
          const unionArea = areaA + areaB - interArea;
          const iou = interArea / unionArea;

          if (iou >= iouThresh) {
            active[j] = false;
          }
        }
      }
    }

    return selected;
  }
}

export const yoloWebEngine = new YOLOWebEngine();
export default yoloWebEngine;
