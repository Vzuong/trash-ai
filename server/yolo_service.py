import os
import io
import time
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Ultralytics not installed. Run: pip install ultralytics")

app = Flask(__name__)
CORS(app)

# Model state
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "best.pt")
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = None
model_metadata = {}

def load_yolo_model():
    global model, WEIGHTS_PATH, DEVICE, model_metadata
    
    # Priority list of weights paths
    candidates = [
        os.path.join(BASE_DIR, "best.pt"),
        os.path.join(BASE_DIR, "weights", "best.pt")
    ]
    
    import glob
    candidates.extend(glob.glob(os.path.join(BASE_DIR, "runs", "**", "weights", "best.pt"), recursive=True))
    candidates.append(os.path.join(BASE_DIR, "yolo11s.pt"))
    
    resolved_path = None
    for cand in candidates:
        if os.path.exists(cand):
            resolved_path = cand
            break
            
    if not resolved_path:
        resolved_path = "yolo11s.pt"
        
    WEIGHTS_PATH = resolved_path
    
    print(f"==================================================")
    print(f" 🚀 Loading Real YOLO11 Model: {WEIGHTS_PATH}")
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f" ⚡ Hardware Device: {DEVICE}")
    
    # Extract metadata from checkpoint if available
    model_metadata = {
        'weights_file': os.path.basename(WEIGHTS_PATH),
        'weights_path': WEIGHTS_PATH,
        'weights_size_bytes': os.path.getsize(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else 0,
        'device': DEVICE,
        'loaded_at': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        if os.path.exists(WEIGHTS_PATH):
            ckpt = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict):
                model_metadata['train_metrics'] = ckpt.get('train_metrics', {})
                model_metadata['train_args'] = ckpt.get('train_args', {})
                model_metadata['date'] = ckpt.get('date')
                model_metadata['version'] = ckpt.get('version')
    except Exception as e:
        print(f" [INFO] Could not read ckpt metadata: {e}")
        
    model = YOLO(WEIGHTS_PATH)
    print(f" ✅ YOLO11 Model loaded successfully with {len(model.names)} classes: {model.names}")
    print(f"==================================================")
    return model

# Initial load
load_yolo_model()

# Waste class mapping metadata
CLASS_META = {
    'battery': {
        'code': 'battery',
        'name': 'Rác pin',
        'category': 'Rác nguy hại',
        'color': '#ef4444',
        'icon': 'bi-battery-charging',
        'binColor': 'Thùng rác màu cam/đỏ (Rác nguy hại)',
        'instruction': 'Tuyệt đối không vứt vào thùng rác chung hoặc đốt. Cần thu gom riêng gửi về điểm thu gom pin chuyên dụng.'
    },
    'cardboard': {
        'code': 'cardboard',
        'name': 'Rác bìa carton',
        'category': 'Rác tái chế',
        'color': '#d97706',
        'icon': 'bi-box-seam',
        'binColor': 'Thùng rác màu vàng/xanh dương (Rác tái chế)',
        'instruction': 'Gấp phẳng thùng carton, giữ sạch và khô ráo để chuyển đến các nhà máy tái chế giấy.'
    },
    'paper': {
        'code': 'paper',
        'name': 'Rác giấy',
        'category': 'Rác tái chế',
        'color': '#f59e0b',
        'icon': 'bi-file-earmark-text',
        'binColor': 'Thùng rác màu vàng/xanh dương (Rác tái chế)',
        'instruction': 'Giữ giấy khô ráo, không dính dầu mỡ thực phẩm. Có thể tái chế thành tập vở, khăn giấy.'
    },
    'glass': {
        'code': 'glass',
        'name': 'Rác thủy tinh',
        'category': 'Rác tái chế',
        'color': '#06b6d4',
        'icon': 'bi-cup-straw',
        'binColor': 'Thùng rác màu xanh dương (Rác tái chế)',
        'instruction': 'Rửa sạch chai lọ thủy tinh, phân loại riêng đồ vỡ để đảm bảo an toàn cho nhân viên thu gom.'
    },
    'metal': {
        'code': 'metal',
        'name': 'Rác kim loại',
        'category': 'Rác tái chế',
        'color': '#64748b',
        'icon': 'bi-hammer',
        'binColor': 'Thùng rác màu xanh dương (Rác tái chế)',
        'instruction': 'Ép xẹp lon nhôm/hộp kim loại sau khi đã rửa sạch đồ bên trong để tiết kiệm không gian lưu trữ.'
    },
    'plastic': {
        'code': 'plastic',
        'name': 'Rác nhựa',
        'category': 'Rác tái chế',
        'color': '#3b82f6',
        'icon': 'bi-droplet-half',
        'binColor': 'Thùng rác màu xanh dương (Rác tái chế)',
        'instruction': 'Tráng sạch chất lỏng, tháo nắp và ép dẹp chai nhựa trước khi cho vào thùng rác tái chế.'
    },
    'organic': {
        'code': 'organic',
        'name': 'Rác hữu cơ',
        'category': 'Rác hữu cơ',
        'color': '#10b981',
        'icon': 'bi-tree',
        'binColor': 'Thùng rác màu xanh lá cây (Rác hữu cơ)',
        'instruction': 'Bao gồm vỏ trái cây, rau củ quả, thức ăn thừa. Thích hợp ủ làm phân bón hữu cơ (Compost).'
    }
}

def format_detections(result, img_w, img_h):
    detections = []
    if not result.boxes:
        return detections

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].item())
        conf_val = float(box.conf[0].item())
        raw_name = result.names.get(cls_id, f"class_{cls_id}").lower().strip()
        
        # Match class code
        meta = CLASS_META.get(raw_name, {
            'code': raw_name,
            'name': f'Rác {raw_name}',
            'category': 'Rác thải',
            'color': '#8b5cf6',
            'icon': 'bi-trash',
            'binColor': 'Thùng rác phân loại thông thường',
            'instruction': 'Vui lòng vệ sinh sạch sẽ và bỏ vào đúng thùng rác quy định.'
        })

        xyxy = box.xyxy[0].tolist()
        bbox = {
            'x1': int(xyxy[0]),
            'y1': int(xyxy[1]),
            'x2': int(xyxy[2]),
            'y2': int(xyxy[3])
        }
        bbox_norm = {
            'x1': max(0.0, min(1.0, float(xyxy[0] / max(1, img_w)))),
            'y1': max(0.0, min(1.0, float(xyxy[1] / max(1, img_h)))),
            'x2': max(0.0, min(1.0, float(xyxy[2] / max(1, img_w)))),
            'y2': max(0.0, min(1.0, float(xyxy[3] / max(1, img_h))))
        }

        detections.append({
            'id': i + 1,
            'classCode': meta['code'],
            'className': meta['name'],
            'category': meta['category'],
            'color': meta['color'],
            'icon': meta['icon'],
            'confidence': round(conf_val, 3),
            'confidencePercent': int(round(conf_val * 100)),
            'bbox': bbox,
            'bboxNorm': bbox_norm,
            'instruction': meta['instruction'],
            'binColor': meta['binColor']
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'model': WEIGHTS_PATH,
        'weights_file': os.path.basename(WEIGHTS_PATH),
        'device': DEVICE,
        'classes': list(model.names.values()) if model else []
    })

@app.route('/reload_model', methods=['GET', 'POST'])
def reload_model():
    try:
        load_yolo_model()
        return jsonify({
            'success': True,
            'message': f'Đã cập nhật và nạp lại mô hình thành công: {os.path.basename(WEIGHTS_PATH)}',
            'model': WEIGHTS_PATH,
            'device': DEVICE,
            'metadata': model_metadata
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Lỗi khi nạp lại mô hình: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'success': True,
        'model': WEIGHTS_PATH,
        'weights_file': os.path.basename(WEIGHTS_PATH),
        'device': DEVICE,
        'classes': list(model.names.values()) if model else [],
        'metadata': model_metadata
    })

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        start_time = time.time()
        
        # Check if file path is provided or file upload
        data = request.json or {}
        image_path = data.get('image_path')

        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
        elif 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream)
        else:
            return jsonify({'success': False, 'message': 'Không tìm thấy hình ảnh để xử lý!'}), 400

        img_w, img_h = img.size

        # Run real YOLO inference
        results = model.predict(
            source=img,
            conf=0.25,
            iou=0.45,
            agnostic_nms=True,
            imgsz=640,
            device=DEVICE,
            verbose=False
        )

        res = results[0]
        detections = format_detections(res, img_w, img_h)
        inference_time = int((time.time() - start_time) * 1000)

        primary = detections[0] if detections else {
            'className': 'Không phát hiện rác',
            'classCode': 'none',
            'category': 'Không xác định',
            'color': '#94a3b8',
            'confidence': 0.0,
            'confidencePercent': 0,
            'instruction': 'Không phát hiện thấy vật thể rác thải nào trong ảnh.',
            'binColor': 'Vui lòng kiểm tra lại góc chụp'
        }

        return jsonify({
            'success': True,
            'detections': detections,
            'primaryResult': primary,
            'totalObjects': len(detections),
            'inferenceTime': inference_time,
            'model': f'YOLO11s ({os.path.basename(WEIGHTS_PATH)})'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        start_time = time.time()
        data = request.json or {}
        frame_base64 = data.get('frame')

        if not frame_base64:
            return jsonify({'success': False, 'message': 'No frame data'}), 400

        # Decode base64
        if ',' in frame_base64:
            frame_base64 = frame_base64.split(',', 1)[1]
        img_bytes = base64.b64decode(frame_base64)
        img = Image.open(io.BytesIO(img_bytes))
        img_w, img_h = img.size

        # Real YOLO inference on webcam frame with tuned threshold (conf=0.35, iou=0.45)
        results = model.predict(
            source=img,
            conf=0.35,
            iou=0.45,
            agnostic_nms=True,
            imgsz=640,
            device=DEVICE,
            verbose=False
        )

        res = results[0]
        detections = format_detections(res, img_w, img_h)
        inference_time = int((time.time() - start_time) * 1000)

        primary = detections[0] if detections else None

        return jsonify({
            'success': True,
            'detections': detections,
            'primaryResult': primary,
            'totalObjects': len(detections),
            'inferenceTime': inference_time
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("YOLO_PORT", 5001))
    print(f" 🚀 YOLO Python Service running on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
