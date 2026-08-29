import os
import io
import time
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps

app = Flask(__name__)
CORS(app)

# Base directories and model candidates
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(BASE_DIR, "best.onnx")
PT_PATH = os.path.join(BASE_DIR, "best.pt")

session = None
pt_model = None
active_backend = "none"
model_metadata = {}

CLASS_NAMES = {
    0: 'battery',
    1: 'cardboard',
    2: 'paper',
    3: 'glass',
    4: 'metal',
    5: 'plastic',
    6: 'organic'
}

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

def load_ai_model():
    global session, pt_model, active_backend, model_metadata
    print("==================================================")
    
    # 1. Try ONNX Runtime (Ultra-lightweight ~100MB RAM)
    try:
        import onnxruntime as ort
        candidate_onnx = [
            ONNX_PATH,
            os.path.join(BASE_DIR, "weights", "best.onnx"),
            "best.onnx"
        ]
        resolved_onnx = next((p for p in candidate_onnx if os.path.exists(p)), None)
        
        if resolved_onnx:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            session = ort.InferenceSession(resolved_onnx, sess_options=opts, providers=['CPUExecutionProvider'])
            active_backend = "onnx"
            model_metadata = {
                'engine': 'ONNX Runtime (Ultra-Lightweight & Fast)',
                'weights_file': os.path.basename(resolved_onnx),
                'weights_path': resolved_onnx,
                'weights_size_mb': round(os.path.getsize(resolved_onnx) / (1024 * 1024), 2),
                'device': 'CPU (Optimized)',
                'loaded_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            print(f" 🚀 Loaded ONNX Model: {resolved_onnx} ({model_metadata['weights_size_mb']} MB)")
            print(f" ✅ Active Backend: ONNX Runtime (Memory safe for Cloud)")
            print("==================================================")
            return
    except Exception as e:
        print(f" [WARN] Could not load ONNX session: {e}")

    # 2. Fallback to Ultralytics PyTorch if available
    try:
        from ultralytics import YOLO
        candidate_pt = [
            PT_PATH,
            os.path.join(BASE_DIR, "weights", "best.pt"),
            "best.pt"
        ]
        resolved_pt = next((p for p in candidate_pt if os.path.exists(p)), "best.pt")
        pt_model = YOLO(resolved_pt)
        active_backend = "ultralytics"
        model_metadata = {
            'engine': 'Ultralytics PyTorch',
            'weights_file': os.path.basename(resolved_pt),
            'weights_path': resolved_pt,
            'weights_size_mb': round(os.path.getsize(resolved_pt) / (1024 * 1024), 2) if os.path.exists(resolved_pt) else 0,
            'device': 'CPU',
            'loaded_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        print(f" 🚀 Loaded PyTorch Model: {resolved_pt}")
        print(f" ✅ Active Backend: Ultralytics")
        print("==================================================")
    except Exception as e:
        print(f" [ERROR] Could not load any AI model: {e}")
        active_backend = "none"

load_ai_model()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2] # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def run_onnx_inference(img_bgr, orig_w=None, orig_h=None, conf_thresh=0.25, iou_thresh=0.45):
    h_curr, w_curr = img_bgr.shape[:2]
    if orig_w is None: orig_w = w_curr
    if orig_h is None: orig_h = h_curr
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_padded, r, (dw, dh) = letterbox(img_rgb)
    
    blob = img_padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})[0]
    preds = np.transpose(outputs[0]) # (8400, 11)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for pred in preds:
        xc, yc, w, h = pred[:4]
        scores = pred[4:]
        cls_id = int(np.argmax(scores))
        conf = float(scores[cls_id])
        if conf >= conf_thresh:
            x1 = (xc - w / 2 - dw) / r
            y1 = (yc - h / 2 - dh) / r
            x2 = (xc + w / 2 - dw) / r
            y2 = (yc + h / 2 - dh) / r
            
            x1 = max(0.0, min(float(w_curr), float(x1)))
            y1 = max(0.0, min(float(h_curr), float(y1)))
            x2 = max(0.0, min(float(w_curr), float(x2)))
            y2 = max(0.0, min(float(h_curr), float(y2)))
            
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            confidences.append(conf)
            class_ids.append(cls_id)
            
    if not boxes:
        return []
        
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
    detections = []
    
    if len(indices) > 0:
        for idx in indices:
            i = int(idx)
            bx = boxes[i]
            x1_c, y1_c, bw_c, bh_c = bx
            x2_c = x1_c + bw_c
            y2_c = y1_c + bh_c
            
            # Compute normalized coordinates relative to current processed image
            norm_x1 = max(0.0, min(1.0, float(x1_c / max(1, w_curr))))
            norm_y1 = max(0.0, min(1.0, float(y1_c / max(1, h_curr))))
            norm_x2 = max(0.0, min(1.0, float(x2_c / max(1, w_curr))))
            norm_y2 = max(0.0, min(1.0, float(y2_c / max(1, h_curr))))
            
            # Scaled box relative to original upload image dimensions
            orig_x1 = int(round(norm_x1 * orig_w))
            orig_y1 = int(round(norm_y1 * orig_h))
            orig_x2 = int(round(norm_x2 * orig_w))
            orig_y2 = int(round(norm_y2 * orig_h))
            
            cls_id = class_ids[i]
            conf_val = confidences[i]
            raw_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}").lower().strip()
            meta = CLASS_META.get(raw_name, {
                'code': raw_name,
                'name': f'Rác {raw_name}',
                'category': 'Rác thải',
                'color': '#8b5cf6',
                'icon': 'bi-trash',
                'binColor': 'Thùng rác phân loại thông thường',
                'instruction': 'Vui lòng bỏ vào đúng thùng rác quy định.'
            })
            
            bbox = {'x1': orig_x1, 'y1': orig_y1, 'x2': orig_x2, 'y2': orig_y2}
            bbox_norm = {
                'x1': norm_x1,
                'y1': norm_y1,
                'x2': norm_x2,
                'y2': norm_y2
            }
            
            detections.append({
                'id': len(detections) + 1,
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
            
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'backend': active_backend,
        'model': model_metadata.get('weights_file', 'unknown'),
        'classes': list(CLASS_NAMES.values()),
        'metadata': model_metadata
    })

@app.route('/reload_model', methods=['GET', 'POST'])
def reload_model():
    try:
        load_ai_model()
        return jsonify({
            'success': True,
            'message': f'Đã nạp lại mô hình thành công ({active_backend})',
            'metadata': model_metadata
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'success': True,
        'backend': active_backend,
        'classes': list(CLASS_NAMES.values()),
        'metadata': model_metadata
    })

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        start_time = time.time()
        data = request.json or {}
        image_path = data.get('image_path')
        
        pil_img = None
        if image_path and os.path.exists(image_path):
            pil_img = Image.open(image_path)
        elif 'image' in request.files:
            pil_img = Image.open(request.files['image'].stream)
            
        if pil_img is None:
            return jsonify({'success': False, 'message': 'Không tìm thấy hình ảnh hợp lệ!'}), 400
            
        # Fix EXIF orientation for smartphone photos (rotate portrait/landscape correctly)
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass
            
        pil_img = pil_img.convert('RGB')
        orig_w, orig_h = pil_img.size
        
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Memory protection: resize giant images (> 1280px) to prevent cloud OOM
        h, w = img_bgr.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1280:
            scale = 1280.0 / max_dim
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        if active_backend == "onnx" and session is not None:
            detections = run_onnx_inference(img_bgr, orig_w=orig_w, orig_h=orig_h, conf_thresh=0.25, iou_thresh=0.45)
        else:
            detections = []
            
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
            'model': f"YOLO11s ONNX ({model_metadata.get('weights_file', 'best.onnx')})"
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
            
        if ',' in frame_base64:
            frame_base64 = frame_base64.split(',', 1)[1]
            
        img_bytes = base64.b64decode(frame_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({'success': False, 'message': 'Invalid frame data'}), 400
            
        h_f, w_f = img_bgr.shape[:2]
        
        # Fast ONNX inference on webcam frame (conf=0.30, iou=0.45)
        if active_backend == "onnx" and session is not None:
            detections = run_onnx_inference(img_bgr, orig_w=w_f, orig_h=h_f, conf_thresh=0.30, iou_thresh=0.45)
        else:
            detections = []
            
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
    print(f" 🚀 Ultra-Lightweight YOLO ONNX Service running on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
