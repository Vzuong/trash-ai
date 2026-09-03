import os
import sys
import time
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps

try:
    import onnxruntime as ort
except Exception:
    ort = None

app = Flask(__name__)
CORS(app)

# Base directories and model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PT_PATH = os.path.join(BASE_DIR, "best.pt")
ONNX_PATH = os.path.join(BASE_DIR, "best.onnx")

# AI Reference Parameters (Source of Truth: test_webcam.py)
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640

# Class Definitions
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

model_instance = None
onnx_session = None
active_backend = "none"
model_metadata = {}

def load_ai_model():
    global model_instance, onnx_session, active_backend, model_metadata
    print("\n" + "=" * 60)
    print(" [AI] Loading AI trash recognition model...")

    # 1. Check if NVIDIA CUDA GPU is available
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass

    if cuda_available:
        try:
            from ultralytics import YOLO
            import torch
            candidate_pt = [PT_PATH, os.path.join(BASE_DIR, "weights", "best.pt"), "best.pt"]
            resolved_pt = next((p for p in candidate_pt if os.path.isfile(p)), None)
            if resolved_pt:
                model_instance = YOLO(resolved_pt)
                active_backend = "ultralytics"
                gpu_name = torch.cuda.get_device_name(0)
                model_metadata = {
                    'backend': 'Ultralytics PyTorch (GPU CUDA)',
                    'weights_file': os.path.basename(resolved_pt),
                    'weights_path': resolved_pt,
                    'device': f'GPU CUDA:0 ({gpu_name})',
                    'loaded_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                print(f" [AI] Model loaded: {os.path.basename(resolved_pt)}")
                print(f" [AI] Backend: Ultralytics PyTorch")
                print(f" [AI] Device: {model_metadata['device']}")
                print("=" * 60 + "\n")
                return
        except Exception as e:
            print(f" [AI WARN] Could not load GPU PyTorch: {e}")

    # 2. If no CUDA GPU, load ONNX Runtime CPU
    if ort is not None and os.path.isfile(ONNX_PATH):
        try:
            print(" [AI INFO] Loading ONNX Runtime with CPUExecutionProvider...")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            onnx_session = ort.InferenceSession(ONNX_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
            active_backend = "onnx"
            model_metadata = {
                'backend': 'ONNX Runtime CPU',
                'weights_file': os.path.basename(ONNX_PATH),
                'weights_path': ONNX_PATH,
                'device': 'CPU (Optimized 2-Thread)',
                'loaded_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            print(" [AI SUCCESS] Loaded ONNX Runtime CPU successfully!")
            print("=" * 60 + "\n")
            return
        except Exception as e:
            print(f" [AI WARN] Could not load ONNX Runtime CPU: {e}")

    # 3. Model load failure
    active_backend = "error"
    print(" [AI ERROR] Failed to load any AI model (best.onnx or best.pt)!")
    print("=" * 60 + "\n")

load_ai_model()

# ==============================================================================
# Helper Functions for Inference
# ==============================================================================

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

def format_ultralytics_results(results, img_w, img_h):
    """Format standard Ultralytics YOLO Results object (Matches test_webcam.py)"""
    detections = []
    if not results.boxes or len(results.boxes) == 0:
        return detections

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0].item())
        conf_val = float(box.conf[0].item())
        raw_name = results.names.get(cls_id, f"class_{cls_id}").lower().strip()
        meta = CLASS_META.get(raw_name, {
            'code': raw_name,
            'name': f'Rác {raw_name}',
            'category': 'Rác thải',
            'color': '#8b5cf6',
            'icon': 'bi-trash',
            'binColor': 'Thùng rác phân loại thông thường',
            'instruction': 'Vui lòng bỏ vào đúng thùng rác quy định.'
        })
        
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        
        bbox = {
            'x1': int(round(x1)),
            'y1': int(round(y1)),
            'x2': int(round(x2)),
            'y2': int(round(y2))
        }
        bbox_norm = {
            'x1': max(0.0, min(1.0, float(x1 / max(1, img_w)))),
            'y1': max(0.0, min(1.0, float(y1 / max(1, img_h)))),
            'x2': max(0.0, min(1.0, float(x2 / max(1, img_w)))),
            'y2': max(0.0, min(1.0, float(y2 / max(1, img_h))))
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

    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections

def run_onnx_inference(img_bgr, orig_w, orig_h, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    """Accurate ONNX Runtime Inference matching Ultralytics decoding"""
    h_curr, w_curr = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_padded, r, (dw, dh) = letterbox(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    
    blob = img_padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]
    
    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: blob})[0]
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
            
            norm_x1 = max(0.0, min(1.0, float(x1_c / max(1, w_curr))))
            norm_y1 = max(0.0, min(1.0, float(y1_c / max(1, h_curr))))
            norm_x2 = max(0.0, min(1.0, float(x2_c / max(1, w_curr))))
            norm_y2 = max(0.0, min(1.0, float(y2_c / max(1, h_curr))))
            
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

# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================

@app.route('/health', methods=['GET'])
def health():
    is_ready = active_backend in ["ultralytics", "onnx", "client_webgpu_only"]
    return jsonify({
        'status': 'ready' if is_ready else 'error',
        'backend': model_metadata.get('backend', 'none'),
        'model': model_metadata.get('weights_file', 'none'),
        'device': model_metadata.get('device', 'none'),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }), (200 if is_ready else 503)

@app.route('/model_info', methods=['GET'])
def model_info():
    is_ready = active_backend in ["ultralytics", "onnx", "client_webgpu_only"]
    return jsonify({
        'success': is_ready,
        'backend': model_metadata.get('backend', 'none'),
        'model': model_metadata.get('weights_file', 'none'),
        'device': model_metadata.get('device', 'none'),
        'classes': list(CLASS_NAMES.values()),
        'metadata': model_metadata
    }), (200 if is_ready else 503)

@app.route('/reload_model', methods=['POST'])
def reload_model():
    try:
        load_ai_model()
        is_ready = active_backend in ["ultralytics", "onnx", "client_webgpu_only"]
        if is_ready:
            return jsonify({
                'success': True,
                'message': f"Đã nạp lại mô hình thành công ({model_metadata.get('backend')})",
                'metadata': model_metadata
            })
        return jsonify({'success': False, 'message': 'Không thể nạp mô hình'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if active_backend not in ["ultralytics", "onnx"]:
        return jsonify({'success': False, 'message': 'AI Model is not ready or loaded'}), 503

    start_time = time.time()
    try:
        data = request.json or {}
        image_path = data.get('image_path')
        
        pil_img = None
        if image_path and os.path.isfile(image_path):
            pil_img = Image.open(image_path)
        elif 'image' in request.files:
            pil_img = Image.open(request.files['image'].stream)
            
        if pil_img is None:
            return jsonify({'success': False, 'message': 'Không tìm thấy hình ảnh hợp lệ!'}), 400
            
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass
            
        pil_img = pil_img.convert('RGB')
        orig_w, orig_h = pil_img.size
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Inference
        if active_backend == "ultralytics" and model_instance is not None:
            res = model_instance.predict(
                source=img_bgr,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False
            )[0]
            detections = format_ultralytics_results(res, orig_w, orig_h)
        elif active_backend == "onnx" and onnx_session is not None:
            detections = run_onnx_inference(img_bgr, orig_w, orig_h, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD)
        else:
            detections = []
            
        inference_time = int((time.time() - start_time) * 1000)
        print(f"[AI] Image inference: {len(detections)} objects detected in {inference_time}ms")
        
        primary = detections[0] if len(detections) > 0 else None
        
        return jsonify({
            'success': True,
            'detections': detections,
            'primaryResult': primary,
            'totalObjects': len(detections),
            'inferenceTime': inference_time,
            'model': model_metadata.get('weights_file', 'YOLO11')
        })
    except Exception as e:
        print(f"[AI ERROR] predict_image failed: {e}", file=sys.stderr)
        return jsonify({'success': False, 'message': f"AI inference failed: {str(e)}"}), 500

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    if active_backend not in ["ultralytics", "onnx"]:
        return jsonify({'success': False, 'message': 'AI Model is not ready or loaded'}), 503

    start_time = time.time()
    try:
        data = request.json or {}
        frame_base64 = data.get('frame')
        
        if not frame_base64:
            return jsonify({'success': False, 'message': 'No frame data provided'}), 400
            
        if ',' in frame_base64:
            frame_base64 = frame_base64.split(',', 1)[1]
            
        img_bytes = base64.b64decode(frame_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({'success': False, 'message': 'Invalid frame data decoded'}), 400
            
        h_f, w_f = img_bgr.shape[:2]
        
        # Inference
        if active_backend == "ultralytics" and model_instance is not None:
            res = model_instance.predict(
                source=img_bgr,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False
            )[0]
            detections = format_ultralytics_results(res, w_f, h_f)
        elif active_backend == "onnx" and onnx_session is not None:
            detections = run_onnx_inference(img_bgr, w_f, h_f, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD)
        else:
            detections = []
            
        inference_time = int((time.time() - start_time) * 1000)
        primary = detections[0] if len(detections) > 0 else None
        
        return jsonify({
            'success': True,
            'detections': detections,
            'primaryResult': primary,
            'totalObjects': len(detections),
            'inferenceTime': inference_time
        })
    except Exception as e:
        print(f"[AI ERROR] predict_frame failed: {e}", file=sys.stderr)
        return jsonify({'success': False, 'message': f"AI inference failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('YOLO_PORT', 5001))
    print(f" [AI] Starting YOLO Python Service on port {port}...")
    app.run(host='0.0.0.0', port=port, threaded=True)
