import os
import sys
import time
import argparse
from datetime import datetime

import cv2
import torch

# Đăng ký module CBAM trước khi nạp model
try:
    import modules.cbam as cbam_mod
    if '__main__' in sys.modules:
        setattr(sys.modules['__main__'], 'CBAM', cbam_mod.CBAM)
        setattr(sys.modules['__main__'], 'ChannelAttention', cbam_mod.ChannelAttention)
        setattr(sys.modules['__main__'], 'SpatialAttention', cbam_mod.SpatialAttention)
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([cbam_mod.CBAM, cbam_mod.ChannelAttention, cbam_mod.SpatialAttention])
except Exception as e:
    print(f"[CẢNH BÁO] Không thể import modules.cbam: {e}")

from ultralytics import YOLO

# ============================================================
# CẤU HÌNH & BẢNG MÀU 7 LỚP RÁC
# ============================================================

# Tên tiếng Việt và nhãn loại rác
CLASS_VN = {
    'battery': 'Pin (Nguy hai)',
    'cardboard': 'Bia carton',
    'paper': 'Giay',
    'glass': 'Thuy tinh',
    'metal': 'Kim loai',
    'plastic': 'Nhua',
    'organic': 'Huu co'
}

# Bảng màu riêng cho 7 lớp (BGR)
CLASS_COLORS = {
    'battery': (0, 0, 230),       # Đỏ tươi
    'cardboard': (40, 100, 180),  # Nâu vàng
    'paper': (220, 180, 50),      # Xanh lam
    'glass': (220, 220, 0),       # Cyan / Xanh ngọc
    'metal': (180, 180, 180),     # Xám bạc
    'plastic': (0, 140, 255),     # Cam sáng
    'organic': (50, 205, 50)      # Xanh lá cây
}

def draw_hud(frame, fps, infer_ms, conf_thresh, counts, model_title="YOLO11s-CBAM"):
    """Vẽ thanh thông số HUD hiện đại phía trên khung hình"""
    h, w = frame.shape[:2]
    
    # Thanh nền mờ phía trên
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 25), -1)
    
    # Thống kê nhanh số lượng rác nhận diện được (nếu có)
    summary_text = ""
    if counts:
        parts = [f"{cls_name}: {cnt}" for cls_name, cnt in counts.items()]
        summary_text = " | ".join(parts[:3])  # hiển thị tối đa 3 loại đầu
    
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # In thông tin hệ thống
    status_text = f"[{model_title}]  FPS: {fps:.1f}  |  Lat: {infer_ms:.1f}ms  |  Conf: {conf_thresh:.2f}"
    cv2.putText(frame, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)
    
    guide_text = "Phim: [Q] Thoat | [S] Chup anh | [+] [-] Chinh Conf"
    if summary_text:
        guide_text = f"Phat hien: {summary_text}  --  {guide_text}"
    cv2.putText(frame, guide_text, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

def open_working_camera(preferred_idx=0):
    """Tự động dò và mở camera hoạt động tốt nhất (ưu tiên preferred_idx, sau đó thử các cổng khác)"""
    candidate_indices = [preferred_idx] + [i for i in [0, 1, 2] if i != preferred_idx]
    
    for idx in candidate_indices:
        print(f"-> Đang thử mở Camera index {idx}...")
        # Thử DirectShow (DSHOW) trên Windows trước, nếu không được dùng mặc định
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
            
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"✅ Kết nối thành công Camera index {idx}!")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap, idx
            cap.release()
            
    return None, -1

def main():
    parser = argparse.ArgumentParser(description="Test Webcam với mô hình YOLO11s-CBAM")
    parser.add_argument("--model", type=str, default="bestfold1.pt", help="Đường dẫn file trọng số model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0: Webcam tích hợp, 1: Iriun/Cam ngoài)")
    parser.add_argument("--conf", type=float, default=0.35, help="Ngưỡng tự tin Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="Ngưỡng NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    args = parser.parse_args()

    # Kiểm tra đường dẫn model
    model_path = args.model
    if not os.path.exists(model_path):
        candidates = ["bestfold1.pt", "best.pt", "weights/bestfold1.pt", "weights/best.pt"]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file trọng số '{args.model}'!")
        print("Vui lòng kiểm tra lại đường dẫn file .pt.")
        return

    print(f"📦 Đang tải mô hình từ: {model_path}...")
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"⚡ Thiết bị tăng tốc: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo mô hình: {e}")
        return

    # Mở Camera
    cap, used_idx = open_working_camera(args.camera)
    if cap is None:
        print("❌ Không thể mở bất kỳ Camera nào! Hãy kiểm tra lại kết nối webcam hoặc quyền truy cập camera.")
        return

    window_name = f"Trash AI - YOLO11s-CBAM ({os.path.basename(model_path)})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 700)

    conf_thresh = args.conf
    prev_time = time.perf_counter()
    fps = 0.0
    save_dir = "captured_detections"
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("🚀 ĐANG CHẠY TEST WEBCAM - PHÂN LOẠI RÁC THẢI")
    print(" - Nhấn 'Q' hoặc 'ESC' để THOÁT")
    print(" - Nhấn 'S' hoặc 'C' để CHỤP VÀ LƯU ẢNH")
    print(" - Nhấn '+' / '-' để TĂNG / GIẢM NGƯỠNG CONFIDENCE")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        start_infer = time.perf_counter()
        results = model.predict(
            source=frame,
            conf=conf_thresh,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            verbose=False
        )[0]
        infer_ms = (time.perf_counter() - start_infer) * 1000

        # Tính FPS mượt mà
        curr_time = time.perf_counter()
        delta = curr_time - prev_time
        prev_time = curr_time
        instant_fps = 1.0 / max(delta, 1e-6)
        fps = fps * 0.9 + instant_fps * 0.1

        # Tạo bản vẽ
        annotated = frame.copy()
        counts = {}

        # Vẽ bounding boxes với màu sắc và tên tiếng Việt
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_en = model.names.get(cls_id, str(cls_id))
                cls_vn = CLASS_VN.get(cls_en, cls_en)
                
                counts[cls_en] = counts.get(cls_en, 0) + 1
                color = CLASS_COLORS.get(cls_en, (0, 255, 0))

                # Vẽ Box
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Vẽ nhãn phía trên box
                label = f"{cls_vn} {conf*100:.1f}%"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                
                label_y1 = max(0, y1 - th - 8)
                label_y2 = y1
                cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 8, label_y2), color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255) if color != (220, 220, 0) else (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

        # Vẽ thanh HUD thông tin
        draw_hud(annotated, fps, infer_ms, conf_thresh, counts, model_title="YOLO11s-CBAM")

        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q hoặc ESC
            break
        elif key == ord('s') or key == ord('c'):  # Lưu ảnh chụp
            fname = os.path.join(save_dir, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(fname, annotated)
            print(f"📸 Đã lưu ảnh phát hiện vào: {fname}")
        elif key == ord('+') or key == ord('='):  # Tăng confidence
            conf_thresh = min(0.95, conf_thresh + 0.05)
            print(f"🔼 Ngưỡng Confidence: {conf_thresh:.2f}")
        elif key == ord('-') or key == ord('_'):  # Giảm confidence
            conf_thresh = max(0.10, conf_thresh - 0.05)
            print(f"🔽 Ngưỡng Confidence: {conf_thresh:.2f}")

        # Nếu người dùng bấm dấu X trên cửa sổ để tắt
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Đã dừng chương trình test webcam.")

if __name__ == "__main__":
    main()
