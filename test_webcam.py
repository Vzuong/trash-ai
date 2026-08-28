import os
import time
import argparse
import cv2
import torch
from util import ULTRALYTICS_AVAILABLE

if ULTRALYTICS_AVAILABLE:
    from ultralytics import YOLO

def find_latest_best_weights(search_dir="runs"):
    import glob
    if os.path.exists("best.pt"):
        return "best.pt"
    if os.path.exists(os.path.join("weights", "best.pt")):
        return os.path.join("weights", "best.pt")
    weights_files = glob.glob(os.path.join(search_dir, "**", "weights", "best.pt"), recursive=True)
    if weights_files:
        return max(weights_files, key=os.path.getmtime)
    return 'yolo11s.pt'

def open_working_camera(camera_index=1):
    """
    Opens camera at specified index (prioritizing Iriun/External Webcams).
    """
    # Try standard backend first (best for Iriun Webcam & phone virtual cams)
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[SUCCESS] Connected to External/Iriun Webcam Index: {camera_index}")
            return cap, camera_index
        cap.release()

    # Try DirectShow backend as secondary fallback
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[SUCCESS] Connected to Camera Index {camera_index} (DirectShow)")
            return cap, camera_index
        cap.release()

    return None, camera_index

def run_webcam():
    parser = argparse.ArgumentParser(description='YOLOv11 Real-Time Webcam Trash Recognition (Locked 30 FPS)')
    parser.add_argument('--weights', type=str, default=None, help='Path to trained weights file')
    parser.add_argument('--camera', type=int, default=1, help='Webcam camera index')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold (default: 0.35 to filter background noise)')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold (default: 0.45)')
    parser.add_argument('--img_size', type=int, default=640, help='Inference image size (default: 640)')
    parser.add_argument('--target_fps', type=int, default=30, help='Target locked FPS limit (default: 30)')
    args = parser.parse_args()

    if not ULTRALYTICS_AVAILABLE:
        print("[ERROR] Ultralytics is not installed. Please run: pip install ultralytics")
        return

    # Find weights
    weights_path = args.weights
    if not weights_path or not os.path.exists(weights_path):
        weights_path = find_latest_best_weights("runs")
        print(f"[INFO] Using latest trained model weights: '{weights_path}'")

    # Hardware Acceleration settings
    device = '0' if torch.cuda.is_available() else 'cpu'

    # Load YOLO model
    model = YOLO(weights_path)

    print("=" * 60)
    print(f"🎬 STARTING WEBCAM TRASH RECOGNITION (LOCKED AT {args.target_fps} FPS)")
    print(f"  Device                 : {'GPU (NVIDIA GTX 1650)' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Confidence Threshold   : {args.conf}")
    print(f"  IoU Threshold          : {args.iou}")
    print(f"  Model Internal Classes : {model.names}")
    print(f"  Model Weights          : {weights_path}")
    print(f"  Press 'q' or 'ESC' to exit")
    print("=" * 60)

    # Open specified camera
    cap, cam_idx = open_working_camera(args.camera)
    
    if cap is None:
        print(f"[WARNING] Could not open camera at index {args.camera}. Checking fallback indices (0, 1, 2)...")
        for fallback_idx in [1, 0, 2]:
            if fallback_idx != args.camera:
                cap, cam_idx = open_working_camera(fallback_idx)
                if cap is not None:
                    break

    if cap is None:
        print("[ERROR] Could not open webcam!")
        return

    target_frame_duration = 1.0 / max(args.target_fps, 1)
    prev_time = time.time()

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Failed to grab frame from webcam. Exiting...")
            break

        # Run GPU Accelerated YOLOv11 prediction on current frame with Agnostic NMS (1 box per object)
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            agnostic_nms=True,
            imgsz=args.img_size,
            device=device,
            verbose=False
        )

        annotated_frame = results[0].plot()

        # Calculate actual FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        cv2.putText(annotated_frame, f"FPS: {fps:.1f} / 30.0", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show frame window
        cv2.imshow("YOLOv11 Real-Time Trash Recognition", annotated_frame)

        # FPS Lock Sleep Control (30 FPS max)
        frame_elapsed = time.time() - frame_start
        sleep_duration = target_frame_duration - frame_elapsed
        if sleep_duration > 0:
            time.sleep(sleep_duration)

        # Exit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n[INFO] Exiting webcam recognition...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_webcam()
