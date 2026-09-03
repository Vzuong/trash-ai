import os
import sys
import argparse
import cv2
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test YOLO model directly on an image file")
    parser.add_argument("image", nargs="?", default="server/uploads/sample_battery.jpg", help="Path to image file")
    parser.add_argument("--weights", default="best.pt", help="Path to model weights (default: best.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold (default: 0.45)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image file not found: {args.image}")
        return

    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights file not found: {args.weights}")
        return

    device = '0' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print(f"🔍 TESTING YOLO MODEL ON IMAGE: {args.image}")
    print(f"  Device: {'GPU (NVIDIA GTX 1650)' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Weights: {args.weights}")
    print(f"  Confidence Threshold: {args.conf}")
    print("=" * 60)

    model = YOLO(args.weights)
    img = cv2.imread(args.image)
    
    results = model.predict(source=img, conf=args.conf, iou=args.iou, device=device, verbose=True)[0]

    print("\n--- KẾT QUẢ DỰ ĐOÁN TỪ MODEL RAW ---")
    if not results.boxes or len(results.boxes) == 0:
        print(" -> Không phát hiện vật thể nào (No objects detected).")
    else:
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = results.names.get(cls_id, f"class_{cls_id}")
            xyxy = box.xyxy[0].tolist()
            print(f" [{i+1}] Class: {name:<12} | Conf: {conf*100:.1f}% | Box: {[int(x) for x in xyxy]}")

    # Save output visualization
    out_path = "test_output.jpg"
    annotated = results.plot()
    cv2.imwrite(out_path, annotated)
    print(f"\n✅ Đã lưu ảnh kết quả trực quan tại: {out_path}")

if __name__ == "__main__":
    main()
