import os
import glob
import argparse

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

def find_latest_best_weights(search_dir="runs"):
    """
    Finds the best.pt weight file, prioritizing root best.pt, weights/best.pt, or runs.
    """
    if os.path.exists("best.pt"):
        return "best.pt"
    if os.path.exists(os.path.join("weights", "best.pt")):
        return os.path.join("weights", "best.pt")
    weights_files = glob.glob(os.path.join(search_dir, "**", "weights", "best.pt"), recursive=True)
    if weights_files:
        return max(weights_files, key=os.path.getmtime)
    return 'yolo11s.pt'

def get_test_parser():
    parser = argparse.ArgumentParser(description='YOLOv11 Trash Recognition Testing / Evaluation')
    parser.add_argument('--weights', type=str, default=None, help='Path to trained weights file (Auto-finds latest best.pt if not specified)')
    parser.add_argument('--source', type=str, default='Trash_dataset_balanced/test/images', help='Path to test image file, folder, or video')
    parser.add_argument('--data_yaml', type=str, default='data_balanced.yaml', help='Path to dataset YAML file for evaluation metrics')
    parser.add_argument('--img_size', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='0', help='CUDA device or cpu')
    parser.add_argument('--save', action='store_true', default=True, help='Save detection results with bounding boxes')
    parser.add_argument('--val', action='store_true', default=True, help='Run model validation to compute evaluation metrics')
    return parser.parse_args()

def run_inference():
    args = get_test_parser()

    if not ULTRALYTICS_AVAILABLE:
        print("[ERROR] Ultralytics is not installed. Please run: pip install ultralytics")
        return

    # Check weights path or find latest best.pt
    weights_path = args.weights
    if not weights_path or not os.path.exists(weights_path):
        latest = find_latest_best_weights("runs")
        print(f"[INFO] Using latest detected trained weights: '{latest}'")
        weights_path = latest

    print("=" * 65)
    print(f" 🚀 Running YOLOv11 Trash Recognition Evaluation & Test...")
    print(f" Model Weights       : {weights_path}")
    print(f" Input Source        : {args.source}")
    print(f" Confidence Threshold: {args.conf}")
    print("=" * 65)

    # Load YOLOv11 model
    model = YOLO(weights_path)

    # 1. Run validation to get exact metrics if source is a dataset directory
    is_dataset_eval = args.val and os.path.exists(args.data_yaml) and os.path.isdir(args.source)
    if is_dataset_eval:
        eval_split = 'test' if 'test' in args.source else 'val'
        print(f"\n📊 [EVALUATION] Calculating quantitative metrics on '{eval_split}' set (Precision, Recall, mAP)...")
        val_results = model.val(
            data=args.data_yaml,
            split=eval_split,
            imgsz=args.img_size,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=True
        )

        p = val_results.box.mp
        r = val_results.box.mr
        map50 = val_results.box.map50
        map50_95 = val_results.box.map

        print("\n" + "=" * 65)
        print(" 🎯 BẢNG SỐ LIỆU ĐÁNH GIÁ MÔ HÌNH (YOLO EVALUATION METRICS)")
        print("=" * 65)
        print(f"  • Precision (Độ chính xác)      : {p:.4f} ({p*100:.2f}%)")
        print(f"  • Recall (Độ bao phủ / nhạy)     : {r:.4f} ({r*100:.2f}%)")
        print(f"  • mAP@50 (Mean Avg Precision)    : {map50:.4f} ({map50*100:.2f}%)")
        print(f"  • mAP@50-95                      : {map50_95:.4f} ({map50_95*100:.2f}%)")
        print("-" * 65)

        # Print per-class metrics if available
        if hasattr(val_results.box, 'maps') and val_results.box.maps is not None:
            class_names = val_results.names
            print(" 📋 CHI TIẾT THEO TỪNG LỚP RÁC (CLASS-WISE METRICS):")
            for cls_id, map_val in enumerate(val_results.box.maps):
                c_name = class_names.get(cls_id, f"Class {cls_id}")
                print(f"   - {c_name:<12}: mAP50-95 = {map_val:.4f}")
        print("=" * 65)

        save_dir = getattr(val_results, 'save_dir', None)
        if save_dir:
            print(f"\n 📈 BIỂU ĐỒ KẾT QUẢ ĐÃ ĐƯỢC TỰ ĐỘNG TẠO TẠI THƯ MỤC: {save_dir}")
            print(f"   • Biểu đồ loss & mAP qua các epoch : {os.path.join(save_dir, 'results.png')}")
            print(f"   • Ma trận nhầm lẫn (Confusion)     : {os.path.join(save_dir, 'confusion_matrix.png')}")
            print(f"   • Đường cong Precision - Recall    : {os.path.join(save_dir, 'BoxPR_curve.png')}")
            print(f"   • Đường cong F1-Score              : {os.path.join(save_dir, 'BoxF1_curve.png')}")
            print(f"   • File dữ liệu thô chi tiết CSV    : {os.path.join(save_dir, 'results.csv')}")

    # 2. Predict on test source
    print(f"\n🔍 [PREDICT] Running image inference on: '{args.source}'...")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=True,
        imgsz=args.img_size,
        device=args.device,
        save=args.save,
        verbose=False
    )

    if isinstance(results, list):
        print(f"\n[SUCCESS] Processed {len(results)} image(s).")
        for i, res in enumerate(results[:10]):  # Print detailed summary for up to 10 images
            num_objects = len(res.boxes)
            img_name = os.path.basename(args.source) if os.path.isfile(args.source) else f"Image {i+1}"
            print(f"\n 🖼️ [{img_name}] Phát hiện {num_objects} vật thể rác:")
            if num_objects > 0:
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    conf_val = float(box.conf[0].item())
                    class_name = res.names.get(cls_id, f"Class {cls_id}")
                    print(f"    • Lớp rác: {class_name:<12} (Độ tin cậy: {conf_val*100:.1f}%)")
            else:
                print("    (Không phát hiện vật thể rác nào vượt ngưỡng conf)")
            
            save_path = getattr(res, 'save_dir', 'runs/detect/predict')
            print(f"    📌 Ảnh kết quả khoanh vùng đã lưu tại: {os.path.join(save_path, img_name)}")

if __name__ == '__main__':
    run_inference()

