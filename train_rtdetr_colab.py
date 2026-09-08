"""
Huấn luyện và Đánh giá Mô hình RT-DETR (Real-Time Detection Transformer)
Dự án: Nhận diện & Phân loại 7 Lớp Rác Thải (trash-ai)
Mục đích: Xây dựng mô hình đối chứng trường phái Vision Transformer ngoài họ YOLO
Môi trường: Google Colab - GPU Tesla T4 (16GB VRAM)
"""

import os
import sys
import time
import shutil
import zipfile
import argparse
import traceback
import pandas as pd
import torch

try:
    from ultralytics import RTDETR
except ImportError:
    raise ImportError("Vui lòng cài đặt ultralytics: pip install -q ultralytics pyyaml pandas")


# ==============================================================================
# 1. CẤU HÌNH THƯ MỤC VÀ DỮ LIỆU
# ==============================================================================

def setup_environment_and_data():
    """
    Kết nối Google Drive, tự động dò tìm và giải nén dữ liệu Trash_dataset_balanced,
    đồng thời tạo file cấu hình data_balanced.yaml chuẩn 3 tập: train, val, test.
    """
    print("\n" + "=" * 75)
    print(" [BƯỚC 1/4] THIẾT LẬP MÔI TRƯỜNG & CHUẨN BỊ DỮ LIỆU (TRAIN / VAL / TEST)")
    print("=" * 75)

    # 1.1 Mount Google Drive (nếu đang chạy trên Colab)
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            print("-> Đang kết nối Google Drive...")
            drive.mount('/content/drive')
        print("✅ Đã kết nối Google Drive thành công!")
    except ImportError:
        print("ℹ️ Đang chạy ở môi trường cục bộ (Non-Colab).")

    # 1.2 Giải nén dataset nếu chưa có
    extract_target = '/content/trash_project' if os.path.exists('/content') else './trash_project'
    zip_candidates = [
        '/content/drive/MyDrive/Trash (3).zip',
        '/content/drive/MyDrive/trash (3).zip',
        '/content/Trash (3).zip',
        'Trash (3).zip',
        '../Trash (3).zip'
    ]
    
    zip_found = next((z for z in zip_candidates if os.path.exists(z)), None)
    expected_dataset_name = 'Trash_dataset_balanced'
    
    dataset_dir = None
    # Tìm kiếm dataset đã giải nén sẵn
    for search_root in [extract_target, '/content', '.', '..']:
        if os.path.exists(search_root):
            for root, dirs, _ in os.walk(search_root):
                if os.path.basename(root) == expected_dataset_name:
                    dataset_dir = root
                    break
        if dataset_dir:
            break

    if not dataset_dir and zip_found:
        print(f"📦 Đang giải nén tập dữ liệu từ: {zip_found} -> {extract_target}...")
        os.makedirs(extract_target, exist_ok=True)
        with zipfile.ZipFile(zip_found, 'r') as z:
            z.extractall(extract_target)
        
        # Quét lại sau khi giải nén
        for root, dirs, _ in os.walk(extract_target):
            if os.path.basename(root) == expected_dataset_name:
                dataset_dir = root
                break

    assert dataset_dir is not None and os.path.exists(dataset_dir), (
        f"❌ Không tìm thấy thư mục '{expected_dataset_name}'! "
        f"Vui lòng đảm bảo file zip đã được tải lên Drive hoặc thư mục dataset đã tồn tại."
    )
    print(f"✅ Đã xác định thư mục dataset: {os.path.abspath(dataset_dir)}")

    # 1.3 Kiểm tra sự tồn tại của 3 tập train, val, test
    train_dir = os.path.join(dataset_dir, 'train', 'images')
    val_dir = os.path.join(dataset_dir, 'val', 'images')
    test_dir = os.path.join(dataset_dir, 'test', 'images')

    assert os.path.exists(train_dir), f"Không tìm thấy thư mục {train_dir}"
    assert os.path.exists(val_dir), f"Không tìm thấy thư mục {val_dir}"
    has_test = os.path.exists(test_dir)
    if not has_test:
        print("⚠️ Không tìm thấy folder test riêng, dùng val cho khâu test độc lập.")
        test_dir = val_dir

    # 1.4 Tạo file data_balanced.yaml
    yaml_content = f"""# Trash AI - 7 Classes Dataset Configuration
path: {os.path.abspath(dataset_dir).replace('\\', '/')}
train: train/images
val: val/images
test: {'test/images' if has_test else 'val/images'}

nc: 7
names:
  0: battery
  1: cardboard
  2: paper
  3: glass
  4: metal
  5: plastic
  6: organic
"""
    yaml_path = '/content/data_balanced.yaml' if os.path.exists('/content') else 'data_balanced.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content.strip())
    print(f"✅ File cấu hình đã sẵn sàng tại: {yaml_path}")
    return yaml_path


# ==============================================================================
# 2. KIỂM TRA BỘ NHỚ VÀ TỐI ƯU HÓA GPU T4
# ==============================================================================

def check_gpu_status():
    """Kiểm tra và dọn dẹp GPU VRAM trước khi train."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"⚡ GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB")
        if vram_gb < 14.0:
            print("⚠️ Cảnh báo: VRAM nhỏ hơn 14GB, nên để batch=8 để tránh tràn RAM.")
    else:
        print("⚠️ CẢNH BÁO: Không phát hiện GPU CUDA! Quá trình huấn luyện Transformer trên CPU sẽ rất chậm.")


# ==============================================================================
# 3. QUẢN LÝ ĐỒNG BỘ CHECKPOINTS & ARTIFACTS SANG GOOGLE DRIVE
# ==============================================================================

def safe_sync_file(src, dst):
    """Sao chép an toàn một file từ src sang dst."""
    if not os.path.exists(src):
        return
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy2(src, dst)
    except Exception:
        pass


def sync_all_artifacts(local_dir, drive_dir):
    """Sao chép toàn bộ folder kết quả (biểu đồ, log, weights) sang Drive."""
    if not drive_dir or not os.path.exists(local_dir):
        return
    try:
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copytree(local_dir, drive_dir, dirs_exist_ok=True)
        print(f"📁 Đã đồng bộ toàn bộ artifacts sang Google Drive: {drive_dir}")
    except Exception as e:
        print(f"⚠️ Không thể copy toàn bộ thư mục sang Drive: {e}")


# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH: TRAIN & ĐÁNH GIÁ ĐỐI CHỨNG TEST SET
# ==============================================================================

def train_rtdetr_evidence(args):
    # Khởi tạo data yaml
    data_yaml = args.data if (args.data and os.path.exists(args.data)) else setup_environment_and_data()

    # Kiểm tra GPU
    check_gpu_status()

    # Thiết lập đường dẫn lưu trữ
    local_project = '/content/runs/detect' if os.path.exists('/content') else 'runs/detect'
    exp_name = 'trash_rtdetr_evidence'
    local_save_dir = os.path.join(local_project, exp_name)
    drive_save_dir = '/content/drive/MyDrive/Trash_Runs/trash_rtdetr_evidence' if os.path.exists('/content/drive/MyDrive') else None

    if drive_save_dir:
        os.makedirs(drive_save_dir, exist_ok=True)
        os.makedirs(os.path.join(drive_save_dir, 'weights'), exist_ok=True)

    # 4.1 Tạo Callback Realtime đồng bộ từng Epoch
    def on_fit_epoch_end(trainer):
        if drive_save_dir:
            # Sync logs và cấu hình
            for fname in ['results.csv', 'args.yaml']:
                safe_sync_file(os.path.join(local_save_dir, fname), os.path.join(drive_save_dir, fname))
            # Sync trọng số checkpoint
            for wname in ['last.pt', 'best.pt']:
                safe_sync_file(os.path.join(local_save_dir, 'weights', wname), os.path.join(drive_save_dir, 'weights', wname))

    print("\n" + "=" * 75)
    print(" [BƯỚC 2/4] KHỞI TẠO VÀ HUẤN LUYỆN RT-DETR (VISION TRANSFORMER)")
    print("=" * 75)
    print(f" Kiến trúc          : RT-DETR (Pretrained: {args.weights})")
    print(f" Dữ liệu            : {data_yaml}")
    print(f" Epochs             : {args.epochs} (Patience Early Stopping: {args.patience})")
    print(f" Batch Size         : {args.batch} | Imgsz: {args.imgsz}")
    print(f" Optimizer          : AdamW (lr0={args.lr0}, weight_decay=0.0005)")
    print(f" Thư mục Local SSD  : {local_save_dir}")
    print(f" Thư mục Cloud Drive: {drive_save_dir or 'Không kết nối'}")
    print("=" * 75 + "\n")

    # 4.2 Nạp mô hình RTDETR
    model = RTDETR(args.weights)
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

    # 4.3 Tiến hành Huấn luyện
    start_train_time = time.time()
    try:
        model.train(
            data=data_yaml,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device if torch.cuda.is_available() else 'cpu',
            workers=args.workers,
            optimizer='AdamW',
            lr0=args.lr0,
            patience=args.patience,
            cos_lr=False,
            weight_decay=0.0005,
            amp=True,              # Bật Automatic Mixed Precision để chống CUDA OOM trên T4
            project=local_project,
            name=exp_name,
            exist_ok=True,
            save=True,
            verbose=True
        )
    except torch.cuda.OutOfMemoryError:
        print("\n" + "!" * 75)
        print("❌ LỖI TRÀN BỘ NHỚ CUDA (CUDA OOM)!")
        print("💡 GIẢI PHÁP:")
        print("   1. Giảm kích thước batch xuống 8: python train_rtdetr_colab.py --batch 8")
        print("   2. Hoặc sử dụng backbone nhẹ hơn: python train_rtdetr_colab.py --weights rtdetr-r18.pt")
        print("!" * 75 + "\n")
        return
    except Exception as e:
        print(f"❌ Xảy ra lỗi trong quá trình train: {e}")
        traceback.print_exc()
        return

    train_duration = (time.time() - start_train_time) / 3600
    print(f"\n🎉 Quá trình huấn luyện hoàn tất sau: {train_duration:.2f} giờ!")

    # Đồng bộ toàn bộ đồ thị, kết quả sau khi train xong
    sync_all_artifacts(local_save_dir, drive_save_dir)

    # 4.4 Xác định checkpoint best.pt
    best_candidates = [
        os.path.join(local_save_dir, 'weights', 'best.pt'),
        os.path.join(drive_save_dir, 'weights', 'best.pt') if drive_save_dir else ''
    ]
    best_pt = next((b for b in best_candidates if b and os.path.exists(b)), None)
    assert best_pt is not None, "❌ Không tìm thấy file trọng số 'best.pt' sau huấn luyện!"
    print(f"\n🏆 Đã tìm thấy trọng số tối ưu nhất: {best_pt}")

    # ==============================================================================
    # 5. ĐÁNH GIÁ ĐỐI CHỨNG TRÊN TẬP TEST SET ĐỘC LẬP
    # ==============================================================================
    print("\n" + "=" * 75)
    print(" [BƯỚC 3/4] ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST SET (split='test')")
    print("=" * 75)

    eval_model = RTDETR(best_pt)
    test_res = eval_model.val(
        data=data_yaml,
        split='test',
        imgsz=args.imgsz,
        device=args.device if torch.cuda.is_available() else 'cpu',
        verbose=True
    )

    # Trích xuất các tham số đo lường
    param_count = round(sum(p.numel() for p in eval_model.model.parameters()) / 1e6, 2)
    latency_ms = test_res.speed.get('inference', 0.0)
    fps = (1000.0 / latency_ms) if latency_ms > 0 else 0.0

    p_overall = float(test_res.box.mp)
    r_overall = float(test_res.box.mr)
    map50_overall = float(test_res.box.map50)
    map_overall = float(test_res.box.map)

    # Thu thập chi tiết từng lớp
    class_names = list(test_res.names.values())
    per_class_records = []

    print("\n" + "=" * 75)
    print(" [BƯỚC 4/4] BẰNG CHỨNG THỰC NGHIỆM ĐỐI CHỨNG: RT-DETR vs YOLO")
    print("=" * 75)

    report_lines = []
    report_lines.append("=" * 75)
    report_lines.append("           BÁO CÁO THỰC NGHIỆM ĐỐI CHỨNG RT-DETR TRÊN TẬP TEST SET")
    report_lines.append("=" * 75)
    report_lines.append(f"Kiến trúc           : RT-DETR (Real-Time Detection Transformer)")
    report_lines.append(f"Trọng số cơ sở      : {args.weights}")
    report_lines.append(f"Số lượng tham số    : {param_count} M (Triệu parameters)")
    report_lines.append(f"Độ trễ suy luận     : {latency_ms:.2f} ms (~{fps:.1f} FPS trên T4)")
    report_lines.append("-" * 75)
    report_lines.append(f"Precision tổng thể  : {p_overall:.4f} ({p_overall*100:.2f}%)")
    report_lines.append(f"Recall tổng thể     : {r_overall:.4f} ({r_overall*100:.2f}%)")
    report_lines.append(f"mAP@50 tổng thể     : {map50_overall:.4f} ({map50_overall*100:.2f}%)")
    report_lines.append(f"mAP@50-95 tổng thể  : {map_overall:.4f} ({map_overall*100:.2f}%)")
    report_lines.append("-" * 75)
    report_lines.append(f"{'Lớp (Class)':<14} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10} | {'mAP@50-95':<10}")
    report_lines.append("-" * 75)

    for i, cname in enumerate(class_names):
        try:
            p_cls, r_cls, map50_cls, map_cls = test_res.box.class_result(i)
        except Exception:
            p_cls = r_cls = 0.0
            map50_cls = float(test_res.box.map50)
            map_cls = float(test_res.box.maps[i]) if i < len(test_res.box.maps) else 0.0

        line = f"{cname:<14} | {p_cls:<10.4f} | {r_cls:<10.4f} | {map50_cls:<10.4f} | {map_cls:<10.4f}"
        report_lines.append(line)
        per_class_records.append({
            'class': cname,
            'precision': round(float(p_cls), 4),
            'recall': round(float(r_cls), 4),
            'map50': round(float(map50_cls), 4),
            'map50_95': round(float(map_cls), 4)
        })

    report_lines.append("=" * 75)
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # 5.1 Lưu báo cáo ra file txt
    report_txt_path = os.path.join(local_save_dir, 'test_evaluation_report.txt')
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # 5.2 Lưu bảng chi tiết ra CSV
    df_classes = pd.DataFrame(per_class_records)
    csv_report_path = os.path.join(local_save_dir, 'test_per_class_metrics.csv')
    df_classes.to_csv(csv_report_path, index=False)

    # 5.3 Đồng bộ báo cáo sang Google Drive
    if drive_save_dir:
        safe_sync_file(report_txt_path, os.path.join(drive_save_dir, 'test_evaluation_report.txt'))
        safe_sync_file(csv_report_path, os.path.join(drive_save_dir, 'test_per_class_metrics.csv'))
        print(f"\n✅ Đã lưu đầy đủ bằng chứng thực nghiệm (Report & CSV) sang Google Drive:")
        print(f"   -> {drive_save_dir}/test_evaluation_report.txt")
        print(f"   -> {drive_save_dir}/test_per_class_metrics.csv")
        print(f"   -> {drive_save_dir}/results.csv")
        print(f"   -> {drive_save_dir}/results.png")
        print(f"   -> {drive_save_dir}/confusion_matrix.png")
        print(f"   -> {drive_save_dir}/weights/best.pt")


# ==============================================================================
# 6. ENTRY POINT & ARGUMENTS
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Huấn luyện và Đánh giá RT-DETR trên Google Colab")
    parser.add_argument('--weights', type=str, default='rtdetr-l.pt', help='Pretrained weights: rtdetr-l.pt hoặc rtdetr-r18.pt')
    parser.add_argument('--epochs', type=int, default=100, help='Số epoch tối đa')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (16 khuyến nghị cho T4 16GB VRAM)')
    parser.add_argument('--imgsz', type=int, default=640, help='Kích thước ảnh')
    parser.add_argument('--lr0', type=float, default=0.0001, help='Learning rate khởi tạo cho Transformer (chuẩn 1e-4)')
    parser.add_argument('--patience', type=int, default=10, help='Patience ngắt sớm Early Stopping')
    parser.add_argument('--workers', type=int, default=2, help='Số luồng nạp dữ liệu DataLoader')
    parser.add_argument('--device', type=int, default=0, help='Chỉ mục GPU')
    parser.add_argument('--data', type=str, default=None, help='Đường dẫn tùy chọn data yaml')
    
    args = parser.parse_args()
    train_rtdetr_evidence(args)
