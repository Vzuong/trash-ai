"""
================================================================================
BENCHMARK TOÀN DIỆN MÔ HÌNH NHẬN DIỆN RÁC (YOLO, CBAM, RT-DETR, FASTER R-CNN)
Chuẩn hóa đo đạc GPU Inference Latency & FPS trên cùng tập ảnh test cố định.
================================================================================
Hỗ trợ đo đồng thời cả 3 trường phái kiến trúc:
1. One-Stage: YOLO11 (s, n, m), YOLOv8s, YOLO11s-CBAM (Attention)
2. Two-Stage: Faster R-CNN (MobileNetV3-Large FPN / torchvision)
3. Transformer: RT-DETR (RT-DETR-R18 / ultralytics)
================================================================================
"""

import os
import sys
import glob
import time
import zipfile
import random
import warnings
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import cv2

warnings.filterwarnings("ignore")

# ============================================================
# 1. ĐĂNG KÝ MODULE CBAM (Hỗ trợ nạp YOLO11s-CBAM)
# ============================================================
try:
    import torch.nn as nn
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block
    import ultralytics.nn.tasks as tasks

    class ChannelAttention(nn.Module):
        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            reduced_ch = max(channels // reduction, 8)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.mlp = nn.Sequential(
                nn.Conv2d(channels, reduced_ch, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced_ch, channels, kernel_size=1, bias=False)
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return x * self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size: int = 7):
            super().__init__()
            padding = 3 if kernel_size == 7 else 1
            self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            scale = torch.cat([avg_out, max_out], dim=1)
            return x * self.sigmoid(self.conv(scale))

    class CBAM(nn.Module):
        def __init__(self, c1: int, c2: int = None, reduction: int = 16, kernel_size: int = 7):
            super().__init__()
            self.channel_attention = ChannelAttention(c1, reduction=reduction)
            self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

        def forward(self, x):
            return self.spatial_attention(self.channel_attention(x))

    setattr(modules, 'CBAM', CBAM)
    setattr(block, 'CBAM', CBAM)
    setattr(tasks, 'CBAM', CBAM)
    tasks.__dict__['CBAM'] = CBAM
    if '__main__' in sys.modules:
        setattr(sys.modules['__main__'], 'CBAM', CBAM)
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([CBAM, ChannelAttention, SpatialAttention])
except Exception:
    pass

from ultralytics import YOLO, RTDETR
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# 2. CỐ ĐỊNH DANH SÁCH ẢNH TEST
# ============================================================
def get_fixed_test_images(num_samples=200, seed=42, cache_file='benchmark_200_images.txt'):
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            imgs = [line.strip() for line in f if line.strip() and os.path.exists(line.strip())]
        if len(imgs) >= num_samples:
            print(f"[OK] Đã nạp {len(imgs[:num_samples])} ảnh test cố định từ cache: {cache_file}")
            return imgs[:num_samples]

    dataset_candidates = [
        'Trash_dataset_balanced',
        '../Trash_dataset_balanced',
        '/content/trash_project/Trash_dataset_balanced',
        '/content/Trash_dataset_balanced'
    ]
    dataset_dir = None
    for d in dataset_candidates:
        if os.path.exists(d):
            dataset_dir = d
            break

    if not dataset_dir:
        zip_candidates = [
            '/content/drive/MyDrive/Trash (3).zip',
            '/content/drive/MyDrive/trash (3).zip',
            'Trash (3).zip'
        ]
        zip_file = next((z for z in zip_candidates if os.path.exists(z)), None)
        if zip_file:
            print(f"[*] Đang giải nén dataset: {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall('/content/trash_project')
            dataset_dir = '/content/trash_project/Trash_dataset_balanced'

    all_imgs = []
    if dataset_dir and os.path.exists(dataset_dir):
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
            all_imgs.extend(glob.glob(f'{dataset_dir}/**/{ext}', recursive=True))
        all_imgs = sorted(list(set(all_imgs)))

    if len(all_imgs) < num_samples:
        print(f"[!] Cảnh báo: Chỉ tìm thấy {len(all_imgs)} ảnh. Sẽ dùng tất cả ảnh hiện có hoặc tạo dummy.")
        if len(all_imgs) == 0:
            return []

    rng = random.Random(seed)
    selected = sorted(rng.sample(all_imgs, min(num_samples, len(all_imgs))))

    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.writelines(f"{p}\n" for p in selected)
    except Exception:
        pass

    return selected


# ============================================================
# 3. CHUẨN BỊ TENSOR DÙNG CHUNG
# ============================================================
def load_test_tensors(image_paths, imgsz=640, use_fp16=True, device='cuda:0'):
    tensors = []
    print(f"[*] Đang nạp và chuẩn hóa {len(image_paths)} tensor ({imgsz}x{imgsz})...")
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).unsqueeze(0).to(device=device, non_blocking=False)
        tensor = tensor.float().div_(255.0)
        if use_fp16:
            tensor = tensor.half()
        tensors.append(tensor)
    torch.cuda.synchronize()
    return tensors


def count_parameters(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 2)


# ============================================================
# 4. BENCHMARK INFERENCE MỘT LƯỢT
# ============================================================
def benchmark_single_repeat(model, tensors, warmup_iters=50, is_faster=False):
    model.eval()
    dummy = tensors[0]

    # Warmup GPU
    with torch.inference_mode():
        for _ in range(warmup_iters):
            if is_faster:
                _ = model([dummy[0]])
            else:
                _ = model(dummy)
    torch.cuda.synchronize()

    # Benchmark với torch.cuda.Event chính xác từng micro-giây
    latencies = []
    with torch.inference_mode():
        for tensor in tensors:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            if is_faster:
                _ = model([tensor[0]])
            else:
                _ = model(tensor)
            end_event.record()

            end_event.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

    return np.asarray(latencies, dtype=np.float64)


# ============================================================
# 5. KHỞI TẠO VÀ ĐO ĐẠC MÔ HÌNH (HỖ TRỢ YOLO, RT-DETR, FASTER)
# ============================================================
def benchmark_model(name, weights, tensors, repeat_id=1, num_repeats=1, warmup_iters=50, use_fp16=True, device='cuda:0'):
    print(f"\n[{repeat_id}/{num_repeats}] Đang đo mô hình: {name} ...")
    assert os.path.exists(weights), f"Không tìm thấy file trọng số: {weights}"

    detector = None
    is_faster = False

    # 1. Faster R-CNN (torchvision)
    if 'faster' in name.lower() or 'faster' in weights.lower():
        is_faster = True
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=8)  # 7 classes + background

        ckpt = torch.load(weights, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        model = model.to(device)

    # 2. RT-DETR (Vision Transformer)
    elif 'rtdetr' in name.lower() or 'rtdetr' in weights.lower():
        detector = RTDETR(weights)
        model = detector.model.to(device)

    # 3. YOLO (YOLO11, YOLOv8, YOLO11-CBAM)
    else:
        detector = YOLO(weights)
        model = detector.model.to(device)

    model.eval()
    if use_fp16:
        model = model.half()
    else:
        model = model.float()

    params = count_parameters(model)
    latencies = benchmark_single_repeat(model, tensors, warmup_iters=warmup_iters, is_faster=is_faster)

    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    std_ms = float(np.std(latencies, ddof=1))
    p95_ms = float(np.percentile(latencies, 95))
    p99_ms = float(np.percentile(latencies, 99))
    fps_mean = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print(f"    Params: {params:.2f}M | Mean Latency: {mean_ms:.2f} ms | P95: {p95_ms:.2f} ms | FPS: {fps_mean:.1f}")

    if detector is not None:
        del detector
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {
        'Model': name,
        'Repeat': repeat_id,
        'Params (M)': params,
        'Mean (ms)': round(mean_ms, 3),
        'Median (ms)': round(median_ms, 3),
        'Std (ms)': round(std_ms, 3),
        'P95 (ms)': round(p95_ms, 3),
        'P99 (ms)': round(p99_ms, 3),
        'Pure FPS': round(fps_mean, 2)
    }


# ============================================================
# 6. HÀM MAIN THỰC THI TOÀN DIỆN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU Inference: YOLO vs RT-DETR vs Faster R-CNN")
    parser.add_argument('--samples', type=int, default=200, help="Số lượng ảnh test chuẩn hóa")
    parser.add_argument('--imgsz', type=int, default=640, help="Kích thước ảnh đầu vào (640)")
    parser.add_argument('--repeats', type=int, default=5, help="Số lần lặp độc lập")
    parser.add_argument('--warmup', type=int, default=50, help="Số lượt warmup GPU")
    parser.add_argument('--fp32', action='store_true', help="Chạy FP32 thay vì FP16")
    parser.add_argument('--device', type=str, default='cuda:0', help="Thiết bị tính toán")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Yêu cầu môi trường có GPU để đo đạc chuẩn xác!"
    use_fp16 = not args.fp32
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 100)
    print("       CHƯƠNG TRÌNH BENCHMARK TOÀN DIỆN 4 DÒNG KIẾN TRÚC MÔ HÌNH")
    print(f"       GPU: {torch.cuda.get_device_name(0)} | Precision: {'FP16' if use_fp16 else 'FP32'}")
    print("=" * 100)

    # Danh mục các mô hình so sánh (tự động lọc file tồn tại)
    MODELS_CANDIDATES = {
        'YOLO11s (Gốc)': 'yolo11s.pt',
        'YOLO11s-CBAM (Đề xuất)': 'bestfold1.pt',
        'YOLO11n (Nano)': 'yolo11n.pt',
        'YOLOv8s (Tiền nhiệm)': 'yolov8s.pt',
        'RT-DETR-R18 (Transformer)': 'best_rtdetr.pt',
        'Faster R-CNN (Two-Stage)': 'best_fasterrcnn_mobilenetv3.pth'
    }

    models_to_run = {name: path for name, path in MODELS_CANDIDATES.items() if os.path.exists(path)}

    if not models_to_run:
        print("[!] Không tìm thấy trọng số mặc định ở thư mục hiện tại.")
        print("[*] Kiểm tra các file .pt trong thư mục:")
        for f in glob.glob("*.pt"):
            models_to_run[f] = f

    if not models_to_run:
        print("[ERROR] Không tìm thấy bất kỳ file trọng số nào để benchmark!")
        return

    print(f"[+] Tìm thấy {len(models_to_run)} mô hình sẵn sàng để đo:")
    for name, path in models_to_run.items():
        print(f"    - {name:<30}: {path}")

    # Chuẩn bị ảnh và tensor
    image_paths = get_fixed_test_images(num_samples=args.samples)
    if not image_paths:
        print("[*] Tạo tensor ngẫu nhiên chuẩn hóa 640x640 thay thế do không có dataset ảnh...")
        tensors = [torch.randn(1, 3, args.imgsz, args.imgsz, dtype=torch.float16 if use_fp16 else torch.float32, device=device) for _ in range(min(args.samples, 50))]
    else:
        tensors = load_test_tensors(image_paths, imgsz=args.imgsz, use_fp16=use_fp16, device=device)

    all_results = []
    for r in range(1, args.repeats + 1):
        print(f"\n>>> VÒNG ĐO (REPEAT) {r}/{args.repeats} <<<")
        items = list(models_to_run.items())
        random.Random(42 + r).shuffle(items)
        for name, path in items:
            res = benchmark_model(name, path, tensors, repeat_id=r, num_repeats=args.repeats, warmup_iters=args.warmup, use_fp16=use_fp16, device=device)
            all_results.append(res)

    df_raw = pd.DataFrame(all_results)
    summary_rows = []
    for model_name in models_to_run.keys():
        sub = df_raw[df_raw['Model'] == model_name]
        summary_rows.append({
            'Model': model_name,
            'Params (M)': sub['Params (M)'].iloc[0],
            'Mean Latency (ms)': round(float(sub['Mean (ms)'].median()), 2),
            'P95 Latency (ms)': round(float(sub['P95 (ms)'].median()), 2),
            'Pure FPS': round(float(sub['Pure FPS'].median()), 1)
        })

    df_summary = pd.DataFrame(summary_rows).sort_values(by='Pure FPS', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("                       BẢNG TỔNG HỢP HIỆU NĂNG SUY LUẬN CUỐI CÙNG")
    print("=" * 100)
    print(df_summary.to_string(index=False))
    print("=" * 100)

    out_csv = "benchmark_summary_results.csv"
    df_summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"[+] Đã lưu kết quả tóm tắt ra file: {out_csv}")


if __name__ == '__main__':
    main()
