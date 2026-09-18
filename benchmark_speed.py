import os
import sys
import time
import numpy as np
import cv2
import torch
import modules.cbam
from ultralytics import YOLO

def get_stats(times_ms):
    times = np.array(times_ms)
    return {
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "p95": float(np.percentile(times, 95)),
        "std": float(np.std(times)),
        "fps": float(1000.0 / np.mean(times))
    }

def print_result_row(test_name, stats):
    print(f"| {test_name:<40} | {stats['mean']:>8.2f} ms | {stats['median']:>8.2f} ms | {stats['min']:>8.2f} ms | {stats['p95']:>8.2f} ms | {stats['fps']:>7.1f} |")

def main():
    print("=" * 85)
    print("           BENCHMARK TỐC ĐỘ SUY LUẬN (INFERENCE) - BESTFOLD1")
    print("=" * 85)
    
    # Hardware info
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    cpu_name = "Intel Core i5-10300H @ 2.50GHz"
    print(f"[*] PyTorch Version : {torch.__version__}")
    print(f"[*] CUDA Available  : {cuda_available}")
    print(f"[*] GPU             : {gpu_name}")
    print(f"[*] CPU             : {cpu_name}")
    print("=" * 85)

    WARMUP = 20
    NUM_RUNS = 100

    results_table = []

    # -------------------------------------------------------------
    # 1. PURE MODEL FORWARD PASS (PyTorch PT)
    # -------------------------------------------------------------
    pt_path = "bestfold1.pt"
    print(f"\n[1] Đang tải mô hình PyTorch: {pt_path} ...")
    yolo_pt = YOLO(pt_path)
    nn_model = yolo_pt.model
    nn_model.eval()

    # --- GPU FP32 ---
    if cuda_available:
        print(f"    -> Đang đo PyTorch CUDA (FP32) ({WARMUP} warmup, {NUM_RUNS} runs)...")
        nn_model.to('cuda').float()
        dummy_cuda = torch.randn(1, 3, 640, 640, dtype=torch.float32, device='cuda')
        
        with torch.no_grad():
            for _ in range(WARMUP):
                _ = nn_model(dummy_cuda)
            torch.cuda.synchronize()
            
            times = []
            for _ in range(NUM_RUNS):
                start = time.perf_counter()
                _ = nn_model(dummy_cuda)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            stats = get_stats(times)
            results_table.append(("PyTorch GPU (FP32) - Pure Forward", stats))

        # --- GPU FP16 (Half) ---
        print(f"    -> Đang đo PyTorch CUDA (FP16 / Half) ({WARMUP} warmup, {NUM_RUNS} runs)...")
        nn_model.half()
        dummy_half = torch.randn(1, 3, 640, 640, dtype=torch.float16, device='cuda')
        
        with torch.no_grad():
            for _ in range(WARMUP):
                _ = nn_model(dummy_half)
            torch.cuda.synchronize()
            
            times = []
            for _ in range(NUM_RUNS):
                start = time.perf_counter()
                _ = nn_model(dummy_half)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            stats = get_stats(times)
            results_table.append(("PyTorch GPU (FP16) - Pure Forward", stats))

    # --- CPU FP32 ---
    print(f"    -> Đang đo PyTorch CPU (FP32) (10 warmup, 30 runs)...")
    nn_model.to('cpu').float()
    dummy_cpu = torch.randn(1, 3, 640, 640, dtype=torch.float32, device='cpu')
    with torch.no_grad():
        for _ in range(10):
            _ = nn_model(dummy_cpu)
        
        times = []
        for _ in range(30):
            start = time.perf_counter()
            _ = nn_model(dummy_cpu)
            times.append((time.perf_counter() - start) * 1000)
        
        stats = get_stats(times)
        results_table.append(("PyTorch CPU (FP32) - Pure Forward", stats))

    # -------------------------------------------------------------
    # 2. PURE MODEL FORWARD PASS (ONNX Runtime)
    # -------------------------------------------------------------
    onnx_path = "bestfold1.onnx"
    if os.path.exists(onnx_path):
        print(f"\n[2] Đang đo ONNX Runtime: {onnx_path} ...")
        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
            input_name = ort_session.get_inputs()[0].name
            dummy_np = np.random.randn(1, 3, 640, 640).astype(np.float32)

            for _ in range(10):
                _ = ort_session.run(None, {input_name: dummy_np})
            
            times = []
            for _ in range(30):
                start = time.perf_counter()
                _ = ort_session.run(None, {input_name: dummy_np})
                times.append((time.perf_counter() - start) * 1000)
            
            stats = get_stats(times)
            results_table.append(("ONNX Runtime CPU (FP32) - Pure Forward", stats))
        except Exception as e:
            print(f"    [!] Lỗi ONNX: {e}")

    # -------------------------------------------------------------
    # 3. END-TO-END PIPELINE (Preprocess + Inference + NMS)
    # -------------------------------------------------------------
    print(f"\n[3] Đang đo End-to-End Predict (Preprocess + Model + NMS) trên ảnh thực tế ...")
    test_img_path = "server/uploads/sample_battery.jpg"
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
    else:
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Re-instantiate YOLO model for predict
    model_e2e = YOLO(pt_path)

    # End-to-End GPU
    if cuda_available:
        print(f"    -> Đang đo End-to-End trên GPU (20 warmup, 50 runs)...")
        for _ in range(20):
            _ = model_e2e.predict(source=img, device='0', conf=0.35, verbose=False)
        torch.cuda.synchronize()

        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = model_e2e.predict(source=img, device='0', conf=0.35, verbose=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        stats = get_stats(times)
        results_table.append(("End-to-End GPU (Pre+Infer+NMS)", stats))

    # End-to-End CPU
    print(f"    -> Đang đo End-to-End trên CPU (5 warmup, 20 runs)...")
    for _ in range(5):
        _ = model_e2e.predict(source=img, device='cpu', conf=0.35, verbose=False)

    times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = model_e2e.predict(source=img, device='cpu', conf=0.35, verbose=False)
        times.append((time.perf_counter() - start) * 1000)
    
    stats = get_stats(times)
    results_table.append(("End-to-End CPU (Pre+Infer+NMS)", stats))

    # -------------------------------------------------------------
    # SUMMARY REPORT
    # -------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"| {'Kịch bản đo lường (Scenario)':<40} | {'Trung bình':>11} | {'Trung vị':>11} | {'Min':>11} | {'P95':>11} | {'FPS':>7} |")
    print("|" + "-" * 42 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 9 + "|")
    for name, stats in results_table:
        print_result_row(name, stats)
    print("=" * 92)

if __name__ == "__main__":
    main()
