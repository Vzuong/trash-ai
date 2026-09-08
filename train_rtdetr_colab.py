"""
Training and evaluation script for RT-DETR (Real-Time Detection Transformer).
Dataset: 7-class trash classification (battery, cardboard, paper, glass, metal, plastic, organic).
Environment: Google Colab - GPU Tesla T4 (16GB VRAM) / Local PyTorch.
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
    raise ImportError("Ultralytics not found. Please install via: pip install -q ultralytics pyyaml pandas")


# ==============================================================================
# 1. DATASET SETUP & CONFIGURATION
# ==============================================================================

def setup_environment_and_data():
    """
    Mount Google Drive (if on Colab), locate dataset,
    and generate data_balanced.yaml with train, val, test splits.
    """
    print("\n" + "=" * 60)
    print(" Setting up dataset and environment...")
    print("=" * 60)

    # 1.1 Mount Google Drive if on Colab
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
    except ImportError:
        pass

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
        print(f"Extracting dataset: {zip_found} -> {extract_target}")
        os.makedirs(extract_target, exist_ok=True)
        with zipfile.ZipFile(zip_found, 'r') as z:
            z.extractall(extract_target)
        
        for root, dirs, _ in os.walk(extract_target):
            if os.path.basename(root) == expected_dataset_name:
                dataset_dir = root
                break

    assert dataset_dir is not None and os.path.exists(dataset_dir), (
        f"Dataset folder '{expected_dataset_name}' not found. Please verify path."
    )
    print(f"Dataset directory: {os.path.abspath(dataset_dir)}")

    # 1.3 Check train, val, test subdirectories
    train_dir = os.path.join(dataset_dir, 'train', 'images')
    val_dir = os.path.join(dataset_dir, 'val', 'images')
    test_dir = os.path.join(dataset_dir, 'test', 'images')

    assert os.path.exists(train_dir), f"Train dir not found: {train_dir}"
    assert os.path.exists(val_dir), f"Val dir not found: {val_dir}"
    has_test = os.path.exists(test_dir)
    if not has_test:
        test_dir = val_dir

    # 1.4 Generate data_balanced.yaml
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
    print(f"Configuration file written to: {yaml_path}")
    return yaml_path


# ==============================================================================
# 2. GPU HARDWARE CHECK
# ==============================================================================

def check_gpu_status():
    """Verify GPU availability and VRAM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB")
    else:
        print("[WARNING] CUDA GPU not detected. Running on CPU.")


# ==============================================================================
# 3. DRIVE SYNCHRONIZATION
# ==============================================================================

def safe_sync_file(src, dst):
    """Safely copy file from src to dst."""
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
    """Synchronize training artifacts to Google Drive."""
    if not drive_dir or not os.path.exists(local_dir):
        return
    try:
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copytree(local_dir, drive_dir, dirs_exist_ok=True)
        print(f"Artifacts synced to Google Drive: {drive_dir}")
    except Exception as e:
        print(f"[WARN] Artifact sync failed: {e}")


# ==============================================================================
# 4. MODEL TRAINING & TEST EVALUATION
# ==============================================================================

def train_rtdetr(args):
    data_yaml = args.data if (args.data and os.path.exists(args.data)) else setup_environment_and_data()

    check_gpu_status()

    local_project = '/content/runs/detect' if os.path.exists('/content') else 'runs/detect'
    exp_name = 'trash_rtdetr_l'
    local_save_dir = os.path.join(local_project, exp_name)
    drive_save_dir = '/content/drive/MyDrive/Trash_Runs/trash_rtdetr_l' if os.path.exists('/content/drive/MyDrive') else None

    if drive_save_dir:
        os.makedirs(drive_save_dir, exist_ok=True)
        os.makedirs(os.path.join(drive_save_dir, 'weights'), exist_ok=True)

    def on_fit_epoch_end(trainer):
        if drive_save_dir:
            for fname in ['results.csv', 'args.yaml']:
                safe_sync_file(os.path.join(local_save_dir, fname), os.path.join(drive_save_dir, fname))
            for wname in ['last.pt', 'best.pt']:
                safe_sync_file(os.path.join(local_save_dir, 'weights', wname), os.path.join(drive_save_dir, 'weights', wname))

    print("\n" + "=" * 60)
    print(" Training RT-DETR")
    print("=" * 60)
    print(f" Model weights : {args.weights}")
    print(f" Data config   : {data_yaml}")
    print(f" Epochs        : {args.epochs} (Patience: {args.patience})")
    print(f" Batch size    : {args.batch} | Imgsz: {args.imgsz}")
    print(f" Optimizer     : AdamW (lr0={args.lr0}, weight_decay=0.0005)")
    print(f" Output dir    : {local_save_dir}")
    print("=" * 60 + "\n")

    model = RTDETR(args.weights)
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

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
            amp=True,
            project=local_project,
            name=exp_name,
            exist_ok=True,
            save=True,
            verbose=True
        )
    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA out of memory. Try reducing batch size (e.g., --batch 8).")
        return
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        traceback.print_exc()
        return

    train_duration = (time.time() - start_train_time) / 3600
    print(f"\nTraining completed in {train_duration:.2f} hours.")

    sync_all_artifacts(local_save_dir, drive_save_dir)

    best_candidates = [
        os.path.join(local_save_dir, 'weights', 'best.pt'),
        os.path.join(drive_save_dir, 'weights', 'best.pt') if drive_save_dir else ''
    ]
    best_pt = next((b for b in best_candidates if b and os.path.exists(b)), None)
    assert best_pt is not None, "best.pt not found after training."
    print(f"Best checkpoint: {best_pt}")

    # ==============================================================================
    # 5. EVALUATION ON TEST SET
    # ==============================================================================
    print("\n" + "=" * 60)
    print(" Evaluating on test set...")
    print("=" * 60)

    eval_model = RTDETR(best_pt)
    test_res = eval_model.val(
        data=data_yaml,
        split='test',
        imgsz=args.imgsz,
        device=args.device if torch.cuda.is_available() else 'cpu',
        verbose=True
    )

    param_count = round(sum(p.numel() for p in eval_model.model.parameters()) / 1e6, 2)
    latency_ms = test_res.speed.get('inference', 0.0)
    fps = (1000.0 / latency_ms) if latency_ms > 0 else 0.0

    p_overall = float(test_res.box.mp)
    r_overall = float(test_res.box.mr)
    map50_overall = float(test_res.box.map50)
    map_overall = float(test_res.box.map)

    class_names = list(test_res.names.values())
    per_class_records = []

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("               RT-DETR TEST EVALUATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Model Architecture  : RT-DETR")
    report_lines.append(f"Pretrained Weights  : {args.weights}")
    report_lines.append(f"Parameters          : {param_count}M")
    report_lines.append(f"Inference Latency   : {latency_ms:.2f} ms (~{fps:.1f} FPS)")
    report_lines.append("-" * 70)
    report_lines.append(f"Precision           : {p_overall:.4f} ({p_overall*100:.2f}%)")
    report_lines.append(f"Recall              : {r_overall:.4f} ({r_overall*100:.2f}%)")
    report_lines.append(f"mAP@50              : {map50_overall:.4f} ({map50_overall*100:.2f}%)")
    report_lines.append(f"mAP@50-95           : {map_overall:.4f} ({map_overall*100:.2f}%)")
    report_lines.append("-" * 70)
    report_lines.append(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10} | {'mAP@50-95':<10}")
    report_lines.append("-" * 70)

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

    report_lines.append("=" * 70)
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_txt_path = os.path.join(local_save_dir, 'test_evaluation_report.txt')
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    df_classes = pd.DataFrame(per_class_records)
    csv_report_path = os.path.join(local_save_dir, 'test_per_class_metrics.csv')
    df_classes.to_csv(csv_report_path, index=False)

    if drive_save_dir:
        safe_sync_file(report_txt_path, os.path.join(drive_save_dir, 'test_evaluation_report.txt'))
        safe_sync_file(csv_report_path, os.path.join(drive_save_dir, 'test_per_class_metrics.csv'))
        print(f"Evaluation report saved to: {drive_save_dir}")


# ==============================================================================
# 6. ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate RT-DETR model")
    parser.add_argument('--weights', type=str, default='rtdetr-l.pt', help='Pretrained weights (e.g. rtdetr-l.pt, rtdetr-r18.pt)')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--lr0', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index')
    parser.add_argument('--data', type=str, default=None, help='Path to data yaml')
    
    args = parser.parse_args()
    train_rtdetr(args)
