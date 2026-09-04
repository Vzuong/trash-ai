import os
import time
import shutil
import torch
from config import get_parser, get_run_name, get_weights_path
from ultralytics import YOLO

# --- Custom Callbacks for Epoch & Total Training Time Tracking ---
def on_train_start(trainer):
    trainer.start_time = time.time()
    print("\n⏱️ [TIMER] Training started...")
    print("=" * 105)
    print(f"{'Epoch':^10} | {'Train Loss':^15} | {'Precision':^10} | {'Recall':^10} | {'mAP50':^10} | {'mAP50-95':^10} | {'Epoch Time':^12}")
    print("=" * 105)

def on_epoch_start(trainer):
    trainer.epoch_start_time = time.time()

def on_epoch_end(trainer):
    epoch_time = time.time() - getattr(trainer, 'epoch_start_time', time.time())
    current_epoch = trainer.epoch + 1
    total_epochs = trainer.epochs
    mins = int(epoch_time // 60)
    secs = epoch_time % 60
    time_str = f"{mins}m {secs:.1f}s"
    
    # Extract train loss
    tloss = getattr(trainer, 'tloss', None)
    if tloss is not None and len(tloss) >= 3:
        train_box, train_cls, train_dfl = tloss[0].item(), tloss[1].item(), tloss[2].item()
        total_train_loss = train_box + train_cls + train_dfl
        train_str = f"{total_train_loss:.4f}"
    else:
        train_str = "N/A"

    # Extract validation metrics
    metrics = getattr(trainer, 'metrics', {})
    if isinstance(metrics, dict):
        map50 = metrics.get('metrics/mAP50(B)', 0.0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)
        precision = metrics.get('metrics/precision(B)', 0.0)
        recall = metrics.get('metrics/recall(B)', 0.0)
    else:
        map50 = getattr(metrics, 'map50', 0.0)
        map50_95 = getattr(metrics, 'map', 0.0)
        precision = getattr(metrics, 'mp', 0.0)
        recall = getattr(metrics, 'mr', 0.0)

    print(f"{current_epoch:3d}/{total_epochs:<6} | {train_str:^15} | {precision:^10.4f} | {recall:^10.4f} | {map50:^10.4f} | {map50_95:^10.4f} | {time_str:^12}")

def on_train_end(trainer):
    print("=" * 105)
    total_time = time.time() - getattr(trainer, 'start_time', time.time())
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"⏱️ TOTAL TRAINING TIME: {hours}h {minutes}m {seconds:.2f}s ({total_time:.2f}s)")
    print("=" * 105)

def train_yolo(cfg):
    """
    Trains YOLOv11 Object Detection model for Trash Recognition using PyTorch / Ultralytics.
    Uses centralized experiment name from config.py and saves dedicated best_loss.pt checkpoint.
    """
    exp_name = get_run_name(cfg)

    print("=" * 65)
    print(f"Starting YOLOv11 Trash Recognition Training...")
    print(f"Experiment Folder Name : {exp_name}")
    print(f"Dataset YAML           : {cfg.data_yaml}")
    print(f"Image Size             : {cfg.img_size} | Batch Size: {cfg.batch_size}")
    print(f"Epochs                 : {cfg.epochs}")
    print(f"Early Stop Patience    : {cfg.early_stop} epochs without improvement")
    print("=" * 65)

    # 1. Check if data.yaml exists
    if not os.path.exists(cfg.data_yaml):
        print(f"[ERROR] Data config file '{cfg.data_yaml}' not found!")
        return

    # 2. Load YOLOv11 model
    try:
        model = YOLO(cfg.weights)
    except Exception as e:
        print(f"[ERROR] Could not load YOLOv11 weights '{cfg.weights}': {e}")
        return

    # Add Callbacks for Epoch & Total Timing
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_epoch_start", on_epoch_start)
    model.add_callback("on_epoch_end", on_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    # 3. Start training with Early Stopping (patience=cfg.early_stop)
    results = model.train(
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.img_size,
        batch=cfg.batch_size,
        lr0=cfg.lr,
        weight_decay=cfg.weight_decay,
        dropout=cfg.dropout,
        device=cfg.gpu,
        workers=cfg.workers,
        project=cfg.project,
        name=exp_name,
        patience=cfg.early_stop,
        cos_lr=cfg.cos_lr,
        close_mosaic=cfg.close_mosaic,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        erasing=cfg.erasing,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=cfg.shear,
        perspective=cfg.perspective,
        fliplr=cfg.fliplr,
        save=True,
        save_period=1,
        verbose=False
    )

    trainer_obj = getattr(model, 'trainer', None)
    save_dir = getattr(trainer_obj, 'save_dir', None) if trainer_obj else None
    if not save_dir:
        save_dir = os.path.join(cfg.project, exp_name)
    else:
        save_dir = str(save_dir)

    weights_dir = os.path.join(save_dir, "weights")
    best_weights = os.path.join(weights_dir, "best.pt")
    
    # Save a copy as best_loss.pt and dedicated named weights
    best_loss_copy = os.path.join(weights_dir, "best_loss.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, best_loss_copy)

    print("=" * 65)
    print(f"[SUCCESS] Training completed successfully!")
    print(f"  🏆 Best Checkpoint (mAP/Loss) : {best_weights}")
    print(f"  📌 Dedicated Best Loss Copy   : {best_loss_copy}")
    print(f"  🔄 Last Checkpoint            : {os.path.join(weights_dir, 'last.pt')}")
    print(f"  📊 Training Logs & Results    : {save_dir}")
    print("=" * 65)

    # Automatic quantitative evaluation summary on Best Weights
    if os.path.exists(best_weights):
        print("\n📊 [EVALUATION] Calculating quantitative metrics on Best Checkpoint...")
        try:
            val_model = YOLO(best_weights)
            val_res = val_model.val(data=cfg.data_yaml, imgsz=cfg.img_size, device=cfg.gpu, verbose=False)
            
            p = val_res.box.mp
            r = val_res.box.mr
            map50 = val_res.box.map50
            map50_95 = val_res.box.map

            print("\n" + "=" * 65)
            print(" 🎯 BẢNG SỐ LIỆU ĐÁNH GIÁ MÔ HÌNH (FINAL EVALUATION METRICS)")
            print("=" * 65)
            print(f"  • Precision (Độ chính xác)      : {p:.4f} ({p*100:.2f}%)")
            print(f"  • Recall (Độ bao phủ / nhạy)     : {r:.4f} ({r*100:.2f}%)")
            print(f"  • mAP@50 (Mean Avg Precision)    : {map50:.4f} ({map50*100:.2f}%)")
            print(f"  • mAP@50-95                      : {map50_95:.4f} ({map50_95*100:.2f}%)")
            print("-" * 65)
            print(f" 📈 BIỂU ĐỒ KẾT QUẢ TỰ ĐỘNG TẠO TẠI THƯ MỤC: {save_dir}")
            print(f"   • Biểu đồ loss & mAP qua các epoch : {os.path.join(save_dir, 'results.png')}")
            print(f"   • Ma trận nhầm lẫn (Confusion)     : {os.path.join(save_dir, 'confusion_matrix.png')}")
            print(f"   • Đường cong Precision - Recall    : {os.path.join(save_dir, 'BoxPR_curve.png')}")
            print(f"   • Đường cong F1-Score              : {os.path.join(save_dir, 'BoxF1_curve.png')}")
            print(f"   • File dữ liệu thô chi tiết CSV    : {os.path.join(save_dir, 'results.csv')}")
            print("=" * 65)
        except Exception as e:
            print(f"[WARNING] Could not calculate val metrics: {e}")

    return results

def main():
    cfg = get_parser()
    train_yolo(cfg)

if __name__ == '__main__':
    main()
