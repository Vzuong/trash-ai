import argparse
import os
from datetime import datetime

def get_parser():
    parser = argparse.ArgumentParser(description='YOLOv11 for Trash Object Detection & Recognition')
    
    # --------------------------------------------------------------------------
    # 📌 TÊN THỬ NGHIỆM / EXPERIMENT NAME
    # --------------------------------------------------------------------------
    parser.add_argument('--name', type=str, default='trash_yolo11s_rtx4060_balanced', help='Experiment folder name to save checkpoints and results')
    parser.add_argument('--append_timestamp', type=bool, default=True, help='Append date-time timestamp to folder name to avoid overwriting')
    
    # Image & Batch settings (Tối ưu hóa cho Google Colab & RTX 4060 12GB RAM)
    parser.add_argument('--img_size', default=640, type=int, help='Image size for YOLO training (e.g., 640)')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training (Tối ưu cho Google Colab & RTX 4060 12GB RAM)')
    
    # Optimizer & Training Hyperparameters
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate (AdamW)')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay L2 regularization')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate in final classification layer')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs (Set to 100 for deep convergence)')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of warmup epochs')
    parser.add_argument('--early_stop', '--patience', type=int, default=30, help='Patience for early stopping')
    parser.add_argument('--cos_lr', type=bool, default=True, help='Use Cosine learning rate scheduler')
    parser.add_argument('--close_mosaic', type=int, default=15, help='Disable mosaic augmentation for last 15 epochs to refine bounding boxes')
    
    # Anti-Overfitting Data Augmentation Hyperparameters
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic augmentation fraction')
    parser.add_argument('--mixup', type=float, default=0.15, help='Mixup augmentation fraction')
    parser.add_argument('--copy_paste', type=float, default=0.3, help='Copy-paste augmentation fraction')
    parser.add_argument('--erasing', type=float, default=0.4, help='Random erasing / cutout fraction')
    parser.add_argument('--hsv_h', type=float, default=0.015, help='HSV-Hue augmentation fraction')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='HSV-Saturation augmentation fraction')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='HSV-Value/Brightness augmentation fraction')
    parser.add_argument('--degrees', type=float, default=15.0, help='Image rotation degrees')
    parser.add_argument('--translate', type=float, default=0.1, help='Image translation fraction')
    parser.add_argument('--scale', type=float, default=0.5, help='Image scale gain fraction')
    parser.add_argument('--shear', type=float, default=10.0, help='Image shear angle degrees')
    parser.add_argument('--perspective', type=float, default=0.0005, help='Image 3D perspective fraction')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Left-right flip probability')

    # Model & Dataset paths
    parser.add_argument('--weights', type=str, default='yolo11s.pt', help='Pretrained YOLOv11 weights (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt)')
    parser.add_argument('--data_yaml', type=str, default='data_balanced.yaml', help='Path to dataset configuration YAML file')
    parser.add_argument('--dataset_path', type=str, default='Trash_dataset_balanced', help='Root path to Trash dataset directory')
    
    # System & Save settings
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID (0 for RTX 4060 / Colab GPU)')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader worker threads')
    parser.add_argument('--project', type=str, default='runs', help='Project root save directory')

    cfg = parser.parse_args()
    return cfg

def get_run_name(cfg):
    """
    Generates unique experiment run folder name using config name and optional timestamp.
    """
    if cfg.append_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{cfg.name}_{timestamp}"
    return cfg.name

def get_weights_path(cfg, run_name=None, filename="best.pt"):
    """
    Returns full path to saved checkpoint file (e.g., runs/trash_yolo11_v1_20260816/weights/best.pt)
    """
    name = run_name if run_name else cfg.name
    return os.path.join(cfg.project, name, "weights", filename)

if __name__ == '__main__':
    cfg = get_parser()
    run_name = get_run_name(cfg)
    print("Loaded Trash Recognition Configuration:")
    print(f"  Experiment Folder Name: {run_name}")
    print(f"  Best Weights Path     : {get_weights_path(cfg, run_name)}")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")
