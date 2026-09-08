import argparse
import os
from datetime import datetime

def get_parser():
    parser = argparse.ArgumentParser(description='YOLOv11 for Trash Object Detection')
    
    # Experiment name
    parser.add_argument('--name', type=str, default='trash_yolo11_balanced', help='Experiment folder name')
    parser.add_argument('--append_timestamp', type=bool, default=True, help='Append timestamp to folder name')
    
    # Image & Batch settings
    parser.add_argument('--img_size', default=640, type=int, help='Input image size')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    
    # Training hyperparameters
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay L2 regularization')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of warmup epochs')
    parser.add_argument('--early_stop', '--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--cos_lr', type=bool, default=True, help='Cosine learning rate scheduler')
    
    # Data Augmentation
    parser.add_argument('--mixup', type=float, default=0.15, help='Mixup fraction')
    parser.add_argument('--copy_paste', type=float, default=0.3, help='Copy-paste fraction')
    parser.add_argument('--erasing', type=float, default=0.4, help='Random erasing fraction')
    parser.add_argument('--hsv_h', type=float, default=0.015, help='HSV-Hue fraction')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='HSV-Saturation fraction')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='HSV-Value fraction')
    parser.add_argument('--degrees', type=float, default=15.0, help='Rotation degrees')
    parser.add_argument('--translate', type=float, default=0.1, help='Translation fraction')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale fraction')
    parser.add_argument('--shear', type=float, default=10.0, help='Shear angle degrees')
    parser.add_argument('--perspective', type=float, default=0.0005, help='Perspective fraction')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Horizontal flip probability')

    # Model & Dataset paths
    parser.add_argument('--weights', type=str, default='yolo11s.pt', choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt'], help='Pretrained weights')
    parser.add_argument('--data_yaml', type=str, default='data_balanced.yaml', help='Path to dataset yaml')
    parser.add_argument('--dataset_path', type=str, default='Trash_dataset_balanced', help='Path to dataset directory')
    
    # System settings
    parser.add_argument('--gpu', type=str, default='0', help='CUDA device id')
    parser.add_argument('--workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--project', type=str, default='runs', help='Project save directory')

    cfg = parser.parse_args()
    return cfg

def get_run_name(cfg):
    model_tag = os.path.splitext(os.path.basename(cfg.weights))[0] if hasattr(cfg, 'weights') and cfg.weights else 'yolo11'
    base_name = cfg.name
    if 'yolo11' in base_name and model_tag not in base_name:
        base_name = base_name.replace('yolo11', model_tag)
    elif base_name == 'trash_yolo11_balanced':
        base_name = f"trash_{model_tag}_balanced"

    if cfg.append_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"
    return base_name

def get_weights_path(cfg, run_name=None, filename="best.pt"):
    name = run_name if run_name else cfg.name
    return os.path.join(cfg.project, name, "weights", filename)

if __name__ == '__main__':
    cfg = get_parser()
    run_name = get_run_name(cfg)
    print("Configuration:")
    print(f"  Run name     : {run_name}")
    print(f"  Weights path : {get_weights_path(cfg, run_name)}")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")
