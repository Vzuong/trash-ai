import os
import sys
import glob
import shutil
import zipfile
import torch
import torch.nn as nn
import inspect

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

def register_cbam():
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block
    import ultralytics.nn.tasks as tasks

    setattr(modules, 'CBAM', CBAM)
    setattr(block, 'CBAM', CBAM)
    setattr(tasks, 'CBAM', CBAM)
    tasks.__dict__['CBAM'] = CBAM
    
    if '__main__' in sys.modules:
        setattr(sys.modules['__main__'], 'CBAM', CBAM)

    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([CBAM, ChannelAttention, SpatialAttention])

    try:
        src = inspect.getsource(tasks.parse_model)
        if 'CBAM,' not in src:
            src_patched = src.replace('C2PSA,', 'C2PSA, CBAM,')
            exec_globals = tasks.__dict__
            exec(src_patched, exec_globals)
            tasks.parse_model = exec_globals['parse_model']
    except Exception:
        pass

register_cbam()

from ultralytics import YOLO

import random
import yaml
import argparse

def setup_data(fold: int = 1):
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive')
    except ImportError:
        pass

    local_root = '/content/trash_project/Trash_dataset_balanced'
    if not os.path.exists(local_root):
        zip_candidates = [
            '/content/drive/MyDrive/Trash (3).zip',
            '/content/drive/MyDrive/trash (3).zip',
            'Trash (3).zip'
        ]
        zip_file = next((z for z in zip_candidates if os.path.exists(z)), None)
        if zip_file:
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall('/content/trash_project')

    dataset_dir = None
    for search_path in ['/content/trash_project', './Trash_dataset_balanced', '../Trash_dataset_balanced']:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                if os.path.basename(root) == 'Trash_dataset_balanced':
                    dataset_dir = root
                    break
        if dataset_dir:
            break

    assert dataset_dir is not None and os.path.exists(dataset_dir), "Không tìm thấy thư mục Trash_dataset_balanced!"

    all_imgs = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
        all_imgs.extend(glob.glob(f'{dataset_dir}/**/images/{ext}', recursive=True))
    all_imgs = sorted(list(set(all_imgs)))
    assert len(all_imgs) > 0, "Không tìm thấy ảnh nào trong dataset!"

    random.seed(42)
    random.shuffle(all_imgs)

    n = len(all_imgs)
    f1, f2 = n // 3, 2 * (n // 3)
    folds = [all_imgs[:f1], all_imgs[f1:f2], all_imgs[f2:]]

    prefix = '/content' if os.path.exists('/content') else '.'
    val_imgs = folds[fold - 1]
    train_imgs = [img for i, f in enumerate(folds) if i != (fold - 1) for img in f]

    train_txt = os.path.join(prefix, f'train_fold{fold}.txt')
    val_txt = os.path.join(prefix, f'val_fold{fold}.txt')
    with open(train_txt, 'w') as f:
        f.writelines(f'{p}\n' for p in train_imgs)
    with open(val_txt, 'w') as f:
        f.writelines(f'{p}\n' for p in val_imgs)

    class_names = ['battery', 'cardboard', 'paper', 'glass', 'metal', 'plastic', 'organic']
    yaml_dict = {
        'train': os.path.abspath(train_txt),
        'val': os.path.abspath(val_txt),
        'nc': 7,
        'names': {i: name for i, name in enumerate(class_names)}
    }
    yaml_path = os.path.join(prefix, f'data_fold{fold}.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"Fold {fold}: Train={len(train_imgs)}, Val={len(val_imgs)} (Tổng={len(all_imgs)}) -> {yaml_path}")
    return yaml_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3])
    args, _ = parser.parse_known_args()
    fold = args.fold

    yaml_data = setup_data(fold=fold)
    
    model_cfg = 'models/yolo11s-cbam.yaml'
    if not os.path.exists(model_cfg):
        candidates = glob.glob('**/yolo11s-cbam.yaml', recursive=True)
        if candidates:
            model_cfg = candidates[0]

    model = YOLO(model_cfg)
    
    weights = 'yolo11s.pt'
    if not os.path.exists(weights):
        weights = '/content/yolo11s.pt' if os.path.exists('/content/yolo11s.pt') else 'yolo11s.pt'
    model.load(weights)

    local_project = '/content/runs/detect' if os.path.exists('/content') else 'runs/detect'
    exp_name = f'trash_yolo11s_cbam_fold{fold}'
    local_save_dir = os.path.join(local_project, exp_name)
    drive_save_dir = f'/content/drive/MyDrive/Trash_YOLO11_Balanced_Runs/{exp_name}' if os.path.exists('/content/drive/MyDrive') else None

    def on_fit_epoch_end(trainer):
        if hasattr(trainer, 'stopper') and trainer.stopper is not None:
            delta = trainer.epoch - trainer.stopper.best_epoch
            if delta == 5 and not getattr(trainer, '_lr_reduced_on_plateau', False):
                trainer._lr_reduced_on_plateau = True
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                if hasattr(trainer, 'scheduler') and hasattr(trainer.scheduler, 'base_lrs'):
                    trainer.scheduler.base_lrs = [b * 0.5 for b in trainer.scheduler.base_lrs]
                lr = trainer.optimizer.param_groups[0]['lr']
                print(f"[ReduceLROnPlateau] Reducing LR to: {lr:.6f}")
            elif delta < 5:
                trainer._lr_reduced_on_plateau = False

        if drive_save_dir:
            try:
                os.makedirs(drive_save_dir, exist_ok=True)
                os.makedirs(os.path.join(drive_save_dir, 'weights'), exist_ok=True)
                for fname in ['results.csv', 'args.yaml']:
                    src = os.path.join(local_save_dir, fname)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(drive_save_dir, fname))
                for wname in ['last.pt', 'best.pt']:
                    src = os.path.join(local_save_dir, 'weights', wname)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(drive_save_dir, 'weights', wname))
            except Exception:
                pass

    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

    model.train(
        data=yaml_data,
        epochs=100,
        batch=32,
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=2,
        optimizer='AdamW',
        lr0=0.001,
        patience=10,
        cos_lr=False,
        weight_decay=0.0005,
        dropout=0.1,
        mixup=0.15,
        copy_paste=0.3,
        erasing=0.4,
        project=local_project,
        name=exp_name,
        exist_ok=True,
        save=True,
        verbose=True
    )

    if drive_save_dir and os.path.exists(local_save_dir):
        shutil.copytree(local_save_dir, drive_save_dir, dirs_exist_ok=True)

    best_weights = os.path.join(local_save_dir, 'weights', 'best.pt')
    if not os.path.exists(best_weights) and drive_save_dir:
        best_weights = os.path.join(drive_save_dir, 'weights', 'best.pt')

    if os.path.exists(best_weights):
        eval_model = YOLO(best_weights)
        val_res = eval_model.val(
            data=yaml_data,
            split='val',
            imgsz=640,
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=True
        )
        print(f"Precision: {val_res.box.mp:.4f}")
        print(f"Recall: {val_res.box.mr:.4f}")
        print(f"mAP50: {val_res.box.map50:.4f}")
        print(f"mAP50-95: {val_res.box.map:.4f}")
        print(f"Latency: {val_res.speed.get('inference', 0.0):.2f} ms")

if __name__ == '__main__':
    main()
