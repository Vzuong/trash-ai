import os
import sys
import glob
import shutil
import pandas as pd
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

def main():
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive')
    except ImportError:
        pass

    drive_candidates = [
        '/content/drive/MyDrive/Trash_YOLO11_Balanced_Runs/trash_yolo11s_cbam_custom',
        '/content/drive/MyDrive/Trash_YOLO11_Balanced_Runs/trash_yolov8s_balanced',
        '/content/drive/MyDrive/trash_kfold/fold1_run'
    ]
    
    drive_exp_dir = next((c for c in drive_candidates if os.path.exists(os.path.join(c, 'weights', 'last.pt'))), None)
    if drive_exp_dir is None:
        all_lasts = glob.glob('/content/drive/MyDrive/**/last.pt', recursive=True)
        if all_lasts:
            drive_exp_dir = os.path.dirname(os.path.dirname(all_lasts[0]))
        else:
            raise FileNotFoundError("Không tìm thấy last.pt trên Google Drive")

    exp_name = os.path.basename(drive_exp_dir)
    local_exp_dir = f'/content/runs/detect/{exp_name}' if os.path.exists('/content') else f'runs/detect/{exp_name}'
    os.makedirs(os.path.join(local_exp_dir, 'weights'), exist_ok=True)

    src_last = os.path.join(drive_exp_dir, 'weights', 'last.pt')
    dst_last = os.path.join(local_exp_dir, 'weights', 'last.pt')
    shutil.copy2(src_last, dst_last)

    src_args = os.path.join(drive_exp_dir, 'args.yaml')
    if os.path.exists(src_args):
        shutil.copy2(src_args, os.path.join(local_exp_dir, 'args.yaml'))

    csv_candidates = glob.glob(os.path.join(drive_exp_dir, 'results*.csv'))
    if csv_candidates:
        dfs = []
        for csv_f in csv_candidates:
            if os.path.getsize(csv_f) > 0:
                try:
                    df = pd.read_csv(csv_f)
                    df.columns = df.columns.str.strip()
                    dfs.append(df)
                except Exception:
                    pass
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            merged = merged.drop_duplicates(subset=['epoch']).sort_values('epoch')
            merged.to_csv(os.path.join(local_exp_dir, 'results.csv'), index=False)

    # Resume automatically reads the dataset config path from args.yaml

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

        try:
            os.makedirs(drive_exp_dir, exist_ok=True)
            os.makedirs(os.path.join(drive_exp_dir, 'weights'), exist_ok=True)
            for fname in ['results.csv', 'args.yaml']:
                src = os.path.join(local_exp_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(drive_exp_dir, fname))
            for wname in ['last.pt', 'best.pt']:
                src = os.path.join(local_exp_dir, 'weights', wname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(drive_exp_dir, 'weights', wname))
        except Exception:
            pass

    model = YOLO(dst_last)
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
    model.train(resume=True)

    if os.path.exists(local_exp_dir):
        shutil.copytree(local_exp_dir, drive_exp_dir, dirs_exist_ok=True)

if __name__ == '__main__':
    main()
