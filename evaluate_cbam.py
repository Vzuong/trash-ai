import os
import sys
import glob
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

register_cbam()

from ultralytics import YOLO

def evaluate(weights_path=None, data_yaml=None, fold=1):
    if weights_path is None:
        candidates = [
            f'/content/runs/detect/trash_yolo11s_cbam_fold{fold}/weights/best.pt',
            f'runs/detect/trash_yolo11s_cbam_fold{fold}/weights/best.pt',
            f'/content/drive/MyDrive/Trash_YOLO11_Balanced_Runs/trash_yolo11s_cbam_fold{fold}/weights/best.pt',
            'best.pt'
        ]
        weights_path = next((w for w in candidates if os.path.exists(w)), None)
        if weights_path is None:
            all_bests = glob.glob('**/best.pt', recursive=True)
            if all_bests:
                weights_path = all_bests[0]
            else:
                raise FileNotFoundError("Không tìm thấy weights best.pt")

    if data_yaml is None:
        data_candidates = [
            f'/content/data_fold{fold}.yaml',
            f'data_fold{fold}.yaml',
            '/content/data_balanced.yaml',
            'data_balanced.yaml'
        ]
        data_yaml = next((d for d in data_candidates if os.path.exists(d)), f'data_fold{fold}.yaml')

    model = YOLO(weights_path)
    res = model.val(
        data=data_yaml,
        split='val',
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=True
    )

    params = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    latency = res.speed.get("inference", 0.0)

    print(f"Parameters: {params}M")
    print(f"Precision: {res.box.mp:.4f}")
    print(f"Recall: {res.box.mr:.4f}")
    print(f"mAP50: {res.box.map50:.4f}")
    print(f"mAP50-95: {res.box.map:.4f}")
    print(f"Latency: {latency:.2f} ms")

    for i, c in enumerate(res.names.values()):
        print(f"  {c:<12}: {res.box.maps[i]:.4f}")

if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else None
    y = sys.argv[2] if len(sys.argv) > 2 else None
    f = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    evaluate(w, y, f)

