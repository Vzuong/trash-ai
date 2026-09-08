import torch
import torch.nn as nn
import inspect
import sys

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
        setattr(sys.modules['__main__'], 'ChannelAttention', ChannelAttention)
        setattr(sys.modules['__main__'], 'SpatialAttention', SpatialAttention)

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
