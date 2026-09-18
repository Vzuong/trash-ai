import sys
import os
import glob
import torch
import torch.nn as nn

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Giả lập class CBAM để torch.load đọc an toàn không bị lỗi pickle
class CBAM(nn.Module):
    pass
class ChannelAttention(nn.Module):
    pass
class SpatialAttention(nn.Module):
    pass

setattr(sys.modules['__main__'], 'CBAM', CBAM)
setattr(sys.modules['__main__'], 'ChannelAttention', ChannelAttention)
setattr(sys.modules['__main__'], 'SpatialAttention', SpatialAttention)

def check_pt_file(path):
    if not os.path.isfile(path) or not path.lower().endswith('.pt'):
        return
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(ckpt, dict):
            return
        args = ckpt.get('train_args', {})
        if hasattr(args, '__dict__'):
            args = vars(args)
        elif not isinstance(args, dict):
            args = {}
        
        name = str(args.get('name', 'N/A'))
        data = str(args.get('data', 'N/A'))
        date = str(ckpt.get('date', 'N/A'))
        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        # Nhận diện Fold
        fold_detected = "Mô hình khác (Không phải K-Fold)"
        if 'fold1' in name.lower() or 'fold1' in data.lower():
            fold_detected = "⭐ FOLD 1 (Mô hình chính)"
        elif 'fold2' in name.lower() or 'fold2' in data.lower():
            fold_detected = "🔥 FOLD 2 (Tìm thấy)"
        elif 'fold3' in name.lower() or 'fold3' in data.lower():
            fold_detected = "🚀 FOLD 3 (Tìm thấy)"
            
        print("-" * 75)
        print(f"📂 File: {os.path.basename(path)} ({size_mb:.2f} MB)")
        print(f"   Đường dẫn: {os.path.abspath(path)}")
        print(f"   🎯 NHẬN DIỆN    : {fold_detected}")
        print(f"   - Tên Run       : {name}")
        print(f"   - Dataset YAML  : {data}")
        print(f"   - Ngày train    : {date}")
    except Exception as e:
        print(f"[-] Lỗi đọc {os.path.basename(path)}: {e}")

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else '.'
    print("=" * 75)
    print("      CÔNG CỤ PHÂN LOẠI CHECKPOINT K-FOLD (YOLO / CBAM)")
    print(f"      Đang quét: {os.path.abspath(target)}")
    print("=" * 75)
    
    if os.path.isfile(target):
        check_pt_file(target)
    elif os.path.isdir(target):
        files = glob.glob(os.path.join(target, '**', '*.pt'), recursive=True)
        if not files:
            files = glob.glob(os.path.join(target, '*.pt'))
        for f in sorted(files):
            check_pt_file(f)
    print("\n" + "=" * 75)
    print("Hoàn tất quét!")
