import os
import sys
import glob
import shutil
import random
import uuid
import argparse
from collections import Counter, defaultdict
import cv2
import numpy as np
from PIL import Image, ImageEnhance

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CLASS_NAMES = {
    0: 'battery',
    1: 'cardboard',
    2: 'paper',
    3: 'glass',
    4: 'metal',
    5: 'plastic',
    6: 'organic'
}

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def scan_dataset(data_dir, split="train"):
    """
    Scans images and YOLO labels, counting bounding boxes per class.
    """
    img_dir = os.path.join(data_dir, split, "images")
    lbl_dir = os.path.join(data_dir, split, "labels")

    if not os.path.exists(img_dir):
        img_dir = os.path.join(data_dir, "images", split)
        lbl_dir = os.path.join(data_dir, "labels", split)

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        return {}, {}, {}

    images = {}
    class_counts = Counter()
    img_to_classes = defaultdict(set)
    img_to_boxes = defaultdict(list)

    for img_file in os.listdir(img_dir):
        base, ext = os.path.splitext(img_file)
        if ext.lower() in VALID_EXTS:
            img_path = os.path.join(img_dir, img_file)
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            
            boxes = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            coords = [float(p) for p in parts[1:5]]
                            boxes.append((cls_id, coords))
                            class_counts[cls_id] += 1
                            img_to_classes[base].add(cls_id)
                            
            images[base] = {
                'img_path': img_path,
                'lbl_path': lbl_path,
                'boxes': boxes,
                'ext': ext
            }
            img_to_boxes[base] = boxes

    return images, class_counts, img_to_classes

def augment_image_and_boxes(image_np, boxes):
    """
    Applies random realistic data augmentations to image and adjusts YOLO bounding boxes.
    """
    h, w = image_np.shape[:2]
    aug_img = image_np.copy()
    new_boxes = []

    # 1. Random Horizontal Flip (50% chance)
    h_flip = random.random() > 0.5
    if h_flip:
        aug_img = cv2.flip(aug_img, 1)

    # 2. Random Brightness & Contrast
    alpha = random.uniform(0.75, 1.25) # Contrast
    beta = random.uniform(-25, 25)     # Brightness
    aug_img = np.clip(alpha * aug_img + beta, 0, 255).astype(np.uint8)

    # 3. Random Color Jitter (HSV)
    if random.random() > 0.4:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180 # Hue
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255) # Saturation
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4. Slight Gaussian Blur (30% chance) to simulate motion/camera focus
    if random.random() > 0.7:
        ksize = random.choice([3, 5])
        aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)

    # Transform bounding boxes
    for cls_id, (xc, yc, bw, bh) in boxes:
        if h_flip:
            xc = 1.0 - xc
        
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        new_boxes.append((cls_id, [xc, yc, bw, bh]))

    return aug_img, new_boxes

def balance_dataset(
    source_dir="Trash_dataset",
    output_dir="Trash_dataset_balanced",
    target_boxes_per_class=3800,
    min_boxes_per_class=None,
    split="train",
    do_augmentation=True
):
    if min_boxes_per_class is None:
        min_boxes_per_class = int(target_boxes_per_class * 0.92)

    print("=" * 80)
    print(" ⚖️ BẮT ĐẦU CÂN BẰNG DATASET PHÂN LOẠI RÁC (YOLO DATASET BALANCER)")
    print(f"  Thư mục nguồn (Source)       : {source_dir}")
    print(f"  Thư mục đích (Target)        : {output_dir}")
    print(f"  Số box mục tiêu mỗi lớp      : ~{target_boxes_per_class} boxes")
    print(f"  Tập xử lý (Split)            : {split}")
    print(f"  Tăng cường mẫu thiếu (Aug)   : {'BẬT (Tự động bù đắp các lớp chưa đủ)' if do_augmentation else 'TẮT'}")
    print("=" * 80)

    # Step 1: Scan Source Dataset
    images, class_counts, img_to_classes = scan_dataset(source_dir, split=split)
    if not images:
        print(f"[ERROR] Không tìm thấy dữ liệu ảnh/nhãn trong {source_dir}/{split}!")
        return

    print(f"\n📊 [1/4] PHÂN BỐ DỮ LIỆU HIỆN TẠI (Trước khi cân bằng):")
    print("-" * 60)
    print(f"{'Class ID':<10} | {'Tên lớp':<12} | {'Số lượng Box':<15} | {'Trạng thái'}")
    print("-" * 60)
    for c_id in sorted(CLASS_NAMES.keys()):
        c_name = CLASS_NAMES[c_id]
        cnt = class_counts[c_id]
        if cnt > target_boxes_per_class * 1.3:
            status = "🔴 Quá dư (Cần cắt bớt)"
        elif cnt < min_boxes_per_class:
            status = "🟡 Thiếu (Cần tăng cường)"
        else:
            status = "🟢 Đạt chuẩn cân bằng"
        print(f"{c_id:<10} | {c_name:<12} | {cnt:<15} | {status}")
    print("-" * 60)
    print(f"👉 Tổng số ảnh hiện tại: {len(images)} ảnh | Tổng boxes: {sum(class_counts.values())}")

    # Prepare Target Dirs
    target_img_dir = os.path.join(output_dir, split, "images")
    target_lbl_dir = os.path.join(output_dir, split, "labels")
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lbl_dir, exist_ok=True)

    # Step 2: Intelligent Undersampling for Over-represented classes (e.g., cardboard, organic)
    # Strategy: Keep ALL multi-class images that contain rare classes (battery, glass, plastic, etc.).
    # Only trim single-class images belonging to over-represented classes.
    
    # Step 2: Intelligent Multi-Class Balancing
    # Instead of deleting images sequentially, we greedily select images to preserve balance
    print(f"\n✂️ [2/4] ĐANG CÂN BẰNG TẬP DỮ LIỆU VỀ MỨC MỤC TIÊU ~{target_boxes_per_class} BOXES/LỚP...")
    
    # Sort classes by count ascending (least frequent first)
    sorted_classes = sorted(CLASS_NAMES.keys(), key=lambda c: class_counts[c])
    
    selected_images = set()
    current_counts = Counter()

    # Pass 1: For each class, pick images until target_boxes_per_class is reached
    random.seed(42)
    for c_id in sorted_classes:
        # Find all images containing this class that haven't been selected yet
        c_images = [base for base, c_set in img_to_classes.items() if c_id in c_set and base not in selected_images]
        random.shuffle(c_images)

        for base in c_images:
            if current_counts[c_id] >= target_boxes_per_class:
                break
            
            # Add this image
            selected_images.add(base)
            for cid, _ in images[base]['boxes']:
                current_counts[cid] += 1

    kept_images = selected_images
    discarded_count = len(images) - len(kept_images)
    print(f"  -> Đã chọn {len(kept_images)} ảnh tối ưu, loại bỏ {discarded_count} ảnh dư thừa.")
    print(f"  -> Phân bố sau khi lọc: {[f'{CLASS_NAMES[k]}: {current_counts[k]}' for k in sorted(CLASS_NAMES.keys())]}")

    # Copy Kept Images & Labels to output
    print(f"\n💾 [3/4] ĐANG SAO CHÉP & TĂNG CƯỜNG DỮ LIỆU SANG THƯ MỤC CÂN BẰNG...")
    final_class_counts = Counter()
    copied_count = 0

    for base in kept_images:
        data = images[base]
        dest_img = os.path.join(target_img_dir, base + data['ext'])
        dest_lbl = os.path.join(target_lbl_dir, base + ".txt")

        shutil.copy2(data['img_path'], dest_img)
        shutil.copy2(data['lbl_path'], dest_lbl)
        copied_count += 1

        for c_id, _ in data['boxes']:
            final_class_counts[c_id] += 1

    # Step 3: Offline Data Augmentation for Under-represented classes (battery, glass, plastic)
    augmented_count = 0
    if do_augmentation:
        under_represented = [c for c in CLASS_NAMES.keys() if final_class_counts[c] < min_boxes_per_class]
        
        for under_cls in under_represented:
            needed = target_boxes_per_class - final_class_counts[under_cls]
            if needed <= 0:
                continue

            # Find all images containing this rare class
            source_candidates = [b for b in kept_images if under_cls in img_to_classes[b]]
            if not source_candidates:
                continue

            print(f"  ⚡ Tăng cường lớp '{CLASS_NAMES[under_cls]}' (Hiện có: {final_class_counts[under_cls]} -> Cần thêm: ~{needed} boxes)...")

            # Multiply / Augment images with random variations
            aug_idx = 0
            while final_class_counts[under_cls] < target_boxes_per_class and aug_idx < 10000:
                base = random.choice(source_candidates)
                data = images[base]

                img_cv = cv2.imread(data['img_path'])
                if img_cv is None:
                    aug_idx += 1
                    continue

                aug_img_cv, new_boxes = augment_image_and_boxes(img_cv, data['boxes'])
                
                # Check if rare class still exists in new boxes
                rare_in_aug = [c for c, _ in new_boxes if c == under_cls]
                if not rare_in_aug:
                    aug_idx += 1
                    continue

                # Save new augmented file
                unique_suffix = uuid.uuid4().hex[:6]
                aug_base = f"aug_{CLASS_NAMES[under_cls]}_{base[:20]}_{unique_suffix}"
                aug_img_path = os.path.join(target_img_dir, aug_base + data['ext'])
                aug_lbl_path = os.path.join(target_lbl_dir, aug_base + ".txt")

                cv2.imwrite(aug_img_path, aug_img_cv)
                with open(aug_lbl_path, 'w', encoding='utf-8') as f:
                    for cid, coords in new_boxes:
                        f.write(f"{cid} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")

                for cid, _ in new_boxes:
                    final_class_counts[cid] += 1
                
                augmented_count += 1
                aug_idx += 1

    # Also copy val & test splits untouched to maintain objective evaluation
    for other_split in ["val", "test"]:
        other_src_img = os.path.join(source_dir, other_split, "images")
        other_src_lbl = os.path.join(source_dir, other_split, "labels")
        if os.path.exists(other_src_img) and os.path.exists(other_src_lbl):
            other_dst_img = os.path.join(output_dir, other_split, "images")
            other_dst_lbl = os.path.join(output_dir, other_split, "labels")
            os.makedirs(other_dst_img, exist_ok=True)
            os.makedirs(other_dst_lbl, exist_ok=True)
            for f in os.listdir(other_src_img):
                shutil.copy2(os.path.join(other_src_img, f), os.path.join(other_dst_img, f))
            for f in os.listdir(other_src_lbl):
                shutil.copy2(os.path.join(other_src_lbl, f), os.path.join(other_dst_lbl, f))

    # Generate data_balanced.yaml
    yaml_content = f"""# YOLOv11 Balanced Trash Recognition Dataset Configuration
path: {os.path.abspath(output_dir)}
train: {split}/images
val: val/images
test: test/images

# Number of classes
nc: 7

# Class Names
names:
  0: battery
  1: cardboard
  2: paper
  3: glass
  4: metal
  5: plastic
  6: organic
"""
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "data_balanced.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    # Step 4: Final Summary Report
    print("\n" + "=" * 80)
    print(" 🎉 HOÀN THÀNH CÂN BẰNG DATASET!")
    print("=" * 80)
    print(f"{'Class ID':<10} | {'Tên lớp':<12} | {'Trước (Old)':<14} | {'Sau (Balanced)':<16} | {'Tỷ lệ %'}")
    print("-" * 80)
    total_final_boxes = sum(final_class_counts.values())
    for c_id in sorted(CLASS_NAMES.keys()):
        c_name = CLASS_NAMES[c_id]
        b_old = class_counts[c_id]
        b_new = final_class_counts[c_id]
        pct = (b_new / total_final_boxes * 100) if total_final_boxes > 0 else 0
        print(f"{c_id:<10} | {c_name:<12} | {b_old:<14} | {b_new:<16} | {pct:.1f}%")
    print("-" * 80)
    print(f"👉 Tổng ảnh train mới     : {len(os.listdir(target_img_dir))} ảnh")
    print(f"👉 File cấu hình dataset  : {yaml_path}")
    print(f"👉 Câu lệnh train ngay    : python train.py --data data_balanced.yaml --epochs 50")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cân bằng Dataset Phân Loại Rác YOLO (Smart Undersampling & Augmentation)')
    parser.add_argument('--source', type=str, default='Trash_dataset', help='Thư mục dataset gốc')
    parser.add_argument('--output', type=str, default='Trash_dataset_balanced', help='Thư mục lưu dataset đã cân bằng')
    parser.add_argument('--target_boxes', type=int, default=2800, help='Số lượng bounding box mục tiêu cho mỗi lớp (mặc định: 2800)')
    parser.add_argument('--no_aug', action='store_true', help='Chỉ cắt bớt lớp dư thừa, không tăng cường lớp thiếu')
    args = parser.parse_args()

    balance_dataset(
        source_dir=args.source,
        output_dir=args.output,
        target_boxes_per_class=args.target_boxes,
        do_augmentation=not args.no_aug
    )
