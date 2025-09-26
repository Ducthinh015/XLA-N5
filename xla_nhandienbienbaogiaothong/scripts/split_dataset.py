import os
import shutil
import random

def split_dataset(images_dir, labels_dir, output_dir,
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                  overwrite=True, seed=42):
    # Kiểm tra tổng tỉ lệ
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Tổng tỉ lệ phải bằng 1.0 (hiện tại = {total_ratio})")

    # Xóa output cũ nếu có
    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Tạo thư mục output
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Lấy danh sách file ảnh
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.seed(seed)
    random.shuffle(image_files)

    # Tính số lượng
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Chia dataset
    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }

    # Copy file vào thư mục
    for split, files in splits.items():
        for img_file in files:
            label_file = os.path.splitext(img_file)[0] + ".txt"

            shutil.copy(os.path.join(images_dir, img_file),
                        os.path.join(output_dir, split, "images", img_file))

            if os.path.exists(os.path.join(labels_dir, label_file)):
                shutil.copy(os.path.join(labels_dir, label_file),
                            os.path.join(output_dir, split, "labels", label_file))

        print(f"✅ {split}: {len(files)} ảnh")

    print("\n📂 Dataset đã được chia thành train/val/test trong:", output_dir)

if __name__ == "__main__":
    images_path = r"D:\KI7-2025\XULIHINHANH\XLA_BTN\XLA-N5\split_dataset_v2\train\images" #chỉnh sửa lại đường dẫn
    labels_path = r"D:\KI7-2025\XULIHINHANH\XLA_BTN\XLA-N5\split_dataset_v2\train\labels"
    output_path = r"D:\KI7-2025\XULIHINHANH\XLA_BTN\XLA-N5\split_dataset_v2\output"

    split_dataset(images_path, labels_path, output_path,
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                  overwrite=True, seed=42)
