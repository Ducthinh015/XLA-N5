"""
Script chia dataset thành train/val/test
Sử dụng cấu hình từ config.py hoặc environment variables
"""
import os
import shutil
import random
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO
import argparse

def split_dataset(images_dir, labels_dir, output_dir,
                  train_ratio=None, val_ratio=None, test_ratio=None,
                  overwrite=True, seed=42):
    # Sử dụng giá trị từ config nếu không được cung cấp
    train_ratio = train_ratio or TRAIN_RATIO
    val_ratio = val_ratio or VAL_RATIO
    test_ratio = test_ratio or TEST_RATIO
    
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
    parser = argparse.ArgumentParser(
        description="Chia dataset thành train/val/test"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Đường dẫn thư mục chứa ảnh"
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Đường dẫn thư mục chứa labels"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Đường dẫn thư mục output"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        help=f"Tỉ lệ train (default: {TRAIN_RATIO})"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        help=f"Tỉ lệ val (default: {VAL_RATIO})"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        help=f"Tỉ lệ test (default: {TEST_RATIO})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Không ghi đè thư mục output nếu đã tồn tại"
    )
    
    args = parser.parse_args()
    
    split_dataset(
        args.images,
        args.labels,
        args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        overwrite=not args.no_overwrite,
        seed=args.seed
    )
