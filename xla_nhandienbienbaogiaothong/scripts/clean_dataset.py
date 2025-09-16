import os
import cv2
import hashlib
from PIL import Image

# Thư mục gốc chứa dataset
input_dir = "dataset/archive"
# Thư mục xuất ảnh sạch
output_dir = "clean_dataset"
os.makedirs(output_dir, exist_ok=True)

# Lưu hash để bỏ ảnh trùng
hashes = set()

def get_image_hash(image_path):
    """Tính md5 hash của ảnh (dùng để phát hiện trùng)"""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def is_valid_image(image_path, min_size=(128, 128)):
    """Kiểm tra ảnh có hợp lệ và đủ kích thước không"""
    try:
        img = Image.open(image_path)
        img.verify()  # check corruption
        img = Image.open(image_path).convert("RGB")
        if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
            return False
        return True
    except Exception:
        return False

def is_blurry(image_path, threshold=100.0):
    """Kiểm tra ảnh có bị mờ không (Laplacian variance)"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return lap_var < threshold

# Duyệt toàn bộ folder và subfolder
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        file_path = os.path.join(root, file_name)

        # Kiểm tra ảnh hợp lệ
        if not is_valid_image(file_path):
            print(f"❌ Bỏ ảnh lỗi hoặc quá nhỏ: {file_name}")
            continue

        # Kiểm tra ảnh mờ
        if is_blurry(file_path):
            print(f"❌ Bỏ ảnh mờ: {file_name}")
            continue

        # Kiểm tra trùng lặp
        file_hash = get_image_hash(file_path)
        if file_hash in hashes:
            print(f"❌ Bỏ ảnh trùng: {file_name}")
            continue
        hashes.add(file_hash)

        # Giữ nguyên tên thư mục con khi lưu
        rel_path = os.path.relpath(root, input_dir)
        save_dir = os.path.join(output_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        # Lưu ảnh hợp lệ
        img = cv2.imread(file_path)
        cv2.imwrite(os.path.join(save_dir, file_name), img)

print("✔ Lọc dữ liệu xong! Ảnh sạch nằm ở:", output_dir)
