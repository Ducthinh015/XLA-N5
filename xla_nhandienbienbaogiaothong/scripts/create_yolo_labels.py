import os
import cv2

image_dir = r"N:\BaiTap\XuLyAnhVaThiGiac\XLA-N5\xla_nhandienbienbaogiaothong\dataset\archive\images"
# 5 lớp biển báo
CLASSES = {
    0: "Biển báo cấm",
    1: "Biển báo nguy hiểm và cảnh báo",
    2: "Biển báo hiệu lệnh",
    3: "Biển báo chỉ báo khu vực",
    4: "Biển chỉ dẫn"
}

def create_yolo_label(file_name, boxes, img_width, img_height, label_dir):
    """
    Tạo file YOLO label (.txt) cho 1 ảnh
    boxes: list [(cls_id, x_min, y_min, x_max, y_max), ...]
    """
    label_path = os.path.join(label_dir, os.path.splitext(file_name)[0] + ".txt")
    os.makedirs(label_dir, exist_ok=True)

    with open(label_path, "w") as f:
        for cls_id, x_min, y_min, x_max, y_max in boxes:
            # chuẩn hóa về 0–1
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # giới hạn trong [0,1]
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            width = min(max(width, 0), 1)
            height = min(max(height, 0), 1)

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"✅ Tạo nhãn YOLO: {label_path}")


# =============================
# Ví dụ sử dụng
# =============================
if __name__ == "__main__":
    # Thư mục dataset
    image_dir = r"C:\XLA-N5-main\XLA-N5-main\xla_nhandienbienbaogiaothong\dataset\archive\archive\images"
    label_dir = r"C:\XLA-N5-main\XLA-N5-main\xla_nhandienbienbaogiaothong\dataset\archive\archive\labels"  # nơi lưu file txt YOLO

    # Ví dụ 1 ảnh test
    test_image = "0001.jpg"
    img_path = os.path.join(image_dir, test_image)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Ví dụ bounding box: (cls_id, x_min, y_min, x_max, y_max)
    boxes = [
        (0, 100, 120, 250, 300),  # biển báo cấm
        (4, 300, 200, 450, 400)   # biển chỉ dẫn
    ]

    create_yolo_label(test_image, boxes, w, h, label_dir)


