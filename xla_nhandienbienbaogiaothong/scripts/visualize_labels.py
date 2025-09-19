import os
import cv2
import random

# -------- CONFIG --------
# Thư mục ảnh và nhãn
image_dir = "split_dataset_v2/train/images"
label_dir = "split_dataset_v2/train/labels"

# Danh sách class
CLASSES = ["Biển báo cấm, Biển báo nguy hiểm và cảnh báo, Biển báo hiệu lệnh, Biển báo chỉ báo khu vực,Biển chỉ dẫn"]

# Số ảnh muốn hiển thị ngẫu nhiên
NUM_SAMPLES = 5
# ------------------------

def visualize_image(img_path, label_path):
    # Đọc ảnh
    image = cv2.imread(img_path)
    h, w, _ = image.shape

    if not os.path.exists(label_path):
        print(f"⚠️ Không tìm thấy nhãn cho {img_path}")
        return image

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls_id, x_center, y_center, bw, bh = map(float, line.strip().split())
            cls_id = int(cls_id)

            # Chuyển tọa độ YOLO (0→1) sang pixel để vẽ
            x_center_px, y_center_px, bw_px, bh_px = (
                x_center * w,
                y_center * h,
                bw * w,
                bh * h
            )
            x1 = int(x_center_px - bw_px / 2)
            y1 = int(y_center_px - bh_px / 2)
            x2 = int(x_center_px + bw_px / 2)
            y2 = int(y_center_px + bh_px / 2)

            # Vẽ bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
            cv2.putText(image, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # In ra tọa độ chuẩn YOLO
            print(f"{os.path.basename(img_path)} | class={cls_id}, "
                  f"x={x_center:.3f}, y={y_center:.3f}, w={bw:.3f}, h={bh:.3f}")

    return image

def main():
    # Lấy danh sách ảnh
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    samples = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))

    for img_name in samples:
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        vis_img = visualize_image(img_path, label_path)
        cv2.imshow("Labeled Image", vis_img)
