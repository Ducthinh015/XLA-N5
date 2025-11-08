from ultralytics import YOLO
import os
import cv2

MODEL_PATH = r"D:\TL_XLA\XLA-N5\xla_nhandienbienbaogiaothong\traffic_signs_exp13\weights\best.pt"
IMAGE_PATH = "2874.jpg"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model tại {MODEL_PATH}")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Không tìm thấy ảnh test tại {IMAGE_PATH}")

model = YOLO(MODEL_PATH)
results = model.predict(source=IMAGE_PATH, conf=0.3, save=True)

# Hiển thị hình có khung
annotated = results[0].plot()
cv2.imshow("Kết quả detect", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In thêm thông tin box
for r in results:
    print("Boxes:", r.boxes)
