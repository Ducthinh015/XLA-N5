from ultralytics import YOLO
import os

# 1. Đường dẫn model
MODEL_PATH = os.path.join(r"D:\KI7-2025\XULIHINHANH\XLA_BTN\XLA-N5\xla_nhandienbienbaogiaothong\scripts\traffic_signs_exp13\weights\best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model tại {MODEL_PATH}")

# 2. Load model
model = YOLO(MODEL_PATH)
print("Model YOLO đã load thành công!")

# 3. Test ảnh
IMAGE_PATH = "0133.jpg"   # đổi thành ảnh test của bạn

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Không tìm thấy ảnh test tại {IMAGE_PATH}")

# Thực hiện detect
results = model.predict(source=IMAGE_PATH, save=True, imgsz=640, conf=0.1)

# 4. In kết quả
for r in results:
    print("Kết quả detection:")
    print(r.boxes)   # bounding boxes
    print(r.probs)   # xác suất dự đoán
