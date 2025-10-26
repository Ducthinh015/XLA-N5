# Hướng dẫn sử dụng Scripts

Tất cả các script đã được refactor để sử dụng file `config.py` và environment variables thay vì hardcode paths.

## 📋 Cấu hình

Các tham số mặc định được định nghĩa trong `config.py`. Bạn có thể override bằng cách:

1. **Sử dụng environment variables** (khuyến nghị):
```bash
export MODEL_PATH="runs/train/my_exp/weights/best.pt"
export CONFIDENCE_THRESHOLD=0.5
export EPOCHS=200
```

2. **Sử dụng command line arguments** cho từng script

## 🚀 Training

```bash
# Training với cấu hình mặc định từ config.py
python scripts/train.py

# Training với custom parameters
python scripts/train.py --epochs 200 --batch 32 --device 0 --name my_exp1

# Xem cấu hình hiện tại
python scripts/train.py --config
```

## 🖼️ Phát hiện ảnh

```bash
# Phát hiện với cấu hình mặc định
python scripts/detect_image.py --image path/to/image.jpg

# Phát hiện với model custom
python scripts/detect_image.py --image path/to/image.jpg --model path/to/best.pt --conf 0.5

# Không hiển thị ảnh, chỉ lưu file
python scripts/detect_image.py --image path/to/image.jpg --no-show

# Xem cấu hình
python scripts/detect_image.py --config
```

## 🎥 Phát hiện video

```bash
# Phát hiện từ video file
python scripts/detect_video.py --video path/to/video.mp4

# Phát hiện từ webcam (real-time)
python scripts/detect_video.py --webcam

# Phát hiện với model custom
python scripts/detect_video.py --video path/to/video.mp4 --model path/to/best.pt --conf 0.5

# Xem cấu hình
python scripts/detect_video.py --config
```

## 📊 Chia dataset

```bash
# Chia dataset với cấu hình mặc định (70/20/10)
python scripts/split_dataset.py \
    --images /path/to/images \
    --labels /path/to/labels \
    --output /path/to/output

# Chia dataset với tỉ lệ custom
python scripts/split_dataset.py \
    --images /path/to/images \
    --labels /path/to/labels \
    --output /path/to/output \
    --train-ratio 0.8 \
    --val-ratio 0.15 \
    --test-ratio 0.05
```

## ⚙️ Xem cấu hình

Bất kỳ script nào đều có thể in ra cấu hình:

```bash
python scripts/train.py --config
python scripts/detect_image.py --config
python scripts/detect_video.py --config
```

## 📝 Môi trường (Environment Variables)

Tạo file `.env` (hoặc export) với các biến sau:

```bash
# Model
MODEL_PATH=yolo11n.pt
TRAINED_MODEL_PATH=runs/train/traffic_signs_exp1/weights/best.pt

# Confidence
CONFIDENCE_THRESHOLD=0.3

# Training
EPOCHS=100
BATCH_SIZE=16
IMAGE_SIZE=640
DEVICE=0

# Dataset
DATASET_PATH=datasets/split_dataset
OUTPUT_DIR=output

# Split ratios
TRAIN_RATIO=0.7
VAL_RATIO=0.2
TEST_RATIO=0.1
```

## 💡 Tips

1. **Sử dụng environment variables** cho paths tuyệt đối khác nhau giữa các máy
2. **Xem cấu hình trước** bằng `--config` để đảm bảo paths đúng
3. **Nên dùng relative paths** nếu có thể, sẽ dễ chia sẻ và deploy hơn

