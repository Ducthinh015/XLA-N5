"""
File cấu hình chung cho tất cả script
Sử dụng environment variables hoặc giá trị mặc định
"""
import os
from pathlib import Path

# ==================== MODEL CONFIG ====================
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n.pt")
TRAINED_MODEL_PATH = os.getenv(
    "TRAINED_MODEL_PATH", 
    os.path.join("runs", "train", "traffic_signs_exp1", "weights", "best.pt")
)

# ==================== DETECTION CONFIG ====================
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))

# ==================== TRAINING CONFIG ====================
EPOCHS = int(os.getenv("EPOCHS", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
DEVICE = os.getenv("DEVICE", "0")  # 0=GPU, cpu=CPU

# ==================== DATASET CONFIG ====================
DATASET_PATH = os.getenv("DATASET_PATH", "datasets/split_dataset")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# Test paths (có thể override bằng env vars)
INPUT_IMAGE_PATH = os.getenv("INPUT_IMAGE_PATH", "")
INPUT_VIDEO_PATH = os.getenv("INPUT_VIDEO_PATH", "")

# ==================== SPLIT DATASET CONFIG ====================
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.7"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.2"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.1"))

def get_model_path(model_type="trained"):
    """
    Lấy đường dẫn model
    
    Args:
        model_type: "base" (yolo11n.pt) hoặc "trained" (best.pt)
    
    Returns:
        str: Đường dẫn model
    """
    if model_type == "base":
        return MODEL_PATH
    else:
        return TRAINED_MODEL_PATH

def get_dataset_yaml():
    """Lấy đường dẫn file data.yaml"""
    return os.path.join("configs", "data.yaml")

def print_config():
    """In ra cấu hình hiện tại"""
    print("=" * 60)
    print("📋 CẤU HÌNH HIỆN TẠI")
    print("=" * 60)
    print(f"Model (base): {MODEL_PATH}")
    print(f"Model (trained): {TRAINED_MODEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Training epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    # Test: in ra cấu hình
    print_config()

