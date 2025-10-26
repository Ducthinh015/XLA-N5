"""
Script training model YOLO cho biển báo giao thông
Sử dụng cấu hình từ config.py hoặc environment variables
"""
from ultralytics import YOLO
from config import (
    MODEL_PATH,
    get_dataset_yaml,
    EPOCHS,
    BATCH_SIZE,
    IMAGE_SIZE,
    DEVICE,
    print_config
)
import argparse

def main(model_path=None, data_yaml=None, epochs=None, batch=None, 
         imgsz=None, device=None, name="traffic_signs_exp1"):
    """
    Train model YOLO
    
    Args:
        model_path: Đường dẫn model base (nếu None dùng từ config)
        data_yaml: Đường dẫn file data.yaml
        epochs: Số epochs
        batch: Batch size
        imgsz: Image size
        device: Device ("0" = GPU, "cpu" = CPU)
        name: Tên experiment
    """
    # Sử dụng giá trị từ config nếu không được cung cấp
    model_path = model_path or MODEL_PATH
    data_yaml = data_yaml or get_dataset_yaml()
    epochs = epochs or EPOCHS
    batch = batch or BATCH_SIZE
    imgsz = imgsz or IMAGE_SIZE
    device = device or DEVICE
    
    print("\n" + "="*60)
    print("🚀 BẮT ĐẦU TRAINING MODEL YOLO")
    print("="*60)
    print(f"Model base: {model_path}")
    print(f"Data config: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device}")
    print(f"Experiment name: {name}")
    print("="*60 + "\n")
    
    # Load model gốc YOLOv11 (nano version, nhẹ nhất)
    print(f"📦 Đang tải model: {model_path}")
    model = YOLO(model_path)
    
    # Train
    print("\n🏋️ Bắt đầu training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/train",
        name=name,
        verbose=True
    )

    # Kết quả
    print("\n" + "="*60)
    print("✅ TRAINING HOÀN THÀNH!")
    print("="*60)
    print(f"📁 Model tốt nhất được lưu tại:")
    print(f"   {results.save_dir}")
    print("\n✨ Sử dụng lệnh sau để test:")
    print(f"   python scripts/detect_image.py --image <path> --model {results.save_dir}/weights/best.pt")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training model YOLO cho biển báo giao thông"
    )
    parser.add_argument("--model", type=str, help="Đường dẫn model base")
    parser.add_argument("--data", type=str, help="Đường dẫn file data.yaml")
    parser.add_argument("--epochs", type=int, help="Số epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Device (0=GPU, cpu=CPU)")
    parser.add_argument("--name", type=str, default="traffic_signs_exp1", 
                       help="Tên experiment")
    parser.add_argument("--config", action="store_true", 
                       help="In cấu hình và thoát")
    
    args = parser.parse_args()
    
    if args.config:
        print_config()
    else:
        main(
            model_path=args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            name=args.name
        )
