"""
Script phát hiện biển báo trên ảnh
Sử dụng cấu hình từ config.py hoặc environment variables
"""
from ultralytics import YOLO
import os
import cv2
import argparse
from config import (
    get_model_path, 
    CONFIDENCE_THRESHOLD,
    INPUT_IMAGE_PATH,
    print_config
)

def detect_image(image_path, model_path=None, conf=None, show=True, save=True):
    """
    Phát hiện biển báo trên ảnh
    
    Args:
        image_path: Đường dẫn ảnh
        model_path: Đường dẫn model (nếu None sẽ dùng từ config)
        conf: Ngưỡng confidence
        show: Có hiển thị ảnh không
        save: Có lưu ảnh không
    
    Returns:
        results: Kết quả từ YOLO
    """
    if model_path is None:
        model_path = get_model_path("trained")
    
    if conf is None:
        conf = CONFIDENCE_THRESHOLD
    
    # Kiểm tra file
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh tại {image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")
    
    print(f"\n📸 Đang xử lý ảnh: {os.path.basename(image_path)}")
    print(f"🤖 Model: {os.path.basename(model_path)}")
    print(f"🎯 Confidence threshold: {conf}")
    
    # Load model và predict
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, save=save)
    
    # Hiển thị hình có khung
    annotated = results[0].plot()
    
    # Hiển thị trong cửa sổ
    if show:
        cv2.imshow("Kết quả detect", annotated)
        print("Nhấn phím bất kỳ để đóng cửa sổ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # In thông tin box
    print("\n" + "="*60)
    print("📊 KẾT QUẢ PHÁT HIỆN")
    print("="*60)
    
    for idx, r in enumerate(results):
        if r.boxes is not None:
            print(f"\nFrame {idx}:")
            print(f"  Số lượng biển báo: {len(r.boxes)}")
            
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                print(f"\n  Biển báo #{i+1}:")
                print(f"    Loại: {cls_name} (ID: {cls_id})")
                print(f"    Độ tin cậy: {confidence:.2%}")
                print(f"    Vị trí: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})")
                print(f"    Kích thước: {int(x2-x1)} x {int(y2-y1)} pixels")
        else:
            print(f"\nFrame {idx}: Không phát hiện biển báo")
    
    print("\n" + "="*60)
    print("✅ Hoàn thành!")
    print("="*60)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Phát hiện biển báo giao thông trên ảnh"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_PATH or None,
        help="Đường dẫn ảnh"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Đường dẫn model (.pt file)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        help=f"Ngưỡng confidence (default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Không hiển thị ảnh kết quả"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Không lưu ảnh kết quả"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="In cấu hình và thoát"
    )
    
    args = parser.parse_args()
    
    # In cấu hình
    if args.config:
        print_config()
        return
    
    # Kiểm tra image path
    if not args.image:
        print("❌ Vui lòng cung cấp đường dẫn ảnh: --image <path>")
        parser.print_help()
        return
    
    # Detect
    detect_image(
        args.image,
        model_path=args.model,
        conf=args.conf,
        show=not args.no_show,
        save=not args.no_save
    )

if __name__ == "__main__":
    main()
