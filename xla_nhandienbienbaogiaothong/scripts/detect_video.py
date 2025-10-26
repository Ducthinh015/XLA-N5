"""
Script phát hiện biển báo trên video
"""
import os
import cv2
from ultralytics import YOLO
from pathlib import Path
from config import (
    get_model_path,
    CONFIDENCE_THRESHOLD,
    OUTPUT_DIR,
    print_config
)

class VideoDetector:
    def __init__(self, model_path=None):
        """Khởi tạo detector với model YOLO"""
        if model_path is None:
            model_path = get_model_path("trained")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")
        
        self.model = YOLO(model_path)
        print(f"✅ Đã tải model: {model_path}")
    
    def detect_video(self, video_path, output_path=None, save_annotated=True):
        """
        Phát hiện biển báo trên video
        
        Args:
            video_path: Đường dẫn video input
            output_path: Đường dẫn video output (nếu None sẽ tự tạo)
            save_annotated: Có lưu video đã được annotate không
        
        Returns:
            dict: Thông tin các phát hiện
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy video tại {video_path}")
        
        print(f"\n🎥 Đang xử lý video: {os.path.basename(video_path)}")
        
        # Tạo output path nếu chưa có
        if output_path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            video_name = Path(video_path).stem
            output_path = os.path.join(OUTPUT_DIR, f"{video_name}_detected.mp4")
        
        # Detect với YOLO
        # results là list của từng frame
        results = self.model.predict(
            source=video_path,
            conf=CONFIDENCE_THRESHOLD,
            save=save_annotated,
            project=OUTPUT_DIR,
            name="video_detections",
            verbose=True
        )
        
        # Lưu video đã xử lý
        if save_annotated:
            # YOLO tự động lưu video trong runs/detect/video_detections/
            print(f"✅ Video đã được lưu tại: {output_path}")
        
        # Thống kê
        total_detections = 0
        class_counts = {}
        
        for frame_result in results:
            if frame_result.boxes is not None:
                for box in frame_result.boxes:
                    total_detections += 1
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        # In thống kê
        print("\n" + "="*50)
        print("📊 THỐNG KÊ PHÁT HIỆN")
        print("="*50)
        print(f"Tổng số biển báo phát hiện được: {total_detections}")
        print(f"\nChi tiết theo loại:")
        for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  - {cls_name}: {count}")
        print("="*50)
        
        return {
            "total_detections": total_detections,
            "class_counts": class_counts,
            "output_path": output_path,
            "frames": len(results)
        }
    
    def detect_webcam(self):
        """Phát hiện real-time từ webcam"""
        print("\n📹 Bắt đầu phát hiện từ webcam...")
        print("Nhấn 'q' để thoát")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Không thể mở webcam")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect
                results = self.model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False
                )
                
                # Vẽ kết quả
                annotated_frame = results[0].plot()
                cv2.imshow("Nhận diện biển báo", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Đã dừng webcam")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phát hiện biển báo giao thông trên video"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Đường dẫn video file"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Sử dụng webcam real-time"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Đường dẫn model (.pt file)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Ngưỡng confidence (default: 0.3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Đường dẫn video output"
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
    
    # Khởi tạo detector
    detector = VideoDetector(args.model)
    
    if args.webcam:
        # Phát hiện webcam
        detector.detect_webcam()
    elif args.video:
        # Phát hiện video file
        detector.detect_video(args.video, args.output)
    else:
        print("❌ Vui lòng chọn --video hoặc --webcam")
        parser.print_help()

if __name__ == "__main__":
    main()

