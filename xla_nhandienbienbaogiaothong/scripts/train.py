from ultralytics import YOLO

def main():
    # Load model gốc YOLOv11 (nano version, nhẹ nhất)
    model = YOLO("yolov11n.pt")  
    
    # Train
    results = model.train(
        data="datasets/data.yaml",   # file data.yaml
        epochs=100,                  # số epoch
        imgsz=640,                   # resize ảnh về 640x640
        batch=16,                    # batch size
        device=0,                     # 0 = GPU, "cpu" = chạy CPU
        project="runs/train",         # nơi lưu kết quả
        name="traffic_signs_exp1",    # tên thí nghiệm
        verbose=True
    )

    #  Xuất model tốt nhất
    print("Training finished. Best model saved at:")
    print(results.save_dir)

if __name__ == "__main__":
    main()
