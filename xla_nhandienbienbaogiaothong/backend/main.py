"""
Backend FastAPI cho ứng dụng nhận diện biển báo giao thông
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from typing import List
import os
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import tempfile
import uuid
from pathlib import Path

# Import config từ scripts
import sys
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from config import get_model_path, CONFIDENCE_THRESHOLD
except ImportError:
    # Fallback nếu không tìm thấy config
    def get_model_path(model_type="trained"):
        if model_type == "base":
            return "yolo11n.pt"
        else:
            return os.getenv("TRAINED_MODEL_PATH", "runs/train/traffic_signs_exp1/weights/best.pt")
    
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))

# Khởi tạo app
app = FastAPI(
    title="XLA - Nhận diện biển báo giao thông",
    description="API để phát hiện biển báo giao thông từ ảnh và video",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (lazy loading)
_model = None

def get_model():
    """Load model YOLO (singleton)"""
    global _model
    if _model is None:
        try:
            # Thử load trained model trước
            model_path = get_model_path("trained")
            print(f"Loading model: {model_path}")
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}, using base model")
                model_path = get_model_path("base")
            _model = YOLO(model_path)
            print(f"Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Error loading trained model: {e}")
            print("Fallback to base model...")
            model_path = get_model_path("base")
            _model = YOLO(model_path)
            print(f"Base model loaded successfully: {model_path}")
    return _model


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "Backend FastAPI hoạt động!",
        "status": "running",
        "model": "YOLOv11"
    }


@app.get("/health")
def health_check():
    """Health check with model status"""
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "confidence_threshold": CONFIDENCE_THRESHOLD
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e)
        }


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Upload ảnh và phát hiện biển báo
    
    Returns:
        {
            "detections": [{"class": str, "confidence": float, "bbox": {...}}],
            "annotated_image": str (base64)
        }
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File phải là ảnh (jpg, png, jpeg)")
        
        print(f"Received image: {file.filename}, type: {file.content_type}")
        
        # Tạo temp file để lưu ảnh
        temp_dir = tempfile.gettempdir()
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        temp_file = os.path.join(temp_dir, f"{uuid.uuid4()}.{file_ext}")
        
        print(f"Saving to: {temp_file}")
        
        # Save uploaded file
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"File saved, size: {len(content)} bytes")
        
        # Load model và detect
        print("Loading model...")
        model = get_model()
        
        print("Running prediction...")
        # Predict
        results = model.predict(source=temp_file, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        print(f"Prediction done, got {len(results)} result(s)")
        
        # Parse results
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "class": cls_name,
                    "confidence": confidence,
                    "bbox": {
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    }
                })
        
        print(f"Found {len(detections)} detections")
        
        # Tạo annotated image (base64)
        print("Creating annotated image...")
        annotated_img = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print("Returning response")
        return {
            "detections": detections,
            "annotated_image": annotated_base64,
            "total_detections": len(detections)
        }
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"ERROR in detect_image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temp file
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Cleaned up: {temp_file}")


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Upload video và phát hiện biển báo, trả về video đã xử lý
    
    Returns:
        Video file (blob) với các biển báo đã được vẽ khung
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File phải là video (mp4, avi, mov)")
    
    temp_dir = tempfile.gettempdir()
    temp_input = os.path.join(temp_dir, f"{uuid.uuid4()}.{file.filename.split('.')[-1]}")
    temp_output = os.path.join(temp_dir, f"{uuid.uuid4()}_detected.mp4")
    
    try:
        # Save uploaded file
        with open(temp_input, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load model và detect
        model = get_model()
        
        # Predict trên video
        results = model.predict(
            source=temp_input,
            conf=CONFIDENCE_THRESHOLD,
            save=True,
            project=temp_dir,
            name=f"temp_{uuid.uuid4()}",
            verbose=False
        )
        
        # Tìm file output (YOLO tự động tạo)
        output_files = []
        for result in results:
            if hasattr(result, 'save_dir'):
                output_dir = result.save_dir
                # Tìm file video trong output dir
                if os.path.exists(output_dir):
                    for f in os.listdir(output_dir):
                        if f.endswith('.mp4'):
                            output_files.append(os.path.join(output_dir, f))
        
        if not output_files:
            # Nếu không tìm thấy, tạo video manually từ results
            raise HTTPException(status_code=500, detail="Không thể tạo video output")
        
        # Lấy file output đầu tiên
        detected_video_path = output_files[0]
        
        # Return video file
        return FileResponse(
            detected_video_path,
            media_type="video/mp4",
            filename="detected_video.mp4"
        )
    
    except Exception as e:
        print(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý video: {str(e)}")
    
    finally:
        # Clean up (sẽ clean sau khi response sent)
        pass


@app.get("/model/info")
def get_model_info():
    """Thông tin về model đang sử dụng"""
    try:
        model = get_model()
        return {
            "model_name": "YOLOv11",
            "num_classes": len(model.names) if hasattr(model, 'names') else 0,
            "class_names": list(model.names.values()) if hasattr(model, 'names') else [],
            "confidence_threshold": CONFIDENCE_THRESHOLD
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting Backend API")
    print("="*60)
    print("📍 URL: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
