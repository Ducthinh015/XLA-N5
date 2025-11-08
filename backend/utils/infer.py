import os
import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MODEL_PATH = os.environ.get("MODEL_PATH", "runs/train/traffic_signs_exp12/weights/best.pt")
DEVICE = os.environ.get("DEVICE", "cpu")
CONF_THRES = float(os.environ.get("CONF_THRES", "0.10"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))
IOU_THRES = float(os.environ.get("IOU_THRES", "0.70"))

_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model

def _load_font():
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except Exception:
        return ImageFont.load_default()

def _annotate(img, items, color=(0, 255, 0)):
    draw = ImageDraw.Draw(img)
    font = _load_font()
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        label = f"{it['cls_name']} {it['conf']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        try:
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        except Exception:
            tw, th = (len(label) * 8, 18)
        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - th - 3)), label, font=font, fill=(0, 0, 0))
    return img

def predict_image(file_stream, q_conf=None, q_imgsz=None):
    model = get_model()
    img = Image.open(file_stream).convert("RGB")
    W, H = img.size
    use_conf = float(q_conf) if q_conf else CONF_THRES
    use_imgsz = int(q_imgsz) if q_imgsz else IMG_SIZE
    use_iou = IOU_THRES
    if not q_conf and min(W, H) <= 256:
        use_conf = min(use_conf, 0.05)
    if not q_imgsz and min(W, H) <= 256:
        use_imgsz = max(use_imgsz, 960)

    npimg = np.array(img)
    results = model.predict(source=npimg, conf=use_conf, iou=use_iou, imgsz=use_imgsz, device=DEVICE, verbose=False)

    detections, predictions = [], []
    if results:
        r = results[0]
        names = r.names
        boxes = r.boxes
        for i in range(len(boxes)):
            b = boxes[i]
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            x1 = max(0.0, min(float(x1), float(W)))
            y1 = max(0.0, min(float(y1), float(H)))
            x2 = max(0.0, min(float(x2), float(W)))
            y2 = max(0.0, min(float(y2), float(H)))
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            detections.append({"cls_id": cls_id, "cls_name": cls_name, "conf": conf, "bbox": [x1, y1, x2, y2]})
            predictions.append({"label": cls_name, "score": conf, "box": {"x": x1, "y": y1, "w": w, "h": h}, "nbox": {"x": x1 / W if W else 0.0, "y": y1 / H if H else 0.0, "w": w / W if W else 0.0, "h": h / H if H else 0.0}})

    fallback_used = False
    if len(predictions) == 0 and not q_conf:
        try:
            use_conf = 0.01
            use_imgsz = max(use_imgsz, 1280)
            use_iou = 0.45
            results = model.predict(source=npimg, conf=use_conf, iou=use_iou, imgsz=use_imgsz, device=DEVICE, verbose=False)
            detections, predictions = [], []
            if results:
                r = results[0]
                names = r.names
                boxes = r.boxes
                for i in range(len(boxes)):
                    b = boxes[i]
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0].item())
                    cls_id = int(b.cls[0].item())
                    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    x1 = max(0.0, min(float(x1), float(W)))
                    y1 = max(0.0, min(float(y1), float(H)))
                    x2 = max(0.0, min(float(x2), float(W)))
                    y2 = max(0.0, min(float(y2), float(H)))
                    w = max(0.0, float(x2 - x1))
                    h = max(0.0, float(y2 - y1))
                    detections.append({"cls_id": cls_id, "cls_name": cls_name, "conf": conf, "bbox": [x1, y1, x2, y2]})
                    predictions.append({"label": cls_name, "score": conf, "box": {"x": x1, "y": y1, "w": w, "h": h}, "nbox": {"x": x1 / W if W else 0.0, "y": y1 / H if H else 0.0, "w": w / W if W else 0.0, "h": h / H if H else 0.0}})
            fallback_used = True
        except Exception:
            pass

    img_draw = img.copy()
    _annotate(img_draw, detections, color=(0, 255, 0) if not fallback_used else (255, 165, 0))

    buf = io.BytesIO()
    img_draw.save(buf, format="PNG")
    buf.seek(0)
    annotated_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    total = len(predictions)
    avg_conf = float(sum(p["score"] for p in predictions) / total) if total > 0 else 0.0

    return {
        "W": W,
        "H": H,
        "detections": detections,
        "predictions": predictions,
        "annotated_b64": annotated_b64,
        "avg_conf": avg_conf,
        "used_conf": use_conf,
        "used_imgsz": use_imgsz,
        "used_iou": use_iou,
        "fallback_used": fallback_used,
    }
