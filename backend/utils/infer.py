import os
import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MODEL_PATH = os.environ.get("MODEL_PATH", "model/best_yolo11.pt")
DEVICE = os.environ.get("DEVICE", "cpu")
CONF_THRES = float(os.environ.get("CONF_THRES", "0.50"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))
IOU_THRES = float(os.environ.get("IOU_THRES", "0.30"))

VN_CLASS_NAMES = {
    0: "Cam_Dau_Xe",
    1: "Cam_Di_Nguoc_Chieu",
    2: "Cam_Dung_Va_Dau_Xe",
    3: "Cam_O_To",
    4: "Cam_Re_Trai",
    5: "Cam_XeTai_XeKhach",
    6: "Cam_Xe_Tai",
    7: "Chi_Huong_Di_Phai",
    8: "Cho_Quay_Xe",
    9: "Den_Giao_Thong",
    10: "Duong_Co_Go_Giam_Toc",
    11: "Duong_Cong_Ben_Phai",
    12: "Duong_Nguoi_Di_Bo_Cat_Ngang",
    13: "Duong_Nguoi_Di_Bo_Sang_Ngang",
    14: "Giao_Nhau_Duong_Uu_Tien",
    15: "Gioi_Han_Toc_Do_25",
    16: "Tre_Em",
    17: "Vong_Xuyen",
    18: "Xe_Bus",
}


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
        label = f"{it['cls_name']} – {it['conf']:.2f}"
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

    npimg = np.array(img)
    results = model.predict(source=npimg, conf=use_conf, iou=use_iou, imgsz=use_imgsz, device=DEVICE, verbose=False)

    detections_raw, predictions_raw = [], []
    if results:
        r = results[0]
        names = r.names
        boxes = r.boxes
        for i in range(len(boxes)):
            b = boxes[i]
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            base_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            cls_name = VN_CLASS_NAMES.get(cls_id, "Biển báo")
            x1 = max(0.0, min(float(x1), float(W)))
            y1 = max(0.0, min(float(y1), float(H)))
            x2 = max(0.0, min(float(x2), float(W)))
            y2 = max(0.0, min(float(y2), float(H)))
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            detections_raw.append({"cls_id": cls_id, "cls_name": cls_name, "conf": conf, "bbox": [x1, y1, x2, y2]})
            predictions_raw.append({"label": cls_name, "score": conf, "box": {"x": x1, "y": y1, "w": w, "h": h}, "nbox": {"x": x1 / W if W else 0.0, "y": y1 / H if H else 0.0, "w": w / W if W else 0.0, "h": h / H if H else 0.0}})

    fallback_used = False
    if len(predictions_raw) == 0 and not q_conf:
        try:
            use_conf = 0.01
            use_imgsz = max(use_imgsz, 1280)
            use_iou = 0.50
            results = model.predict(source=npimg, conf=use_conf, iou=use_iou, imgsz=use_imgsz, device=DEVICE, verbose=False)
            detections_raw, predictions_raw = [], []
            if results:
                r = results[0]
                names = r.names
                boxes = r.boxes
                for i in range(len(boxes)):
                    b = boxes[i]
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0].item())
                    cls_id = int(b.cls[0].item())
                    base_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    cls_name = VN_CLASS_NAMES.get(cls_id, "Biển báo")
                    x1 = max(0.0, min(float(x1), float(W)))
                    y1 = max(0.0, min(float(y1), float(H)))
                    x2 = max(0.0, min(float(x2), float(W)))
                    y2 = max(0.0, min(float(y2), float(H)))
                    w = max(0.0, float(x2 - x1))
                    h = max(0.0, float(y2 - y1))
                    detections_raw.append({"cls_id": cls_id, "cls_name": cls_name, "conf": conf, "bbox": [x1, y1, x2, y2]})
                    predictions_raw.append({"label": cls_name, "score": conf, "box": {"x": x1, "y": y1, "w": w, "h": h}, "nbox": {"x": x1 / W if W else 0.0, "y": y1 / H if H else 0.0, "w": w / W if W else 0.0, "h": h / H if H else 0.0}})
            fallback_used = True
        except Exception:
            pass

    # Filter detections (area, aspect ratio, bounds)
    def _valid_box(x1, y1, x2, y2):
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            return False
        area = w * h
        if area <= 500:
            return False
        ratio1 = w / h if h else 0
        ratio2 = h / w if w else 0
        if ratio1 >= 4 or ratio2 >= 4:
            return False
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            return False
        return True

    detections_filtered = []
    for it in detections_raw:
        x1, y1, x2, y2 = it["bbox"]
        if _valid_box(x1, y1, x2, y2):
            detections_filtered.append(it)

    # Build simple objects schema
    objects_simple = [
        {"label": it["cls_name"], "confidence": float(it["conf"]), "bbox": [float(it["bbox"][0]), float(it["bbox"][1]), float(it["bbox"][2]), float(it["bbox"][3])]} 
        for it in detections_filtered
    ]

    img_draw = img.copy()
    _annotate(img_draw, detections_filtered, color=(0, 255, 0) if not fallback_used else (255, 165, 0))

    buf = io.BytesIO()
    img_draw.save(buf, format="PNG")
    buf.seek(0)
    annotated_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    total = len(detections_filtered)
    avg_conf = float(sum(it["conf"] for it in detections_filtered) / total) if total > 0 else 0.0

    return {
        "W": W,
        "H": H,
        # filtered outputs
        "detections": detections_filtered,
        "predictions": predictions_raw,  # keep for backward compat if needed
        "detections_filtered": detections_filtered,
        "objects_simple": objects_simple,
        "annotated_b64": annotated_b64,
        "avg_conf": avg_conf,
        "used_conf": use_conf,
        "used_imgsz": use_imgsz,
        "used_iou": use_iou,
        "fallback_used": fallback_used,
    }
