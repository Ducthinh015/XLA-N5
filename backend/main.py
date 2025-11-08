from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

MODEL_PATH = os.environ.get("MODEL_PATH", "runs/train/traffic_signs_exp1/weights/best.pt")
DEVICE = os.environ.get("DEVICE", "cpu")
CONF_THRES = float(os.environ.get("CONF_THRES", "0.5"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))

model = YOLO(MODEL_PATH)

app = Flask(__name__)
CORS(app)

@app.get("/")
def root():
    return jsonify({"status": "ok", "model": os.path.basename(MODEL_PATH)})

# Map API prefix for frontend compatibility
app.add_url_rule("/api", view_func=root, methods=["GET"])
app.add_url_rule("/api/health", view_func=root, methods=["GET"])

@app.post("/infer/image")
def infer_image():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "message": "missing file", "total": 0, "avg_conf": 0.0, "avg_conf_percent": 0, "predictions": [], "detections": []}), 400
        q_conf = request.args.get("conf")
        q_imgsz = request.args.get("imgsz")
        use_conf = float(q_conf) if q_conf else CONF_THRES
        use_imgsz = int(q_imgsz) if q_imgsz else IMG_SIZE

        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        W, H = img.size
        npimg = np.array(img)
        results = model.predict(source=npimg, conf=use_conf, imgsz=use_imgsz, device=DEVICE, verbose=False)
        detections = []
        predictions = []
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
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
                # clamp values
                x1 = max(0.0, min(float(x1), float(W)))
                y1 = max(0.0, min(float(y1), float(H)))
                x2 = max(0.0, min(float(x2), float(W)))
                y2 = max(0.0, min(float(y2), float(H)))
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                detections.append({
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                predictions.append({
                    "label": cls_name,
                    "score": conf,
                    "box": {"x": x1, "y": y1, "w": w, "h": h},
                    "nbox": {"x": x1 / W if W else 0.0, "y": y1 / H if H else 0.0, "w": w / W if W else 0.0, "h": h / H if H else 0.0}
                })
                # draw annotation
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                label = f"{cls_name} {conf:.2f}"
                try:
                    tw, th = draw.textbbox((0, 0), label, font=font)[2:]
                except Exception:
                    tw, th = (len(label) * 8, 18)
                draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=(0, 255, 0))
                draw.text((x1 + 3, max(0, y1 - th - 3)), label, font=font, fill=(0, 0, 0))
        total = len(predictions)
        avg_conf = float(sum(p["score"] for p in predictions) / total) if total > 0 else 0.0
        # alias-friendly payload for various frontends
        boxes_xywh = [{"x": p["box"]["x"], "y": p["box"]["y"], "w": p["box"]["w"], "h": p["box"]["h"], "label": p["label"], "score": p["score"]} for p in predictions]
        boxes_xyxy = [{"x1": d["bbox"][0], "y1": d["bbox"][1], "x2": d["bbox"][2], "y2": d["bbox"][3], "label": d["cls_name"], "score": d["conf"]} for d in detections]
        # encode annotated image
        buf = io.BytesIO()
        img_draw.save(buf, format="PNG")
        buf.seek(0)
        import base64
        annotated_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

        resp = {
            "success": True,
            "message": "ok",
            "image_width": W,
            "image_height": H,
            "total": total,
            "avg_conf": avg_conf,
            "avg_conf_percent": int(round(avg_conf * 100)) if total > 0 else 0,
            "predictions": predictions,
            "detections": detections,
            "annotated_image": annotated_b64,
            # aliases for FE variants
            "count": total,
            "results": predictions,
            "objects": predictions,
            "boxes_xywh": boxes_xywh,
            "boxes_xyxy": boxes_xyxy
        }
        return jsonify(resp)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "total": 0, "avg_conf": 0.0, "avg_conf_percent": 0, "predictions": [], "detections": []}), 500

# Map API prefix for frontend compatibility
app.add_url_rule("/api/infer/image", view_func=infer_image, methods=["POST"])
app.add_url_rule("/api/detect/image", view_func=infer_image, methods=["POST"])

@app.post("/infer/image-annotated")
def infer_image_annotated():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    npimg = np.array(img)
    results = model.predict(source=npimg, conf=CONF_THRES, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    draw = ImageDraw.Draw(img)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
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
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            label = f"{cls_name} {conf:.2f}"
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=(0, 255, 0))
            draw.text((x1 + 3, max(0, y1 - th - 3)), label, font=font, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

# Map API prefix for frontend compatibility
app.add_url_rule("/api/infer/image-annotated", view_func=infer_image_annotated, methods=["POST"])
app.add_url_rule("/api/detect/image-annotated", view_func=infer_image_annotated, methods=["POST"])

@app.post("/infer/label")
def infer_label():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    npimg = np.array(img)
    results = model.predict(source=npimg, conf=CONF_THRES, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    detections = []
    best = None
    if results:
        r = results[0]
        names = r.names
        boxes = r.boxes
        for i in range(len(boxes)):
            b = boxes[i]
            conf = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            item = {"cls_id": cls_id, "cls_name": cls_name, "conf": conf}
            detections.append(item)
            if best is None or conf > best["conf"]:
                best = item
    return jsonify({"best": best, "all": detections})

# Map API prefix for frontend compatibility
app.add_url_rule("/api/infer/label", view_func=infer_label, methods=["POST"])
app.add_url_rule("/api/detect/label", view_func=infer_label, methods=["POST"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)