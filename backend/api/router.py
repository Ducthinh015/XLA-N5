from flask import Blueprint, jsonify, request, send_file
import io
try:
    # when running as package (python -m backend.main)
    from backend.utils.infer import predict_image, get_model
except Exception:
    try:
        # when running as script (python backend/main.py)
        from utils.infer import predict_image, get_model
    except Exception:
        import os, sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from utils.infer import predict_image, get_model

bp = Blueprint("api", __name__)

@bp.get("/")
def root():
    m = get_model()
    return jsonify({"status": "ok", "model": getattr(m, "ckpt_path", "best.pt").split("/")[-1]})

@bp.post("/infer/image")
def infer_image_route():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "missing file", "total": 0, "avg_conf": 0.0, "avg_conf_percent": 0, "predictions": [], "detections": []}), 400
    q_conf = request.args.get("conf")
    q_imgsz = request.args.get("imgsz")
    out = predict_image(request.files["file"].stream, q_conf=q_conf, q_imgsz=q_imgsz)
    boxes_xywh = [{"x": p["box"]["x"], "y": p["box"]["y"], "w": p["box"]["w"], "h": p["box"]["h"], "label": p["label"], "score": p["score"]} for p in out["predictions"]]
    boxes_xyxy = [{"x1": d["bbox"][0], "y1": d["bbox"][1], "x2": d["bbox"][2], "y2": d["bbox"][3], "label": d["cls_name"], "score": d["conf"]} for d in out["detections"]]
    objects_simple = out.get("objects_simple", [])
    resp = {
        "success": True,
        "message": "ok",
        "image_width": out["W"],
        "image_height": out["H"],
        "total": len(out["predictions"]),
        "avg_conf": out["avg_conf"],
        "avg_conf_percent": int(round(out["avg_conf"] * 100)) if len(out["predictions"]) > 0 else 0,
        "predictions": out["predictions"],
        "detections": out["detections"],
        "annotated_image": out["annotated_b64"],
        "used_conf": out["used_conf"],
        "used_imgsz": out["used_imgsz"],
        "used_iou": out["used_iou"],
        "fallback_used": out["fallback_used"],
        "count": len(out["predictions"]),
        "results": out["predictions"],
        "objects": out["predictions"],
        "boxes_xywh": boxes_xywh,
        "boxes_xyxy": boxes_xyxy,
        "objects_simple": objects_simple
    }
    return jsonify(resp)

@bp.post("/infer/image-annotated")
def infer_image_annotated_route():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    q_conf = request.args.get("conf")
    q_imgsz = request.args.get("imgsz")
    out = predict_image(request.files["file"].stream, q_conf=q_conf, q_imgsz=q_imgsz)
    b = out["annotated_b64"].split(",", 1)[1] if out["annotated_b64"].startswith("data:image") else out["annotated_b64"]
    import base64
    raw = base64.b64decode(b)
    return send_file(io.BytesIO(raw), mimetype="image/png")

@bp.post("/infer/label")
def infer_label_route():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    q_conf = request.args.get("conf")
    q_imgsz = request.args.get("imgsz")
    out = predict_image(request.files["file"].stream, q_conf=q_conf, q_imgsz=q_imgsz)
    best = None
    for p in out["predictions"]:
        item = {"cls_name": p["label"], "conf": p["score"]}
        if best is None or item["conf"] > best["conf"]:
            best = item
    return jsonify({"best": best, "all": [{"cls_name": p["label"], "conf": p["score"]} for p in out["predictions"]]})

# Aliases for frontend compatibility
bp.add_url_rule("/api", view_func=root, methods=["GET"])
bp.add_url_rule("/api/health", view_func=root, methods=["GET"])

bp.add_url_rule("/api/infer/image", view_func=infer_image_route, methods=["POST"])
bp.add_url_rule("/api/detect/image", view_func=infer_image_route, methods=["POST"])

bp.add_url_rule("/api/infer/image-annotated", view_func=infer_image_annotated_route, methods=["POST"])
bp.add_url_rule("/api/detect/image-annotated", view_func=infer_image_annotated_route, methods=["POST"])

bp.add_url_rule("/api/infer/label", view_func=infer_label_route, methods=["POST"])
bp.add_url_rule("/api/detect/label", view_func=infer_label_route, methods=["POST"])

