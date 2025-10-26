import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- CHỈNH ĐƯỜNG DẪN Ở ĐÂY ---
image_dir = r"N:\BaiTap\XuLyAnhVaThiGiac\XLA-N5\xla_nhandienbienbaogiaothong\dataset\archive\images"
label_dir = r"N:\BaiTap\XuLyAnhVaThiGiac\XLA-N5\xla_nhandienbienbaogiaothong\dataset\archive\labels"
output_dir = r"N:\BaiTap\XuLyAnhVaThiGiac\XLA-N5\xla_nhandienbienbaogiaothong\dataset\archive\output"
os.makedirs(output_dir, exist_ok=True)
# -------------------------------

# Class names (nếu thiếu id > len(CLASSES) sẽ in "Biển báo <id>")
CLASSES = [
    "Biển báo cấm",
    "Biển báo nguy hiểm và cảnh báo",
    "Biển báo hiệu lệnh",
    "Biển báo chỉ báo khu vực",
    "Biển chỉ dẫn"
]

# Font unicode (Windows). Thay nếu máy bạn có font khác.
FONT_PATH = "C:/Windows/Fonts/arial.ttf"
try:
    FONT = ImageFont.truetype(FONT_PATH, 18)
except:
    FONT = ImageFont.load_default()

# --- các hàm chuyển đổi format sang bbox pixel ---
def yolo_norm_to_box(coords, W, H):
    cx, cy, rw, rh = coords
    cx *= W; cy *= H; rw *= W; rh *= H
    x1 = int(cx - rw/2); y1 = int(cy - rh/2); x2 = int(cx + rw/2); y2 = int(cy + rh/2)
    return x1,y1,x2,y2

def minmax_norm_to_box(coords, W, H):
    x1 = int(coords[0]*W); y1 = int(coords[1]*H); x2 = int(coords[2]*W); y2 = int(coords[3]*H)
    return x1,y1,x2,y2

def minwh_norm_to_box(coords, W, H):
    x1 = int(coords[0]*W); y1 = int(coords[1]*H); w = coords[2]*W; h = coords[3]*H
    return x1,y1,int(x1+w),int(y1+h)

def yolo_abs_to_box(coords, W, H):
    cx, cy, w_abs, h_abs = coords
    x1 = int(cx - w_abs/2); y1 = int(cy - h_abs/2); x2 = int(cx + w_abs/2); y2 = int(cy + h_abs/2)
    return x1,y1,x2,y2

def corners_abs_to_box(coords, W, H):
    x1 = int(coords[0]); y1 = int(coords[1]); x2 = int(coords[2]); y2 = int(coords[3])
    return x1,y1,x2,y2

# clamp & validity
def clamp_box(box, W, H):
    x1,y1,x2,y2 = box
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2))
    y2 = max(0, min(H-1, y2))
    return x1,y1,x2,y2

def box_valid(box, W, H, min_area_px=50):
    x1,y1,x2,y2 = box
    if x2 <= x1 or y2 <= y1: return False
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H: return False
    area = (x2-x1)*(y2-y1)
    if area < min_area_px: return False
    if area > 0.9 * W * H: return False
    return True

# Tập các "chiến lược" convert để thử
STRATEGIES = {
    "yolo_norm": yolo_norm_to_box,
    "minmax_norm": minmax_norm_to_box,
    "minwh_norm": minwh_norm_to_box,
    "yolo_abs": yolo_abs_to_box,
    "corners_abs": corners_abs_to_box
}

def detect_best_strategy(coords_list, W, H):
    # coords_list: list of lists (4 floats) - chưa có cls
    scores = {}
    for name, func in STRATEGIES.items():
        valid = 0
        total = 0
        for coords in coords_list:
            try:
                box = func(coords, W, H)
            except Exception:
                continue
            box_cl = clamp_box(box, W, H)
            if box_valid(box_cl, W, H):
                valid += 1
            total += 1
        scores[name] = (valid, total)
    # pick strategy with max valid (if tie choose yolo_norm preference)
    best = max(scores.items(), key=lambda x: (x[1][0], x[0] == "yolo_norm"))
    return best[0], scores

def draw_text_pil(cv_img, text, pos):
    # pos: (x,y), color green
    img_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=FONT, fill=(0,255,0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def visualize_one(image_path, label_path, out_path, show=True):
    img = cv2.imread(image_path)
    if img is None:
        print("Không đọc được ảnh:", image_path); return
    H, W = img.shape[:2]

    if not os.path.exists(label_path):
        print("Không có nhãn:", label_path)
        return

    lines = [l.strip() for l in open(label_path, "r", encoding="utf-8").read().splitlines() if l.strip()]
    parsed = []
    parsed_full = []  # keep cls + coords
    for i,l in enumerate(lines):
        parts = l.split()
        if len(parts) < 5:
            # skip invalid
            continue
        cls = parts[0]
        try:
            coords = list(map(float, parts[1:5]))
        except:
            continue
        parsed.append(coords)
        parsed_full.append((int(float(cls)), coords))

    if not parsed:
        print(f"{os.path.basename(image_path)} -> không có coords hợp lệ trong {os.path.basename(label_path)}")
        return

    best_strategy, scores = detect_best_strategy(parsed, W, H)
    print(f"\n{os.path.basename(image_path)}: strategy scores: {scores}, selected: {best_strategy}")

    func = STRATEGIES[best_strategy]
    img_draw = img.copy()
    valid_cnt = 0
    skipped_cnt = 0
    for cls, coords in parsed_full:
        try:
            box = func(coords, W, H)
        except Exception:
            skipped_cnt += 1
            continue
        box = clamp_box(box, W, H)
        if not box_valid(box, W, H):
            skipped_cnt += 1
            continue
        x1,y1,x2,y2 = box
        cv2.rectangle(img_draw, (x1,y1), (x2,y2), (0,255,0), 2)
        if 0 <= cls < len(CLASSES):
            label_text = CLASSES[cls]
        else:
            label_text = f"Biển báo {cls}"
        # vẽ text bằng PIL để hỗ trợ tiếng Việt
        img_draw = draw_text_pil(img_draw, label_text, (x1, max(y1-22, 0)))
        valid_cnt += 1

    print(f"-> {os.path.basename(image_path)}: valid={valid_cnt}, skipped={skipped_cnt}")
    cv2.imwrite(out_path, img_draw)
    if show:
        cv2.imshow("Labeled Image", img_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    for im in sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]):
        image_path = os.path.join(image_dir, im)
        label_path = os.path.join(label_dir, os.path.splitext(im)[0] + ".txt")
        out_path = os.path.join(output_dir, im)
        visualize_one(image_path, label_path, out_path, show=True)

if __name__ == "__main__":
    main()
