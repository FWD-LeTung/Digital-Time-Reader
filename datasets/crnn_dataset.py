import os
import cv2
import re
from tqdm import tqdm
from paddleocr import PaddleOCR

# =========================
# CONFIG
# =========================
INPUT_DIR = r"D:/model2-20260110T081631Z-3-001/model2/dataset/test"
OUTPUT_DIR = r"D:/model2-20260110T081631Z-3-001/model2/dataset/ocr/test"
IMG_OUT_DIR = os.path.join(OUTPUT_DIR, "images")
REJECT_DIR = os.path.join(OUTPUT_DIR, "rejected")
LABEL_FILE = os.path.join(OUTPUT_DIR, "labels.txt")

TARGET_HEIGHT = 24
MAX_WIDTH = 62

CONF_THRESHOLD = 0.5

ENABLE_CONF_FILTER = False
ENABLE_REGEX_FILTER = True
ENABLE_LENGTH_FILTER = False
ENABLE_COLON_FILTER = False

TIME_REGEX = r"^(\d{2}:\d{2}|\d{4})$"

os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

# =========================
# OCR INIT (PIPELINE API)
# =========================
ocr = PaddleOCR(
    lang="en"
)

# =========================
# UTILS
# =========================
def resize_for_crnn(img, target_h=24, max_w=62):
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    new_w = min(new_w, max_w)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

def reject(img, name, reason):
    out_path = os.path.join(REJECT_DIR, f"{reason}_{name}")
    cv2.imwrite(out_path, img)

# =========================
# MAIN LOOP
# =========================
label_lines = []
img_id = 0

image_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

for fname in tqdm(image_files):
    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # đảm bảo 3 channel
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # OCR
    try:
        result = ocr.predict(img)
    except Exception:
        reject(img, fname, "ocr_error")
        continue

    if not result or len(result) == 0:
        reject(img, fname, "empty")
        continue

    pred = result[0]

    texts = pred.get("rec_texts", [])
    scores = pred.get("rec_scores", [])

    if len(texts) == 0:
        reject(img, fname, "no_text")
        continue

    text = texts[0].strip()
    score = float(scores[0])

    # =========================
    # FILTERS
    # =========================
    if ENABLE_CONF_FILTER and score < CONF_THRESHOLD:
        reject(img, fname, "lowconf")
        continue

    if ENABLE_LENGTH_FILTER and len(text) < 4:
        reject(img, fname, "short")
        continue

    if ENABLE_COLON_FILTER and ":" not in text:
        reject(img, fname, "nocolon")
        continue

    if ENABLE_REGEX_FILTER and not re.match(TIME_REGEX, text):
        reject(img, fname, "regex")
        continue

    # =========================
    # ACCEPT
    # =========================
    img_resized = resize_for_crnn(img)

    out_name = f"{img_id:06d}.jpg"
    out_path = os.path.join(IMG_OUT_DIR, out_name)
    cv2.imwrite(out_path, img_resized)

    label_lines.append(f"images/{out_name}\t{text}")
    img_id += 1

# =========================
# WRITE LABEL FILE
# =========================
with open(LABEL_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(label_lines))

print("=" * 60)
print("DATASET BUILD DONE")
print(f"Accepted images : {img_id}")
print(f"Rejected images : {len(image_files) - img_id}")
print(f"Labels file     : {LABEL_FILE}")
print("=" * 60)
