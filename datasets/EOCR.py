import os
import cv2
import re
import shutil
from tqdm import tqdm
import easyocr
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_DIR = r"D:/datasetCRNN_OCR/test/images"
OUTPUT_DIR = r"D:/datasetCRNN_OCR/ocr/test"

IMG_OUT_DIR = os.path.join(OUTPUT_DIR, "images")
REJECT_DIR = os.path.join(OUTPUT_DIR, "rejected")
LABEL_FILE = os.path.join(OUTPUT_DIR, "labels.txt")
REJECT_LOG = os.path.join(OUTPUT_DIR, "rejected_reasons.txt")

TARGET_HEIGHT = 24
MAX_WIDTH = 64

CONF_THRESHOLD = 0.5

ENABLE_CONF_FILTER = True
ENABLE_LENGTH_FILTER = True
ENABLE_TIME_FILTER = True

MIN_DIGITS = 4
MAX_DIGITS = 4

os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

# =========================
# INIT OCR
# =========================
reader = easyocr.Reader(
    ['en'],
    gpu=True
)

# =========================
# HELPERS
# =========================
def preprocess(img):
    """
    Preprocess cho EasyOCR:
    - gray
    - upscale
    - contrast boost
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    img = clahe.apply(img)

    return img


def resize_for_crnn(img):
    h, w = img.shape[:2]
    scale = TARGET_HEIGHT / h
    new_w = int(w * scale)

    if new_w > MAX_WIDTH:
        new_w = MAX_WIDTH

    img = cv2.resize(img, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return img


def extract_digits(ocr_result):
    """
    Gom toàn bộ digit OCR đọc được
    """
    digits = []
    scores = []

    for _, text, conf in ocr_result:
        clean = re.sub(r"\D", "", text)
        if clean:
            digits.append(clean)
            scores.append(conf)

    if not digits:
        return None, 0.0

    final_text = "".join(digits)
    avg_conf = sum(scores) / len(scores)

    return final_text, avg_conf


def valid_time_hhmm(text):
    """
    Kiểm tra HHMM hợp lệ
    """
    if len(text) != 4 or not text.isdigit():
        return False

    hh = int(text[:2])
    mm = int(text[2:])

    if hh > 24:
        return False
    if mm > 59:
        return False

    return True


# =========================
# MAIN
# =========================
labels = []
reject_logs = []

img_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

for fname in tqdm(img_files):
    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path)

    reject_reason = None

    if img is None:
        reject_reason = "read_fail"

    if reject_reason is None:
        proc = preprocess(img)

        result = reader.readtext(
            proc,
            allowlist="0123456789",
            detail=1,
            paragraph=False
        )

        text, conf = extract_digits(result)

        if text is None:
            reject_reason = "no_text"

        # 1️⃣ Length filter
        if reject_reason is None and ENABLE_LENGTH_FILTER:
            if not (MIN_DIGITS <= len(text) <= MAX_DIGITS):
                reject_reason = "length"

        # 2️⃣ Time filter
        if reject_reason is None and ENABLE_TIME_FILTER:
            if not valid_time_hhmm(text):
                reject_reason = "invalid_time"

        # 3️⃣ Confidence filter
        if reject_reason is None and ENABLE_CONF_FILTER:
            if conf < CONF_THRESHOLD:
                reject_reason = "confidence"

    # ===== REJECT =====
    if reject_reason:
        shutil.copy(img_path, os.path.join(REJECT_DIR, fname))
        reject_logs.append(f"{fname}\t{reject_reason}")
        continue

    # ===== ACCEPT =====
    out_img = resize_for_crnn(proc)
    cv2.imwrite(os.path.join(IMG_OUT_DIR, fname), out_img)
    labels.append(f"{fname}\t{text}")

# =========================
# WRITE FILES
# =========================
with open(LABEL_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(labels))

with open(REJECT_LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(reject_logs))

print("========== DONE ==========")
print(f"Accepted : {len(labels)}")
print(f"Rejected : {len(reject_logs)}")
print("==========================")
