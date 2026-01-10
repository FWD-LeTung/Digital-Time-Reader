import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_DIR = r"D:/model2-20260110T081631Z-3-001/model2/dataset/ocr/test/images"
OUTPUT_DIR = r"D:/model2-20260110T081631Z-3-001/model2/datasets/test"

IMG_OUT_DIR = os.path.join(OUTPUT_DIR, "images")
REJECT_DIR = os.path.join(OUTPUT_DIR, "rejected", "unknown_prefix")
LABEL_FILE = os.path.join(OUTPUT_DIR, "labels.txt")
REJECT_LOG = os.path.join(OUTPUT_DIR, "reject_reasons.txt")

TARGET_HEIGHT = 20
MAX_WIDTH = 50

os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

# =========================
# VIDEO → TIME LABEL MAP
# (4 digit, KHÔNG dấu :)
# =========================
VIDEO_TIME_MAP = {
    "WIN_20260110_09_35_52_Pro": "0636",
    "WIN_20260110_09_37_20_Pro": "0737",
    "WIN_20260110_09_39_39_Pro": "0838",
    "WIN_20260110_09_42_01_Pro": "1649",
    "WIN_20260110_09_43_43_Pro": "1749",
    "WIN_20260110_09_45_03_Pro": "1845",
    "WIN_20260110_09_46_37_Pro": "1949",
    "WIN_20260110_09_47_58_Pro": "2056",
    "WIN_20260110_09_50_35_Pro": "2257",
    "WIN_20260110_09_52_21_Pro": "2358",
    # thêm ở đây
}

# =========================
# HELPERS
# =========================
def resize_for_crnn(img):
    h, w = img.shape[:2]
    scale = TARGET_HEIGHT / h
    new_w = int(w * scale)

    if new_w > MAX_WIDTH:
        new_w = MAX_WIDTH

    return cv2.resize(img, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)


def extract_video_stem(filename: str):
    """
    Lấy phần video name trước '_frame'
    """
    if "_frame" not in filename:
        return None
    return filename.split("_frame")[0]


def is_valid_time_label(label: str):
    """
    label: '2358'
    """
    if len(label) != 4 or not label.isdigit():
        return False

    hh = int(label[:2])
    mm = int(label[2:])

    return 0 <= hh <= 23 and 0 <= mm <= 59


# =========================
# MAIN
# =========================
labels = []
reject_logs = []

img_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

for fname in tqdm(img_files, desc="Labeling images"):
    img_path = os.path.join(INPUT_DIR, fname)

    video_stem = extract_video_stem(fname)
    if video_stem is None:
        shutil.copy(img_path, REJECT_DIR)
        reject_logs.append(f"{fname}\tinvalid_filename_format")
        continue

    if video_stem not in VIDEO_TIME_MAP:
        shutil.copy(img_path, REJECT_DIR)
        reject_logs.append(f"{fname}\tunknown_video_prefix")
        continue

    label = VIDEO_TIME_MAP[video_stem]

    if not is_valid_time_label(label):
        shutil.copy(img_path, REJECT_DIR)
        reject_logs.append(f"{fname}\tinvalid_time_label:{label}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        shutil.copy(img_path, REJECT_DIR)
        reject_logs.append(f"{fname}\timage_read_failed")
        continue

    out_img = resize_for_crnn(img)

    out_name = fname
    cv2.imwrite(os.path.join(IMG_OUT_DIR, out_name), out_img)

    labels.append(f"{out_name}\t{label}")

# =========================
# WRITE FILES
# =========================
with open(LABEL_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(labels))

with open(REJECT_LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(reject_logs))

print("=" * 50)
print(f"Done.")
print(f"Accepted: {len(labels)}")
print(f"Rejected: {len(reject_logs)}")
print("=" * 50)
