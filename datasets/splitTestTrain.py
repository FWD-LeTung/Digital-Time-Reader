import re
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ===== CONFIG =====
src_root = Path(r"D:/model2-20260110T081631Z-3-001/model2/dataset/images")
train_dir = Path(r"D:/model2-20260110T081631Z-3-001/model2/dataset/train")
test_dir = Path(r"D:/model2-20260110T081631Z-3-001/model2/dataset/test")

train_ratio = 0.9
random.seed(42)

train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

pattern = re.compile(
    r"(?P<stem>.+)_frame(?P<frame>\d+)_obj(?P<obj>\d+)\.jpg",
    re.IGNORECASE
)

# ===== STEP 1: group theo video_stem -> frame_id =====
videos = defaultdict(lambda: defaultdict(list))

for img in src_root.iterdir():
    if img.suffix.lower() != ".jpg":
        continue

    m = pattern.match(img.name)
    if not m:
        print("SKIP:", img.name)
        continue

    video_stem = m.group("stem")
    frame_id = int(m.group("frame"))

    videos[video_stem][frame_id].append(img)

# ===== STEP 2: split cho từng video =====
for video_stem, frames in videos.items():
    frame_ids = list(frames.keys())
    random.shuffle(frame_ids)

    n_train = int(len(frame_ids) * train_ratio)
    train_frames = set(frame_ids[:n_train])

    for frame_id, files in frames.items():
        dst_root = train_dir if frame_id in train_frames else test_dir
        for f in files:
            shutil.copy2(f, dst_root / f.name)

    print(
        f"{video_stem}: "
        f"{len(train_frames)} train frames / "
        f"{len(frame_ids) - len(train_frames)} test frames"
    )
