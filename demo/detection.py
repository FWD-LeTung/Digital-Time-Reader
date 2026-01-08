from ultralytics import YOLO
import cv2
import numpy as np
import os

from src.ultis.four_point_transform import four_point_transform



# ========= CONFIG =========
MODEL_PATH = "best.pt"
VIDEO_PATH = "C:/Users/Admin.ADMIN-PC/Downloads/WIN_20260108_09_51_48_Pro.mp4"
OUT_DIR = "split"
IMG_SIZE = 640
CONF_THRES = 0.15
KP_CONF_THRES = 0.9
DEVICE = "cpu"
# ==========================

os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Không mở được video"

frame_id = 0
save_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device=DEVICE,
        verbose=False
    )

    r = results[0]
    if r.keypoints is None:
        continue

    kpts_xy = r.keypoints.xy.cpu().numpy()      # (N, 4, 2)
    kpts_conf = r.keypoints.conf.cpu().numpy()  # (N, 4)

    for obj_id, pts in enumerate(kpts_xy):

        # bỏ object có keypoint confidence thấp
        if np.any(kpts_conf[obj_id] < KP_CONF_THRES):
            continue

        pts = pts.astype(np.float32)

        # ===== THỨ TỰ CHUẨN =====
        tl, tr, br, bl = pts

        crop = four_point_transform(
            frame,
            np.array([tl, tr, br, bl], dtype=np.float32)
        )

        if crop is None:
            continue

        save_name = f"frame{frame_id:06d}_obj{obj_id:02d}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, save_name), crop)
        save_id += 1

        # ---- debug (tuỳ chọn) ----
        colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
        for i, (x, y) in enumerate([tl, tr, br, bl]):
            cv2.circle(frame, (int(x), int(y)), 6, colors[i], -1)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Done. Saved {save_id} crops to {OUT_DIR}")
