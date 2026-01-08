from ultralytics import YOLO
import cv2
import os

# ====== CONFIG ======
MODEL_PATH = "model/bestDetect.pt"
VIDEO_PATH = "C:/Users/letung373/Pictures/Camera Roll/WIN_20260108_09_51_48_Pro.mp4"    # VIDEO input
IMG_SIZE = 640
CONF_THRES = 0.15
IOU_THRES = 0.7
SAVE_DIR = "F:/aecdatasets/mlcom2/runs/infer"
OUTPUT_VIDEO = os.path.join(SAVE_DIR, f"{VIDEO_PATH}output.mp4")
# ====================

os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Không mở được video"

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference từng frame
    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device="cpu",
        verbose=False
    )

    result = results[0]

    # Vẽ bounding box
    if result.keypoints is not None:
        kps = result.keypoints.xy.cpu().numpy()  
        # shape: (num_obj, num_kp, 2)

        for obj_id, pts in enumerate(kps):
            # pts shape: (num_kp, 2)
            xs = pts[:, 0]
            ys = pts[:, 1]

            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())



            # vẽ từng keypoint (debug)
            for (x, y) in pts:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    writer.write(frame)

cap.release()
writer.release()

print(f"Saved output video to: {OUTPUT_VIDEO}")
