from ultralytics import YOLO
import cv2
import numpy as np

# ===== CONFIG =====
MODEL_PATH = "model/bestDetect.pt"
IMG_SIZE = 640
CONF_THRES = 0.1
IOU_THRES = 0.7
DEVICE = "cpu"
SHRINK_RATIO = 0.05
# ==================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Không mở được webcam"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    # ===== KEYPOINTS =====
    if result.keypoints is not None:
        kps = result.keypoints.xy.cpu().numpy()  
        # shape: (num_obj, num_kp, 2)

        for obj_id, pts in enumerate(kps):
            # pts shape: (num_kp, 2)
            xs = pts[:, 0]
            ys = pts[:, 1]

            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())

            # ---- shrink bbox ----
            

            # vẽ từng keypoint (debug)
            for (x, y) in pts:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    cv2.imshow("YOLOv8 Keypoints → BBox", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
