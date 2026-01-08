from ultralytics import YOLO
import cv2
import numpy as np
import os
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.four_point_transform import four_point_transform

# ========= CONFIG =========
MODEL_PATH = "../../model/bestDetect.pt"
VIDEO_PATH = "C:/Users/Admin.ADMIN-PC/Downloads/WIN_20260108_09_51_48_Pro.mp4"
OUT_DIR = "split"
IMG_SIZE = 640
CONF_THRES = 0.5
KP_CONF_THRES = 0.9
DEVICE = "cpu"
# ==========================

os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Không mở được video"

frame_id = 0
save_id = 0


def demo(args):
    frame_id = 0
    save_id = 0
    os.makedirs(args.out_dir, exist_ok=True)
    model = YOLO(args.model_path)

    cap = cv2.VideoCapture(args.video_path)
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

            if np.any(kpts_conf[obj_id] < KP_CONF_THRES):
                continue

            pts = pts.astype(np.float32)
            tl, tr, br, bl = pts

            crop = four_point_transform(
                frame,
                np.array([tl, tr, br, bl], dtype=np.float32)
            )

            if crop is None:
                continue

            save_name = f"frame{frame_id:06d}_obj{obj_id:02d}.jpg"
            cv2.imwrite(os.path.join(args.out_dir, save_name), crop)
            save_id += 1
            colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
            for i, (x, y) in enumerate([tl, tr, br, bl]):
                cv2.circle(frame, (int(x), int(y)), 6, colors[i], -1)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Done. Saved {save_id} crops to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and crop objects from video using YOLO model with keypoints.")

    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the YOLO model")
    parser.add_argument("--video_path", type=str, default=VIDEO_PATH, help="Path to the input video")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR, help="Directory to save cropped images")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Image size for inference")
    parser.add_argument("--conf_thres", type=float, default=CONF_THRES, help="Confidence threshold for detection")
    parser.add_argument("--kp_conf_thres", type=float, default=KP_CONF_THRES, help="Keypoint confidence threshold")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to run inference on")

    args = parser.parse_args()

    demo(args)