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


def input_iterator(video_path=None, image_dir=None):
    """
    Yield (frame_id, frame)
    """
    if image_dir is not None:
        image_files = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        for idx, img_path in enumerate(image_files):
            img = cv2.imread(img_path)
            if img is None:
                continue
            yield idx + 1, img

    else:
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Không mở được video"

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            yield frame_id, frame

        cap.release()


def demo(args):
    os.makedirs(args.out_dir, exist_ok=True)
    model = YOLO(args.model_path)

    save_id = 0

    for frame_id, frame in input_iterator(
        video_path=args.video_path,
        image_dir=args.image_dir
    ):
        results = model.predict(
            source=frame,
            imgsz=args.img_size,
            conf=args.conf_thres,
            device=args.device,
            verbose=False
        )

        r = results[0]
        if r.keypoints is None:
            continue

        kpts_xy = r.keypoints.xy.cpu().numpy()      # (N, 4, 2)
        kpts_conf = r.keypoints.conf.cpu().numpy()  # (N, 4)

        for obj_id, pts in enumerate(kpts_xy):

            if np.any(kpts_conf[obj_id] < args.kp_conf_thres):
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

            # visualize keypoints
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
            for i, (x, y) in enumerate([tl, tr, br, bl]):
                cv2.circle(frame, (int(x), int(y)), 6, colors[i], -1)

        if args.image_dir is None:
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    print(f"Done. Saved {save_id} crops to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and crop objects from video or image folder using YOLO keypoints"
    )

    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--video_path", type=str, default=VIDEO_PATH)
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to image folder (if set, video will be ignored)")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--conf_thres", type=float, default=CONF_THRES)
    parser.add_argument("--kp_conf_thres", type=float, default=KP_CONF_THRES)
    parser.add_argument("--device", type=str, default=DEVICE)

    args = parser.parse_args()
    demo(args)
