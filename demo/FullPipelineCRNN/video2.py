import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
import sys

# Thêm đường dẫn để import four_point_transform
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.four_point_transform import four_point_transform
from model.Reader.crnn.ocrcrnn import OCRCRNNgray

# ================= CONFIG (Đồng bộ với train_CRNN.py) =================
YOLO_MODEL_PATH = "../../model/bestDetect.pt"
OCR_MODEL_PATH = "../../model/Reader/crnn/crnn_synthetic_gray.pth"
VIDEO_PATH = "C:/Users/letung373/Pictures/Camera Roll/WIN_20260108_09_45_18_Pro.mp4"
OUTPUT_VIDEO_PATH = "output_crnn.mp4"
DEVICE = "cpu"
CONF_THRES = 0.3
KP_CONF_THRES = 0.7

CHARS = "0123456789"
BLANK_IDX = len(CHARS) # Index 10
# ======================================================================

# Khởi tạo mô hình
yolo_model = YOLO(YOLO_MODEL_PATH)
ocr_model = OCRCRNNgray(NUM_CLASSES=len(CHARS) + 1).to(DEVICE)
ocr_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=DEVICE))
ocr_model.eval()

def decode_predictions(logits):
    """Giải mã Greedy dựa trên logic của train_CRNN.py"""
    preds = logits.argmax(2).squeeze(0).cpu().numpy()
    
    text = ""
    prev = -1
    for c in preds:
        # 1. Bỏ qua ký tự trùng lặp liên tiếp (CTC logic)
        # 2. Bỏ qua ký tự Blank (index 10)
        if c != prev and c != BLANK_IDX:
            text += CHARS[c]
        prev = c
        
    # Định dạng kết quả thành HH:MM nếu đủ 4 chữ số
    if len(text) == 4:
        return f"{text[:2]}:{text[2:]}"
    return text


def preprocess_for_crnn(img):
    """Resize về 50x20 và chuẩn hóa (theo train_CRNN.py)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # train_CRNN.py dùng IMG_W=50, IMG_H=20
    resized = cv2.resize(gray, (50, 20))
    
    # 3. Chuyển thành Tensor & Chuẩn hóa
    # resized lúc này có shape (20, 50)
    img_tensor = torch.from_numpy(resized).float() / 255.0
    
    # 4. Thêm chiều Batch và Channel: (H, W) -> (1, 1, H, W)
    # CRNN yêu cầu input 4D: [Batch Size, Channels, Height, Width]
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) 
    
    return img_tensor.to(DEVICE)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Không thể mở video!")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model.predict(source=frame, conf=CONF_THRES, device=DEVICE, verbose=False)
        r = results[0]

        if r.keypoints is not None:
            kpts_xy = r.keypoints.xy.cpu().numpy()
            kpts_conf = r.keypoints.conf.cpu().numpy()

            for obj_id, pts in enumerate(kpts_xy):
                if np.any(kpts_conf[obj_id] < KP_CONF_THRES): continue

                # Cắt vùng đồng hồ bằng perspective transform
                clock_crop = four_point_transform(frame, pts.astype(np.float32))
                if clock_crop is None: continue

                # Nhận diện chuỗi thời gian
                input_tensor = preprocess_for_crnn(clock_crop)
                with torch.no_grad():
                    logits = ocr_model(input_tensor)
                    time_str = decode_predictions(logits)

                # Vẽ 4 điểm keypoints và hiển thị text
                for pt in pts[:4]:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

                cv2.putText(frame, time_str, (int(pts[0][0]), int(pts[0][1]) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Digital Time Reader (CRNN)", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
