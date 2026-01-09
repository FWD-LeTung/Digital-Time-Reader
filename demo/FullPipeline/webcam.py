import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from ultralytics import YOLO
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from src.utils.four_point_transform import four_point_transform

def get_four_digits(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, _ = img.shape[:2]
    raw_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > (h_img * 0.2): # Filter nhiễu
            raw_boxes.append([x, y, w, h])

    raw_boxes.sort(key=lambda b: b[0]) # Sắp xếp từ trái sang phải
    digit_crops = []
    for x, y, w, h in raw_boxes:
        if (w / h) > 0.6: # Box quá béo -> tách đôi
            mid = w // 2
            digit_crops.append(img[y:y+h, x:x+mid])
            digit_crops.append(img[y:y+h, x+mid:x+w])
        else:
            digit_crops.append(img[y:y+h, x:x+w])
    
    return digit_crops[:4] if len(digit_crops) == 4 else None

YOLO_MODEL_PATH = "../../model/bestDetect.pt"
OCR_MODEL_PATH = "../../model/weights/ocr_cnn_4cls.pth"
VIDEO_PATH = "C:/Users/Admin.ADMIN-PC/Downloads/WIN_20260108_09_45_18_Pro.mp4"
OUTPUT_VIDEO_PATH = "output.mp4"  # Đường dẫn file video xuất ra
DEVICE =  "cpu"
CONF_THRES = 0.15
KP_CONF_THRES = 0.9

class OCRCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14x14
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

yolo_model = YOLO(YOLO_MODEL_PATH)
ocr_model = OCRCNN(num_classes=4).to(DEVICE)
# Load weights (Lưu ý: map_location giúp load trên CPU nếu không có GPU)
ocr_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=DEVICE))
ocr_model.eval()



def preprocess_digit(digit_img):
    """Tiền xử lý ảnh chữ số để đưa vào mô hình CNN"""
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    # Chuẩn hóa về [0, 1] và định dạng Tensor (B, C, H, W)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor.to(DEVICE)

def main():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()

    # Lấy thông tin video để tạo VideoWriter
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    # Danh sách class cho phép: 0, 3, 8, 9
    TARGET_CLASSES = [0, 9, 3, 8]

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Xử lý 1: Nhận diện bằng YOLO
        results = yolo_model.predict(source=frame, conf=CONF_THRES, device=DEVICE, verbose=False)
        r = results[0]

        if r.keypoints is not None and r.boxes is not None:
            kpts_xy = r.keypoints.xy.cpu().numpy()
            kpts_conf = r.keypoints.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int) # Lấy danh sách ID class

            for obj_id, pts in enumerate(kpts_xy):
                # --- BỔ SUNG: Kiểm tra class ID ---
                if cls_ids[obj_id] not in TARGET_CLASSES:
                    continue

                if np.any(kpts_conf[obj_id] < KP_CONF_THRES): 
                    continue

                # Cắt và căn chỉnh ảnh vùng đồng hồ (dùng 4 điểm keypoints)
                clock_crop = four_point_transform(frame, pts.astype(np.float32))
                if clock_crop is None: continue

                # Xử lý 2: Cắt làm 4 phần chữ số
                digits = get_four_digits(clock_crop)
                
                if digits and len(digits) == 4:
                    predictions = []
                    # Xử lý 3: Dự đoán từng chữ số
                    for d_img in digits:
                        input_tensor = preprocess_digit(d_img)
                        with torch.no_grad():
                            output = ocr_model(input_tensor)
                            pred = torch.argmax(output, dim=1).item()
                            predictions.append(str(pred))

                    # Logic Output: Ghép thành HH:MM
                    time_str = f"{TARGET_CLASSES[int(predictions[0])]}{TARGET_CLASSES[int(predictions[1])]}:{TARGET_CLASSES[int(predictions[2])]}{TARGET_CLASSES[int(predictions[3])]}"
                    
                    # --- BỔ SUNG: Vẽ 4 điểm Keypoints ---
                    for pt in pts[:4]: # Chỉ lấy tối đa 4 điểm đầu tiên
                        curr_x, curr_y = int(pt[0]), int(pt[1])
                        if curr_x > 0 and curr_y > 0:
                            cv2.circle(frame, (curr_x, curr_y), 5, (0, 255, 0), -1)

                    txt_x, txt_y = int(pts[0][0]), int(pts[0][1])
                    cv2.putText(frame, time_str, (txt_x, txt_y - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Digital Time Reader", frame)
        out.write(frame)  # Ghi frame ra video output
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    out.release()  # Giải phóng VideoWriter
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()