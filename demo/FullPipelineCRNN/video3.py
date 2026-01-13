import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
import sys
from collections import deque

# Thêm đường dẫn để import four_point_transform
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.four_point_transform import four_point_transform
from model.Reader.crnn.ocrcrnn import OCRCRNNgray

# ================= CONFIG =================
YOLO_MODEL_PATH = "../../model/bestDetect131.pt"
OCR_MODEL_PATH = "../../model/Reader/crnn/crnn_synthetic_gray.pth"
VIDEO_PATH = "D:/Downloads/WIN_20260113_14_37_18_Pro.mp4"
OUTPUT_VIDEO_PATH = "D:/Downloads/output_crnn.mp4"
DEVICE = "cuda"
CONF_THRES = 0.3
KP_CONF_THRES = 0.7

CHARS = "0123456789"
BLANK_IDX = len(CHARS)  # Index 10

# Temporal filtering config
TEMPORAL_WINDOW = 5  # Số frame để làm mượt
CONFIDENCE_THRESHOLD = 0.6  # Ngưỡng confidence để chấp nhận kết quả
# ==========================================

# Khởi tạo mô hình
yolo_model = YOLO(YOLO_MODEL_PATH)
ocr_model = OCRCRNNgray(NUM_CLASSES=len(CHARS) + 1).to(DEVICE)
ocr_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=DEVICE))
ocr_model.eval()


class TemporalFilter:
    """Bộ lọc thời gian để ổn định kết quả OCR"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.last_valid_time = None
    
    def update(self, time_str, confidence):
        """Cập nhật và trả về kết quả đã được làm mượt"""
        # Chỉ thêm vào history nếu time_str hợp lệ (không None)
        if time_str is not None and confidence > CONFIDENCE_THRESHOLD:
            self.history.append(time_str)
            self.last_valid_time = time_str
        
        # Voting mechanism: chọn giá trị xuất hiện nhiều nhất
        if len(self.history) >= 3:
            from collections import Counter
            counter = Counter(self.history)
            most_common = counter.most_common(1)[0][0]
            return most_common
        
        # Nếu chưa đủ dữ liệu, trả về giá trị hợp lệ cuối cùng
        return self.last_valid_time if self.last_valid_time else "00:00"


def decode_predictions(logits):
    """
    Giải mã CTC predictions với xử lý robust
    
    Pipeline:
    1. CTC decode (greedy)
    2. Lọc chỉ giữ chữ số
    3. Kiểm tra đúng 4 chữ số (bỏ qua nếu không)
    4. Ràng buộc HH:MM hợp lệ
    """
    # Bước 1: CTC Greedy Decode
    preds = logits.argmax(2).squeeze(0).cpu().numpy()
    
    raw_text = ""
    prev = -1
    for c in preds:
        # Bỏ qua ký tự trùng lặp liên tiếp và blank
        if c != prev and c != BLANK_IDX:
            raw_text += CHARS[c]
        prev = c
    
    # Bước 2: Lọc chỉ giữ chữ số
    digits_only = ''.join(filter(str.isdigit, raw_text))
    
    # Bước 3: Kiểm tra đúng 4 chữ số
    normalized = normalize_to_4_digits(digits_only)
    
    # Bỏ qua nếu không phải 4 chữ số
    if normalized is None:
        return None, 0.0
    
    # Bước 4: Ràng buộc HH:MM
    valid_time, confidence = validate_and_fix_time(normalized)
    
    return valid_time, confidence


def normalize_to_4_digits(digits):
    """
    Chuẩn hóa chuỗi chữ số về đúng 4 ký tự
    
    Cases:
    - Không phải 4 chữ số: return None (bỏ qua)
    - Đúng 4: giữ nguyên
    """
    if len(digits) != 4:
        return None
    
    return digits


def score_time_validity(time_str):
    """
    Đánh giá độ hợp lý của chuỗi thời gian (0-100)
    Điểm cao hơn = thời gian hợp lý hơn
    """
    if len(time_str) != 4:
        return 0
    
    score = 50  # baseline
    
    try:
        hh = int(time_str[:2])
        mm = int(time_str[2:])
        
        # Giờ hợp lệ (0-23)
        if 0 <= hh <= 23:
            score += 30
        elif hh == 24:
            score += 10  # Có thể là 24:00
        
        # Phút hợp lệ (0-59)
        if 0 <= mm <= 59:
            score += 30
        
        # Bonus cho thời gian phổ biến
        if 6 <= hh <= 23:  # Giờ thức
            score += 5
        if mm % 5 == 0:  # Phút chia hết cho 5
            score += 5
            
    except ValueError:
        score = 0
    
    return score


def validate_and_fix_time(time_str):
    """
    Ràng buộc và sửa thời gian về định dạng HH:MM hợp lệ
    
    Returns:
        (formatted_time, confidence)
    """
    if len(time_str) != 4:
        return "00:00", 0.0
    
    try:
        hh = int(time_str[:2])
        mm = int(time_str[2:])
        
        confidence = 1.0
        
        # Fix giờ
        if hh > 23:
            if hh < 30:
                hh = hh % 24  # 24->0, 25->1, etc.
                confidence *= 0.7
            else:
                hh = int(str(hh)[0])  # Lấy chữ số đầu: 45->4
                confidence *= 0.5
        
        # Fix phút
        if mm > 59:
            if mm < 70:
                mm = mm % 60  # 60->0, 65->5
                confidence *= 0.7
            else:
                mm = int(str(mm)[0]) * 10  # 75->50, 89->80->20
                if mm > 59:
                    mm = mm % 60
                confidence *= 0.5
        
        formatted = f"{hh:02d}:{mm:02d}"
        return formatted, confidence
        
    except (ValueError, IndexError):
        return "00:00", 0.0


def preprocess_for_crnn(img):
    """Resize về 50x20 và chuẩn hóa"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (50, 20))
    
    img_tensor = torch.from_numpy(resized).float() / 255.0
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

    # Khởi tạo temporal filter cho mỗi đồng hồ
    temporal_filters = {}

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = yolo_model.predict(source=frame, conf=CONF_THRES, device=DEVICE, verbose=False)
        r = results[0]

        if r.keypoints is not None:
            kpts_xy = r.keypoints.xy.cpu().numpy()
            kpts_conf = r.keypoints.conf.cpu().numpy()

            for obj_id, pts in enumerate(kpts_xy):
                if np.any(kpts_conf[obj_id] < KP_CONF_THRES):
                    continue

                # Khởi tạo filter nếu chưa có
                if obj_id not in temporal_filters:
                    temporal_filters[obj_id] = TemporalFilter(TEMPORAL_WINDOW)

                # Cắt vùng đồng hồ
                clock_crop = four_point_transform(frame, pts.astype(np.float32))
                if clock_crop is None:
                    continue

                # Nhận diện thời gian
                input_tensor = preprocess_for_crnn(clock_crop)
                with torch.no_grad():
                    logits = ocr_model(input_tensor)
                    raw_time, confidence = decode_predictions(logits)

                # Bỏ qua nếu không phải 4 chữ số
                if raw_time is None:
                    continue

                # Áp dụng temporal filter
                stable_time = temporal_filters[obj_id].update(raw_time, confidence)
                
                # Vẽ kết quả
                for pt in pts[:4]:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

                # Màu sắc dựa trên confidence
                color = (0, 255, 255) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
                
                cv2.putText(frame, stable_time, 
                           (int(pts[0][0]), int(pts[0][1]) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Hiển thị confidence (debug)
                # cv2.putText(frame, f"{confidence:.2f}", 
                #            (int(pts[0][0]), int(pts[0][1]) - 50),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)
        cv2.imshow("Digital Time Reader (Enhanced)", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Đã xử lý {frame_count} frames")
    print(f"Video output: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()