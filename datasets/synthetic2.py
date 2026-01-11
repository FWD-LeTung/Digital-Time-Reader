import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- CẤU HÌNH ---
OUTPUT_DIR = "D:/model2-20260110T081631Z-3-001/model2/datasets/test2/images"
LABEL_FILE = "D:/model2-20260110T081631Z-3-001/model2/datasets/test2/labels.txt"
FONT_PATH = "D:/Downloads/SF-Pro.ttf"  
IMG_SIZE = (50, 20)      
NUM_SAMPLES = 20000

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_time_variants():
    """Tạo chuỗi giờ ngẫu nhiên"""
    h = random.randint(0, 23)
    m = random.randint(0, 59)
    image_text = f"{h:02d}:{m:02d}"
    label_text = f"{h:02d}{m:02d}"
    return image_text, label_text

# --- CÁC HÀM BIẾN ĐỔI HÌNH HỌC MỚI ---

def random_rotate(image, max_angle, bg_color):
    """Xoay ảnh ngẫu nhiên một góc nhỏ."""
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # Tạo ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Áp dụng biến đổi affine
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
    return rotated

def random_perspective(image, max_distortion, bg_color):
    """Áp dụng biến đổi phối cảnh để tạo hiệu ứng nghiêng/chéo."""
    h, w = image.shape[:2]
    
    # 4 điểm góc của ảnh gốc
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # 4 điểm đích ngẫu nhiên (di chuyển nhẹ các góc vào trong/ra ngoài)
    # max_distortion là tỷ lệ di chuyển tối đa (ví dụ 0.1 = 10% kích thước ảnh)
    dx = w * max_distortion
    dy = h * max_distortion
    
    pts2 = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],           # Top-left
        [random.uniform(w - dx, w), random.uniform(0, dy)],       # Top-right
        [random.uniform(0, dx), random.uniform(h - dy, h)],       # Bottom-left
        [random.uniform(w - dx, w), random.uniform(h - dy, h)]    # Bottom-right
    ])
    
    # Tính ma trận biến đổi phối cảnh
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # Áp dụng biến đổi
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
    return warped

# -------------------------------------

def create_synthetic_data():
    print(f"Bắt đầu tạo dữ liệu đa dạng (Xoay, Nghiêng, Chéo)...")
    with open(LABEL_FILE, "w", encoding="utf-8") as f_label:
        for i in range(NUM_SAMPLES):
            # 1. Tạo nền và vẽ chữ bằng PIL (như cũ)
            bg_val = random.randint(30, 45)
            img_pil = Image.new('L', IMG_SIZE, color=bg_val)
            draw = ImageDraw.Draw(img_pil)
            
            image_text, label_text = get_time_variants()
            font_size = 20
            try:
                font = ImageFont.truetype(FONT_PATH, font_size)
            except:
                font = ImageFont.load_default()

            text_color = random.randint(220, 255)
            draw.text((IMG_SIZE[0]/2, IMG_SIZE[1]/2), image_text, 
                      fill=text_color, font=font, anchor="mm")
            
            # --- BẮT ĐẦU PHẦN BIẾN ĐỔI HÌNH HỌC ---
            
            # Chuyển sang ảnh OpenCV (numpy array uint8) để xử lý hình học
            img_cv = np.array(img_pil)
            
            # Áp dụng ngẫu nhiên các biến đổi
            # Có thể chỉ xoay, chỉ nghiêng, hoặc cả hai
            if random.random() > 0.9: 
                img_cv = random_rotate(img_cv, max_angle=5, bg_color=bg_val)
            
            if random.random() > 0.5: # 70% cơ hội biến đổi phối cảnh (chéo/nghiêng)
                # max_distortion=0.15 nghĩa là các góc có thể dịch chuyển tới 15%
                img_cv = random_perspective(img_cv, max_distortion=0.15, bg_color=bg_val)
                
            # --- KẾT THÚC PHẦN BIẾN ĐỔI HÌNH HỌC ---

            # 3. Xử lý hậu kỳ (Nhiễu và Blur) trên ảnh đã biến đổi
            img_np = img_cv.astype(np.float32)
            
            # Thêm nhiễu nhẹ
            noise = np.random.normal(0, 1.5, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            
            # Blur nhẹ
            img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
            
            # 4. Lưu file
            file_name = f"time_{i:05d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), img_np)
            f_label.write(f"{file_name}\t{label_text}\n")

    print(f"Hoàn tất! Kiểm tra thư mục '{OUTPUT_DIR}'. Ảnh đã có các biến thể xoay và nghiêng.")

if __name__ == "__main__":
    create_synthetic_data()