import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Cấu hình
OUTPUT_DIR = "D:/model2-20260110T081631Z-3-001/model2/datasets/train2/images"
LABEL_FILE = "D:/model2-20260110T081631Z-3-001/model2/datasets/train2/labels.txt"
FONT_PATH = "D:/Downloads/SF-Pro.ttf"  
IMG_SIZE = (50, 20)      
NUM_SAMPLES = 1000      

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_time_variants():
    """Tạo chuỗi giờ ngẫu nhiên"""
    h = random.randint(0, 23)
    m = random.randint(0, 59)
    image_text = f"{h:02d}:{m:02d}"
    label_text = f"{h:02d}{m:02d}"
    return image_text, label_text

def create_synthetic_data():
    print(f"Bắt đầu tạo dữ liệu với font size cực đại...")
    with open(LABEL_FILE, "w", encoding="utf-8") as f_label:
        for i in range(NUM_SAMPLES):
            # 1. Tạo nền xám đậm
            bg_val = random.randint(30, 45)
            img_pil = Image.new('L', IMG_SIZE, color=bg_val)
            draw = ImageDraw.Draw(img_pil)
            
            image_text, label_text = get_time_variants()
            
            font_size = 20
            try:
                font = ImageFont.truetype(FONT_PATH, font_size)
            except:
                print("LỖI: Không tìm thấy font SF-Pro.ttf! Hãy đảm bảo file font nằm cùng thư mục.")
                return # Dừng script nếu không có font

            # 2. Vẽ chữ căn giữa tuyệt đối.
            # anchor="mm" sẽ giữ trọng tâm chữ ở giữa ảnh.
            text_color = random.randint(220, 255)
            draw.text((IMG_SIZE[0]/2, IMG_SIZE[1]/2), image_text, 
                      fill=text_color, font=font, anchor="mm")
            
            # 3. Xử lý hậu kỳ (Nhiễu và Blur)
            img_np = np.array(img_pil).astype(np.float32)
            
            # Thêm nhiễu nhẹ
            noise = np.random.normal(0, 1.5, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            
            # Blur nhẹ để tạo cảm giác ảnh thật
            img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
            
            # 4. Lưu file
            file_name = f"time_{i:05d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), img_np)
            f_label.write(f"{file_name}\t{label_text}\n")

    print(f"Hoàn tất! Hãy kiểm tra thư mục '{OUTPUT_DIR}'. Chữ số bây giờ sẽ rất lớn.")

if __name__ == "__main__":
    create_synthetic_data()