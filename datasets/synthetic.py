import os
os.environ["OMP_NUM_THREADS"] = "8"
import random
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool, cpu_count
# --- CẤU HÌNH ---
OUTPUT_DIR = "D:/model2-20260110T081631Z-3-001/model2/datasets/train3/images"
LABEL_FILE = "D:/model2-20260110T081631Z-3-001/model2/datasets/train3/labels.txt"
FONT_PATH = "D:/Downloads/SF-Pro.ttf"  
IMG_SIZE = (50, 20)      
NUM_SAMPLES = 200000

FORBIDDEN_TIMES = {
    "0636", "0737", "0838", "1649", "1749",
    "1845", "1949", "2056", "2257", "2358"
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_time_variants():
    """Tạo chuỗi giờ ngẫu nhiên"""
    while True:
        h = random.randint(0, 23)
        m = random.randint(0, 59)
        label_text = f"{h:02d}{m:02d}"

        if label_text not in FORBIDDEN_TIMES:
            image_text = f"{h:02d}:{m:02d}"
            return image_text, label_text

# --- CÁC HÀM BIẾN ĐỔI HÌNH HỌC MỚI ---

def add_peripheral_clutter(image, bg_color):
    """Thêm các vệt nhiễu ở cạnh ảnh giống thực tế điện thoại"""
    draw_img = Image.fromarray(image)
    draw = ImageDraw.Draw(draw_img)
    # Thêm 1 vệt mờ ở cạnh dưới (giống chữ thừa)
    if random.random() > 0.98:
        y_pos = random.randint(18,19)
        draw.line([(0, y_pos), (50, y_pos)], fill=random.randint(100, 150), width=1)
    return np.array(draw_img)

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
    """Mô phỏng nghiêng mặt phẳng điện thoại theo các hướng cụ thể"""
    h, w = image.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Chọn ngẫu nhiên 1 trong 4 kiểu nghiêng
    mode = random.randint(0, 3) 
    # Tỷ lệ biến dạng (ví dụ 0.2 là nghiêng khá mạnh)
    dist_h = h * max_distortion
    dist_w = w * max_distortion

    if mode == 0: # Nghiêng trái (Cạnh phải co lại)
        pts2 = np.float32([[0, 0], [w, dist_h], [0, h], [w, h - dist_h]])
    elif mode == 1: # Nghiêng phải (Cạnh trái co lại)
        pts2 = np.float32([[0, dist_h], [w, 0], [0, h - dist_h], [w, h]])
    elif mode == 2: # Nghiêng xa (Cạnh trên co lại - nhìn từ dưới lên)
        pts2 = np.float32([[dist_w, 0], [w - dist_w, 0], [0, h], [w, h]])
    else: # Nghiêng gần (Cạnh dưới co lại - nhìn từ trên xuống)
        pts2 = np.float32([[0, 0], [w, 0], [dist_w, h], [w - dist_w, h]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

# -------------------------------------

def generate_one(i):
    bg_val = random.randint(30, 45)
    img_pil = Image.new('L', IMG_SIZE, color=bg_val)
    draw = ImageDraw.Draw(img_pil)

    image_text, label_text = get_time_variants()

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except:
        font = ImageFont.load_default()

    text_color = random.randint(220, 255)
    draw.text(
        (IMG_SIZE[0] / 2, IMG_SIZE[1] / 2),
        image_text,
        fill=text_color,
        font=font,
        anchor="mm"
    )

    img_cv = np.array(img_pil)

    if random.random() > 0.9:
        img_cv = random_rotate(img_cv, 5, bg_val)

    if random.random() > 0.7:
        img_cv = random_perspective(img_cv, 0.1, bg_val)

    if random.random() > 0.98:
        img_cv = add_peripheral_clutter(img_cv, bg_val)

    img_np = img_cv.astype(np.float32)
    noise = np.random.normal(0, 1.5, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

    file_name = f"time2_{i:05d}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), img_np)

    return f"{file_name}\t{label_text}\n"

def create_synthetic_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    n_proc = min(cpu_count(), 8)   # giới hạn an toàn
    print(f"🚀 Chạy multiprocessing với {n_proc} process")

    labels = []

    with Pool(processes=n_proc) as pool:
        for result in tqdm(
            pool.imap_unordered(generate_one, range(NUM_SAMPLES)),
            total=NUM_SAMPLES
        ):
            labels.append(result)

    with open(LABEL_FILE, "w", encoding="utf-8") as f:
        f.writelines(labels)

    print("✅ Hoàn tất sinh dữ liệu")
if __name__ == "__main__":
    create_synthetic_data()