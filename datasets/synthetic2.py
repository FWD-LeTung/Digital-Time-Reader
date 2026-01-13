import os
os.environ["OMP_NUM_THREADS"] = "8"
import random
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool, cpu_count

# --- CẤU HÌNH ---
OUTPUT_DIR = "D:/model2-20260110T081631Z-3-001/model2/datasets/test4/images"
LABEL_FILE = "D:/model2-20260110T081631Z-3-001/model2/datasets/test4/labels.txt"
FONT_PATH = "D:/Downloads/SF-Pro.ttf"

IMG_SIZE = (50, 20)
NUM_SAMPLES = 110000

FORBIDDEN_TIMES = {}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ----------------- TIME -----------------

def get_time_variants():
    while True:
        h = random.randint(0, 23)
        m = random.randint(0, 59)
        label_text = f"{h:02d}{m:02d}"
        if label_text not in FORBIDDEN_TIMES:
            return f"{h:02d}:{m:02d}", label_text

# ----------------- AUGMENT -----------------

def add_peripheral_clutter(image):
    draw_img = Image.fromarray(image)
    draw = ImageDraw.Draw(draw_img)
    if random.random() > 0.98:
        y = random.randint(18, 19)
        draw.line([(0, y), (50, y)], fill=random.randint(100, 150), width=1)
    return np.array(draw_img)

def random_rotate(image, max_angle, bg):
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bg)

def random_perspective(image, max_distortion, bg):
    h, w = image.shape
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    d_h = h * max_distortion
    d_w = w * max_distortion
    mode = random.randint(0,3)

    if mode == 0:
        pts2 = np.float32([[0,0],[w,d_h],[0,h],[w,h-d_h]])
    elif mode == 1:
        pts2 = np.float32([[0,d_h],[w,0],[0,h-d_h],[w,h]])
    elif mode == 2:
        pts2 = np.float32([[d_w,0],[w-d_w,0],[0,h],[w,h]])
    else:
        pts2 = np.float32([[0,0],[w,0],[d_w,h],[w-d_w,h]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=bg)

# -------- MID-DISTANCE EFFECTS --------

def simulate_mid_distance(image):
    scale = random.uniform(0.75, 0.9)
    h, w = image.shape
    small = cv2.resize(image, (int(w*scale), int(h*scale)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h),
                      interpolation=cv2.INTER_LINEAR)

def mild_blur(image):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(image, (k, k),
                            random.uniform(0.7, 1.0))

def very_light_motion(image):
    if random.random() > 0.85:
        kernel = np.zeros((3,3))
        kernel[1, :] = 1/3
        return cv2.filter2D(image, -1, kernel)
    return image

def slight_contrast_drop(image):
    alpha = random.uniform(0.8, 0.95)
    beta = random.randint(-5, 5)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def camera_noise(image):
    sigma = random.uniform(1.5, 3.0)
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

# ----------------- GENERATE -----------------

def generate_one(i):
    bg_val = random.randint(30, 45)
    img_pil = Image.new('L', IMG_SIZE, color=bg_val)
    draw = ImageDraw.Draw(img_pil)

    image_text, label_text = get_time_variants()

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except:
        font = ImageFont.load_default()

    draw.text((IMG_SIZE[0]/2, IMG_SIZE[1]/2),
              image_text,
              fill=random.randint(220, 255),
              font=font,
              anchor="mm")

    img_cv = np.array(img_pil)

    # Geometry
    if random.random() > 0.9:
        img_cv = random_rotate(img_cv, 5, bg_val)

    if random.random() > 0.7:
        img_cv = random_perspective(img_cv, 0.1, bg_val)

    if random.random() > 0.97:
        img_cv = add_peripheral_clutter(img_cv)

    # ---- MID DISTANCE ----
    if random.random() > 0.7:
        img_cv = simulate_mid_distance(img_cv)

    if random.random() > 0.7:
        img_cv = mild_blur(img_cv)


    if random.random() > 0.5:
        img_cv = slight_contrast_drop(img_cv)

    img_cv = camera_noise(img_cv)

    img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)
    file_name = f"time2_{i:05d}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), img_cv)

    return f"{file_name}\t{label_text}\n"

# ----------------- MAIN -----------------

def create_synthetic_data():
    n_proc = min(cpu_count(), 8)
    print(f"🚀 Multiprocessing với {n_proc} process")

    labels = []
    with Pool(processes=n_proc) as pool:
        for r in tqdm(pool.imap_unordered(generate_one,
                                          range(NUM_SAMPLES)),
                      total=NUM_SAMPLES):
            labels.append(r)

    with open(LABEL_FILE, "w", encoding="utf-8") as f:
        f.writelines(labels)

    print("✅ Hoàn tất sinh dữ liệu")

if __name__ == "__main__":
    create_synthetic_data()
