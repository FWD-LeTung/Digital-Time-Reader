import cv2
import numpy as np
import os
import argparse


def process_to_four_folders(input_folder, output_base="dataset_mnist_ready"):
    # 1. Khởi tạo 4 thư mục con
    digit_folders = [os.path.join(output_base, f"digit_{i+1}") for i in range(4)]
    for folder in digit_folders:
        os.makedirs(folder, exist_ok=True)

    # Lấy danh sách ảnh
    valid_extensions = ('.jpg', '.png', '.jpeg')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    print(f"--- Đang xử lý {len(image_files)} ảnh ---")

    skipped = 0
    saved = 0

    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # ================= PIPELINE XỬ LÝ =================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h_img, _ = img.shape[:2]
        MIN_H = h_img * 0.2

        raw_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > MIN_H:
                raw_boxes.append([x, y, w, h])

        # Sắp xếp từ trái sang phải
        raw_boxes.sort(key=lambda b: b[0])

        current_image_crops = []

        for x, y, w, h in raw_boxes:
            ratio = w / h

            # Nếu box quá béo -> khả năng 2 digit dính nhau
            if ratio > 0.6:
                mid = w // 2
                crop1 = img[y:y+h, x:x+mid]
                crop2 = img[y:y+h, x+mid:x+w]
                current_image_crops.extend([crop1, crop2])
            else:
                crop = img[y:y+h, x:x+w]
                current_image_crops.append(crop)

        # ================= KIỂM TRA NGHIÊM NGẶT =================
        # CHỈ CHẤP NHẬN ĐÚNG 4 DIGIT
        if len(current_image_crops) != 4:
            skipped += 1
            # print(f"[SKIP] {filename}: detected {len(current_image_crops)} digits")
            continue

        # ================= LƯU KẾT QUẢ =================
        base_name = os.path.splitext(filename)[0]
        for i in range(4):
            save_path = os.path.join(
                digit_folders[i],
                f"{base_name}_d{i}.png"
            )
            cv2.imwrite(save_path, current_image_crops[i])

        saved += 1

    print("\n========== SUMMARY ==========")
    print(f"Saved images : {saved}")
    print(f"Skipped images: {skipped}")
    print(f"Output folder: {output_base}")
    print("=============================\n")


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split clock images into exactly 4 digit crops (HHMM)."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing cropped clock images"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save digit_1 ... digit_4"
    )

    args = parser.parse_args()

    process_to_four_folders(args.input_dir, args.output_dir)
