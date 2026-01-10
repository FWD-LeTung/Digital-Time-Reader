import os
import cv2

folder_path = "D:/model2-20260110T081631Z-3-001/model2/datasets/test/images"

total_width = 0
total_height = 0
count = 0

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is not None:
            height, width, _ = img.shape
            total_width += width
            total_height += height
            count += 1

if count > 0:
    print(f"Số ảnh: {count}")
    print(f"Width trung bình: {total_width / count:.2f}")
    print(f"Height trung bình: {total_height / count:.2f}")
else:
    print("Không tìm thấy ảnh hợp lệ.")
