# Digital-Time-Reader

# Digital-Time-Reader

Dự án này cung cấp một giải pháp học sâu (deep learning) toàn diện để nhận diện thời gian từ các màn hình đồng hồ kỹ thuật số. Quy trình thực hiện bao gồm hai giai đoạn chính: phát hiện vùng chứa đồng hồ (Detection) và nhận diện chuỗi chữ số (OCR).

## 📌 Tính năng chính
* **Phát hiện đối tượng:** Sử dụng YOLOv8 (Ultralytics) để xác định chính xác vị trí khung hiển thị thời gian trong ảnh hoặc luồng video.
* **Nhận diện chữ số (OCR):** Triển khai kiến trúc **CRNN** (Convolutional Recurrent Neural Network) với tầng **GRU** hai chiều, cho phép đọc chuỗi thời gian mà không cần phân tách từng chữ số riêng biệt.
* **Xử lý chuỗi linh hoạt:** Sử dụng hàm mất mát **CTC Loss** để huấn luyện mô hình nhận diện các chuỗi có độ dài khác nhau.
* **Dữ liệu tổng hợp:** Tích hợp các công cụ sinh dữ liệu giả lập (synthetic data) để tăng cường độ chính xác cho mô hình trong nhiều điều kiện ánh sáng và phông chữ khác nhau.

## 🛠 Cài đặt
Để cài đặt các thư viện cần thiết, hãy chạy lệnh sau:
```bash
pip install -r requirements.txt

Các thư viện chính bao gồm:

ultralytics>=8.1.0 (YOLOv8)

torch>=2.0 & torchvision>=0.15

opencv-python>=4.8

numpy, Pillow, tqdm

📂 Cấu trúc thư mục
Dựa trên cấu trúc dự án hiện tại:

datasets/: Chứa các script quản lý và sinh dữ liệu tổng hợp (synthetic.py, crnn_dataset.py).

demo/: Các kịch bản chạy thử nghiệm thực tế trên Video và Webcam (hỗ trợ cả pipeline CNN và CRNN).

model/: Lưu trữ các tệp trọng số đã huấn luyện:

bestDetect.pt: Mô hình YOLOv8 để phát hiện vùng đồng hồ.

crnn_synthetic_gray.pth: Mô hình CRNN nhận diện chữ số.

src/: Mã nguồn huấn luyện chính:

train_CRNN.py: Huấn luyện mô hình OCR.

trainDetect.py: Huấn luyện mô hình phát hiện vùng đồng hồ.

utils/: Các công cụ hỗ trợ xử lý hình ảnh như phép biến đổi phối cảnh (four_point_transform.py).

🚀 Hướng dẫn sử dụng
1. Huấn luyện mô hình OCR (CRNN)
Để huấn luyện lại mô hình nhận diện với dữ liệu của bạn, hãy cập nhật đường dẫn DATASET_ROOT trong file src/train_CRNN.py và chạy:

Bash
python src/train_CRNN.py
2. Chạy Demo thực tế
Để chạy hệ thống nhận diện thời gian qua Webcam sử dụng Pipeline CRNN:

Bash
python demo/FullPipelineCRNN/webcam.py
📊 Chi tiết kỹ thuật mô hình CRNN
Đầu vào: Ảnh xám (Grayscale), kích thước 50x20 pixel.

Kiến trúc CNN: 2 tầng Conv2d (32 và 64 filters) giúp trích xuất đặc trưng không gian.

Kiến trúc RNN: Tầng GRU hai chiều (Bi-directional GRU) với 64 hidden units để học đặc trưng chuỗi.

Giải mã: Sử dụng greedy_decode để chuyển đổi kết quả từ mô hình thành chuỗi văn bản (0-9).

Mất mát: Sử dụng CTCLoss để tối ưu hóa việc nhận diện chuỗi.
