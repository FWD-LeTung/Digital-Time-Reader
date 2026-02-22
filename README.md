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
