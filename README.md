# Digital Time Reader - Đọc Giờ Từ Đồng Hồ Trên Màn Hình Điện Thoại

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Dự án nhận diện và đọc thời gian từ ảnh đồng hồ số sử dụng các kỹ thuật xử lý ảnh và OCR. Có khả năng phát hiện, trích xuất và nhận dạng các chữ số hiển thị trên màn hình đồng hồ điện tử.

## 🎯 Tính Năng Chính

- Đọc thời gian từ ảnh đồng hồ số với độ chính xác cao
- Hỗ trợ nhiều định dạng thời gian (HH:MM, HH:MM:SS)
- Xử lý ảnh với các điều kiện ánh sáng khác nhau
- Giao diện dòng lệnh đơn giản, dễ sử dụng
- Xuất kết quả dưới dạng text hoặc JSON

## 🛠️ Công Nghệ Sử Dụng

- **Python 3.7+**: Ngôn ngữ lập trình chính
- **OpenCV**: Xử lý ảnh và phát hiện vùng chứa số
- **Tesseract OCR**: Nhận dạng ký tự quang học
- **NumPy**: Xử lý ma trận và tính toán số học

## 📋 Yêu Cầu Hệ Thống
- Python 3.7 hoặc cao hơn
- Các thư viện Python trong `requirements.txt`

## 📁 Repository Structure
```
Digital-Time-Reader/
├── datasets/ # Tập dữ liệu dùng để huấn luyện / kiểm thử
├── demo/ # Ví dụ chạy thử hoặc script demo
├── model/ # Mô hình đã huấn luyện
├── src/ # Code chính của dự án
├── requirements.txt # Thư viện và packages cần thiết
└── README.md # Tài liệu này
```
## Cài đặt
```
git clone https://github.com/FWD-LeTung/Digital-Time-Reader.git
cd Digital-Time-Reader
pip install -r requirements.txt
```
## Demo
Chỉ dùng module detect
```
python demo/detection/image.py
python demo/detection/videoinference.py
python demo/detection/webcaminference.py
```
Detect + Reader
```
python demo/FullPipelineCRNN/video.py
python demo/FullPipelineCRNN/webcam.py
```
