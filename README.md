# 🤖 Robot Điều Khiển Bằng Mạng Nơron và PSO

Dự án này mô phỏng và điều khiển robot sử dụng mạng nơron và thuật toán PSO (Particle Swarm Optimization). Robot được trang bị cảm biến để nhận biết môi trường và tự động tìm đường đến đích.

## 🎯 Tính năng chính

- Mô phỏng robot với 3 bánh omniwheel
- Sử dụng mạng nơron để xử lý dữ liệu cảm biến và ra quyết định
- Tối ưu hóa tham số mạng nơron bằng thuật toán PSO
- Môi trường mô phỏng 2D với các chướng ngại vật
- Hệ thống cảm biến 12 tia để nhận biết môi trường

## 📁 Cấu trúc dự án

```
.
├── robot.py        # Mã nguồn chính chứa các lớp Robot, Envir và thuật toán PSO
├── map.png         # Bản đồ môi trường
└── xe.png          # Hình ảnh robot
```

## 🛠 Công nghệ sử dụng

- Python
- Pygame (đồ họa và mô phỏng)
- PyTorch (mạng nơron)
- NumPy (tính toán ma trận)

## 🚀 Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install pygame torch numpy
```

2. Chạy chương trình:
```bash
python robot.py
```

## 📊 Cấu trúc mạng nơron

Mạng nơron được thiết kế với các lớp:
- Input: 16 nơron (12 cảm biến + sinθ, cosθ + x, y)
- Hidden layers: 3 lớp với kích thước 48, 24, 12 nơron
- Output: 3 nơron (điều khiển tốc độ 3 bánh xe)

## 🔄 Thuật toán PSO

- Số lượng particles: 1000
- Kích thước vector tham số: 2367
- Hệ số quán tính (w): 0.5
- Hệ số học cá nhân (c1): 1.5
- Hệ số học xã hội (c2): 1.5
- Số vòng lặp tối đa: 100

## 📈 Đánh giá hiệu suất

Robot được đánh giá dựa trên:
- Thời gian đến đích
- Số lần va chạm
- Số điểm chiến lược đi qua
- Khả năng tránh bế tắc

## 📝 Lưu ý

- Đảm bảo các file ảnh (map.png, xe.png) nằm cùng thư mục với robot.py
- Chương trình yêu cầu Python 3.x và các thư viện đã cài đặt
- Kết quả tối ưu hóa sẽ được lưu vào file best_model.pt
