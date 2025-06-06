# Video MME Demo

## Giới thiệu
Dự án này là một demo xử lý video, bao gồm việc trích xuất khung hình từ video và xử lý ảnh.

## Cấu trúc thư mục
- `src/main.py`: File chính để chạy chương trình.
- `sample_videos/`: Chứa video mẫu để xử lý.
- `extracted_frames/`: Chứa các khung hình được trích xuất từ video.
- `requirements.txt`: Danh sách các thư viện cần thiết.

## Hướng dẫn cài đặt và chạy

### 1. Cài đặt môi trường
1. Cài đặt Python (phiên bản >= 3.8).
2. Tạo môi trường ảo:
   ```bash
   python -m venv venv
   ```
3. Kích hoạt môi trường ảo:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

### 2. Cài đặt các thư viện cần thiết
Chạy lệnh sau để cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

### 3. Chạy chương trình
Chạy file `main.py` bằng lệnh:
```bash
streamlit run src/main.py
```

### 4. Lưu ý
- Đảm bảo video cần xử lý nằm trong thư mục `sample_videos/`.
- Các khung hình trích xuất sẽ được lưu trong thư mục `extracted_frames/`.

## Liên hệ
Nếu có bất kỳ câu hỏi nào, vui lòng liên hệ qua email: [haidang29productionsl@example.com].
