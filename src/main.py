# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import os
import math
import shutil
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import io
import tempfile # Thư viện để tạo file/thư mục tạm

# --- Load API Key và Cấu hình Gemini (Giữ nguyên) ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    # Hiển thị lỗi trên giao diện thay vì print và exit
    st.error("Lỗi: Không tìm thấy GOOGLE_API_KEY trong file .env. Vui lòng tạo file .env ở thư mục gốc và thêm key vào.")
    st.stop() # Dừng thực thi ứng dụng
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Lỗi khi cấu hình Gemini API: {e}")
    st.stop()

def extract_and_save_frames(video_path, output_folder, interval_sec=1):
    frames = []
    video_capture = None
    try:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        st.info(f"Đã tạo thư mục lưu khung hình tạm: {output_folder}") # Thông báo trên UI

        if not os.path.exists(video_path):
            st.error(f"Lỗi: File video không tồn tại tại '{video_path}'")
            return None
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            st.error(f"Lỗi: Không thể mở file video '{video_path}'")
            return None
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            st.warning(f"Cảnh báo: Không thể lấy FPS hợp lệ từ video '{video_path}'. Giả sử là 30.")
            fps = 30
        frame_interval = int(fps * interval_sec)
        if frame_interval <= 0:
            frame_interval = 1
        frame_count = 0
        saved_frame_count = 0

        progress_bar = st.progress(0, text="Đang trích xuất khung hình...") # Thêm thanh tiến trình
        total_frames_approx = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) # Ước lượng tổng số frame

        while True:
            success, frame = video_capture.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                saved_frame_count += 1
                frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
                try:
                    cv2.imwrite(frame_filename, frame)
                except Exception as write_e:
                    st.warning(f"Không thể ghi file frame: {frame_filename} - Lỗi: {write_e}")


            frame_count += 1
            # Cập nhật thanh tiến trình
            if total_frames_approx > 0:
                 progress = min(1.0, frame_count / total_frames_approx)
                 progress_bar.progress(progress, text=f"Đang trích xuất khung hình... ({frame_count}/{total_frames_approx})")
            else:
                 progress_bar.progress(frame_count % 100 / 100.0 , text=f"Đang trích xuất khung hình... (frame {frame_count})") # Ước lượng nếu không có total_frames

        progress_bar.progress(1.0, text="Hoàn thành trích xuất!")
        st.success(f"Đã trích xuất và lưu thành công {len(frames)} khung hình vào thư mục tạm.")
        return frames

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi xử lý video: {e}")
        return None
    finally:
        if video_capture:
            video_capture.release()
            st.info("Đã giải phóng tài nguyên video.")


def ask_gemini_vision(prompt, frames):
    model_name = "gemini-1.5-flash-latest"
    try:
        model = genai.GenerativeModel(model_name) # Thử lại với flash
        st.info(f"Đang sử dụng mô hình: {model_name}")

        image_parts = []
        for frame_np in frames:
            try:
                pil_image = Image.fromarray(frame_np)
                image_parts.append(pil_image)
            except Exception as e:
                st.warning(f"Lỗi khi chuyển đổi khung hình sang PIL Image: {e}")

        if not image_parts:
            st.error("Lỗi: Không có khung hình hợp lệ để gửi đến API.")
            return None, model_name

        request_content = [prompt] + image_parts
        st.info(f"Đang gửi yêu cầu đến Gemini với {len(image_parts)} khung hình...")

        # Sử dụng spinner để cho biết đang chờ API
        with st.spinner("Đang chờ phản hồi từ Gemini..."):
            response = model.generate_content(request_content, stream=False,
                                             safety_settings={
                                                 'HATE': 'BLOCK_NONE', 'HARASSMENT': 'BLOCK_NONE',
                                                 'SEXUAL': 'BLOCK_NONE', 'DANGEROUS': 'BLOCK_NONE'
                                             })

        if not response.parts:
             if response.prompt_feedback.block_reason:
                 st.error(f"Yêu cầu bị chặn do: {response.prompt_feedback.block_reason}")
             else:
                 st.error("Lỗi: API không trả về nội dung nào.")
             return None, model_name

        answer_text = response.text
        st.success("Đã nhận phản hồi từ Gemini.")
        return answer_text, model_name

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi gọi Gemini API: {e}")
        if hasattr(e, 'args') and e.args:
            st.error(f"Chi tiết lỗi API: {e.args}")
        return None, None


# --- Giao diện Streamlit ---
st.set_page_config(page_title="Demo Video-MME", layout="wide")
st.title("🎬 Demo Phân tích Video với Gemini (Video-MME)")
st.markdown("Tải lên một video, nhập câu hỏi và các lựa chọn, sau đó xem Gemini phân tích!")

# --- Cột chính ---
col1, col2 = st.columns([2, 1]) # Chia layout thành 2 cột

with col1: # Cột bên trái cho upload và input
    st.header("1. Tải Video Lên")
    uploaded_file = st.file_uploader(
        "Chọn file video (MP4, AVI):",
        type=["mp4", "avi"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Hiển thị video đã tải lên
        st.video(uploaded_file)

        st.header("2. Nhập Câu Hỏi và Lựa Chọn")
        question = st.text_area("Nhập câu hỏi của bạn:", height=100)
        options = {}
        options['A'] = st.text_input("Lựa chọn A:")
        options['B'] = st.text_input("Lựa chọn B:")
        options['C'] = st.text_input("Lựa chọn C:")
        options['D'] = st.text_input("Lựa chọn D:")

        correct_answer = st.radio(
            "Chọn đáp án đúng:",
            ('A', 'B', 'C', 'D'),
            horizontal=True
        )

        st.header("3. Cấu hình (Tùy chọn)")
        frame_interval_seconds = st.slider(
            "Khoảng cách giữa các khung hình (giây):",
            min_value=0.5, max_value=10.0, value=1.0, step=0.5
        )
        MAX_FRAMES_TO_SEND = st.number_input(
             "Số khung hình tối đa gửi đến API:",
             min_value=5, max_value=100, value=50, step=5
        )

        # Nút để bắt đầu phân tích
        analyze_button = st.button("🚀 Bắt đầu Phân tích Video", type="primary", use_container_width=True)

    else:
        st.info("Vui lòng tải lên một file video để bắt đầu.")
        analyze_button = False # Không hiển thị nút nếu chưa có video


with col2: # Cột bên phải cho kết quả
    st.header("📈 Kết Quả Phân Tích")

    if analyze_button:
        # --- Kiểm tra đầu vào ---
        if not question or not all(options.values()):
            st.warning("Vui lòng nhập đầy đủ câu hỏi và tất cả các lựa chọn.")
        else:
            # --- Xử lý ---
            st.info("Bắt đầu quá trình xử lý...")

            # Tạo thư mục tạm để lưu video và frames
            with tempfile.TemporaryDirectory() as temp_dir:
                # Lưu video tải lên vào file tạm
                temp_video_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.info(f"Đã lưu video tạm thời tại: {temp_video_path}")

                # Tạo thư mục con để lưu frame trong thư mục tạm
                frame_output_folder = os.path.join(temp_dir, "extracted_frames")

                # --- Bước 1: Trích xuất và lưu khung hình ---
                with st.spinner("Đang trích xuất khung hình..."):
                     extracted_frames = extract_and_save_frames(temp_video_path, frame_output_folder, frame_interval_seconds)

                if extracted_frames:
                    st.success(f"Đã trích xuất {len(extracted_frames)} khung hình.")

                    # Hiển thị một vài khung hình mẫu
                    st.subheader("Khung hình mẫu:")
                    num_frames_to_show = min(len(extracted_frames), 6) # Hiển thị tối đa 6 frame
                    sample_indices = [int(i * (len(extracted_frames)-1) / (num_frames_to_show-1)) for i in range(num_frames_to_show)] if num_frames_to_show > 1 else [0]
                    frame_cols = st.columns(num_frames_to_show)
                    for i, idx in enumerate(sample_indices):
                         with frame_cols[i]:
                             st.image(extracted_frames[idx], caption=f"Frame ~{idx*frame_interval_seconds:.1f}s", use_container_width=True)


                    # Giới hạn số lượng khung hình gửi đi
                    if len(extracted_frames) > MAX_FRAMES_TO_SEND:
                        st.info(f"Giới hạn số lượng khung hình gửi đi là {MAX_FRAMES_TO_SEND}")
                        indices = [int(i * (len(extracted_frames)-1) / (MAX_FRAMES_TO_SEND-1)) for i in range(MAX_FRAMES_TO_SEND)]
                        frames_to_send = [extracted_frames[i] for i in indices]
                    else:
                        frames_to_send = extracted_frames

                    # --- Bước 2: Chuẩn bị Prompt và Gọi Gemini API ---
                    st.subheader("Gọi Gemini API")
                    prompt_text = f"""
Phân tích các khung hình được cung cấp từ một video.
Dựa CHỈ vào nội dung trong các khung hình này, hãy trả lời câu hỏi trắc nghiệm sau:

Câu hỏi: {question}

Các lựa chọn:
A: {options['A']}
B: {options['B']}
C: {options['C']}
D: {options['D']}

Chỉ trả lời bằng MỘT chữ cái duy nhất (A, B, C, hoặc D) đại diện cho lựa chọn đúng nhất. Không giải thích gì thêm.
"""
                    with st.expander("Xem Prompt đã gửi"):
                        st.text(prompt_text)

                    # Gọi hàm API
                    gemini_response_text, used_model_name = ask_gemini_vision(prompt_text, frames_to_send)

                    # --- Bước 3: Hiển thị kết quả chi tiết ---
                    st.subheader("Kết quả từ Gemini")
                    st.write(f"**Câu hỏi:** {question}")
                    st.write("**Lựa chọn:**")
                    for key, value in options.items():
                        st.write(f"    {key}: {value}")
                    st.write(f"**Đáp án đúng (bạn đã chọn):** {correct_answer}")

                    if gemini_response_text is not None and used_model_name is not None:
                        model_answer = gemini_response_text.strip().upper()
                        final_answer = ""
                        if model_answer and model_answer[0] in ["A", "B", "C", "D"]:
                            final_answer = model_answer[0]
                            st.write(f"**Câu trả lời từ Gemini ({used_model_name}):**")
                            st.code(final_answer, language=None) # Hiển thị rõ ràng hơn

                            if final_answer == correct_answer:
                                st.success("✅ Kết quả ĐÚNG!")
                            else:
                                st.error(f"❌ Kết quả SAI. (Gemini chọn {final_answer}, đúng là {correct_answer})")
                        else:
                            st.warning(f"Gemini ({used_model_name}) trả về không đúng định dạng mong đợi:")
                            st.text(gemini_response_text)
                    elif used_model_name is not None:
                        st.error(f"Không nhận được câu trả lời từ Gemini ({used_model_name}) do lỗi.")
                    else:
                        st.error("Không nhận được câu trả lời từ Gemini do lỗi nghiêm trọng.")

                else:
                    st.error("Không thể trích xuất khung hình từ video đã tải lên.")

