import cv2
import os

def extract_frames_from_video(video_path, output_folder, step=5):
    # Lấy tên file video (không bao gồm đuôi .mp4, .avi...)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Tạo thư mục con riêng cho video này bên trong output_folder
    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    # Đọc video
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    saved = 0

    while success:
        if count % step == 0:
            # Lưu frame với tên bao gồm cả tên video để dễ nhận diện
            frame_name = os.path.join(video_output_folder, f"{video_name}_frame_{saved:04d}.jpg")
            cv2.imwrite(frame_name, image)
            saved += 1
        
        # Đọc frame tiếp theo
        success, image = vidcap.read()
        count += 1
    
    # Giải phóng tài nguyên sau khi đọc xong video
    vidcap.release()
    print(f"  -> Đã tách xong {saved} frames từ video '{video_name}'.")

def process_video_folder(input_folder, output_folder, step=10):
    # Kiểm tra thư mục đầu ra tổng
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Các định dạng video cho phép
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')

    # Duyệt qua toàn bộ file trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            video_path = os.path.join(input_folder, filename)
            print(f"Đang xử lý: {filename}...")
            
            # Gọi hàm tách frame cho từng video
            extract_frames_from_video(video_path, output_folder, step)

    print("\nHoàn thành! Đã xử lý tất cả video trong thư mục.")

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
# Cấu hình đường dẫn thư mục chứa video và thư mục lưu frame
INPUT_DIR = "video"   # Thư mục chứa các video của bạn
OUTPUT_DIR = "val2"   # Thư mục đích để lưu ảnh
STEP_SIZE = 1                  # Lấy 1 frame sau mỗi 15 frames

# Chạy chương trình
process_video_folder(INPUT_DIR, OUTPUT_DIR, step=STEP_SIZE)