import cv2
import os
from ultralytics import YOLO

# ================= CONFIG =================
VIDEO_PATH = "video/record_2026-05-30_13-55-47.avi"
# MODEL_PATH = "runs/detect/ppe-2class-3/weights/best.pt"
MODEL_PATH="runs/detect/ppe-2class-6/weights/best.pt"
CONFIDENCE = 0.5
IMGSZ = 640
DISPLAY_MAX_WIDTH = 1280  # 👈 Chỉnh nhỏ hơn nếu màn hình chật
# ===========================================

def check_video_file(path):
    """Kiểm tra file video có tồn tại và OpenCV có mở được không"""
    if not os.path.exists(path):
        print(f"❌ Lỗi: Không tìm thấy file '{path}'")
        print(f"   Đường dẫn hiện tại: {os.getcwd()}")
        return False
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"❌ Lỗi: OpenCV không thể mở video '{path}'")
        print("   → Thử convert video sang H.264 codec bằng FFmpeg:")
        print(f"      ffmpeg -i {path} -c:v libx264 -preset fast truanay_fixed.mp4")
        cap.release()
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Video OK: {width}x{height} @ {fps:.1f} FPS")
    cap.release()
    return True

def main():
    # 1️⃣ Kiểm tra video trước khi chạy
    if not check_video_file(VIDEO_PATH):
        return
    
    # 2️⃣ Load model
    print(f"🔄 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # 3️⃣ Tạo cửa sổ hiển thị
    cv2.namedWindow("YOLOv11 PPE Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv11 PPE Detection", min(1280, DISPLAY_MAX_WIDTH), 720)
    
    print(f"🎬 Bắt đầu xử lý: {VIDEO_PATH} (nhấn 'q' để thoát)")
    
    # 4️⃣ Predict với stream=True ⭐ QUAN TRỌNG
    results = model.predict(
        source=VIDEO_PATH,
        conf=CONFIDENCE,
        imgsz=IMGSZ,
        verbose=False,
        show=False,      # Tắt window mặc định của Ultralytics
        save=True,       # Vẫn lưu kết quả ra runs/detect/predict/
        # ⭐ Xử lý từng frame, không load hết vào RAM
    )
    
    # 5️⃣ Vòng lặp hiển thị
    frame_count = 0
    for result in results:
        annotated_frame = result.plot()
        
        # Resize để hiển thị (giữ aspect ratio)
        h, w = annotated_frame.shape[:2]
        scale = DISPLAY_MAX_WIDTH / w
        new_w, new_h = int(w * scale), int(h * scale)
        display_frame = cv2.resize(annotated_frame, (new_w, new_h), 
                                   interpolation=cv2.INTER_AREA)
        
        cv2.imshow("YOLOv11 PPE Detection", display_frame)
        
        # Hiển thị FPS đơn giản
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"📊 Đã xử lý {frame_count} frames...")
        
        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Người dùng yêu cầu thoát.")
            break
    
    # Dọn dẹp
    cv2.destroyAllWindows()
    print("✅ Hoàn tất!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Dừng bởi người dùng (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Lỗi không mong đợi: {e}")
        import traceback
        traceback.print_exc()