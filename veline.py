import cv2
import json
import numpy as np
import os

# ================= 1. CẤU HÌNH ĐẦU VÀO =================
# Bác nhớ đổi lại tên ảnh cho khớp với ảnh cắt từ camera thực tế của bác
IMAGE_PATH = 'snapshot_2026-05-27T01-39-57.jpg' 
CONFIG_FILE = 'camera_config2.json'

drawing_mode = 'ROI'
points = {
    'ROI': [],
    'HANDRAIL_LEFT_POLY': [],  
    'HANDRAIL_RIGHT_POLY': [], 
    'STEP_BOTTOM': [],
    'STEP_TOP': []
}

def get_midpoint(p1, p2):
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

# ================= 2. HÀM XỬ LÝ CHUỘT =================
def mouse_handler(event, x, y, flags, param):
    global drawing_mode, points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Bậc đáy và Bậc đỉnh chỉ lấy 2 điểm (vẽ đường thẳng)
        if drawing_mode in ['STEP_BOTTOM', 'STEP_TOP'] and len(points[drawing_mode]) >= 2:
            print(f"[{drawing_mode}] đã đủ 2 điểm! Bấm 'C' nếu muốn vẽ lại.")
            return
            
        # Lan can và ROI là Đa giác (Poly) nên cho phép click nhiều điểm
        points[drawing_mode].append((x, y))
        render_ui()

# ================= 3. HÀM RENDER GIAO DIỆN =================
def render_ui():
    img_display = img_clone.copy()
    
    # 1. Vẽ ROI (Xanh lá)
    if len(points['ROI']) > 0:
        pts = np.array(points['ROI'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for p in points['ROI']: cv2.circle(img_display, p, 5, (0, 0, 255), -1)

    # 2. Vẽ Đa giác Lan can Trái (Cam)
    if len(points['HANDRAIL_LEFT_POLY']) > 0:
        pts = np.array(points['HANDRAIL_LEFT_POLY'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_display, [pts], isClosed=True, color=(0, 165, 255), thickness=2) 
        if len(points['HANDRAIL_LEFT_POLY']) >= 3: 
            overlay = img_display.copy()
            cv2.fillPoly(overlay, [pts], (0, 165, 255))
            cv2.addWeighted(overlay, 0.3, img_display, 0.7, 0, img_display)
        for p in points['HANDRAIL_LEFT_POLY']: cv2.circle(img_display, p, 5, (0, 165, 255), -1)

    # 3. Vẽ Đa giác Lan can Phải (Hồng)
    if len(points['HANDRAIL_RIGHT_POLY']) > 0:
        pts = np.array(points['HANDRAIL_RIGHT_POLY'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_display, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
        if len(points['HANDRAIL_RIGHT_POLY']) >= 3: 
            overlay = img_display.copy()
            cv2.fillPoly(overlay, [pts], (255, 0, 255))
            cv2.addWeighted(overlay, 0.3, img_display, 0.7, 0, img_display)
        for p in points['HANDRAIL_RIGHT_POLY']: cv2.circle(img_display, p, 5, (255, 0, 255), -1)

    # 4. Vẽ Cầu thang & Trục giữa
    mid_bottom, mid_top = None, None
    if len(points['STEP_BOTTOM']) == 2:
        cv2.line(img_display, points['STEP_BOTTOM'][0], points['STEP_BOTTOM'][1], (0, 0, 255), 2)
        mid_bottom = get_midpoint(points['STEP_BOTTOM'][0], points['STEP_BOTTOM'][1])
        cv2.circle(img_display, mid_bottom, 6, (0, 255, 255), -1)
    for p in points['STEP_BOTTOM']: cv2.circle(img_display, p, 5, (0, 0, 255), -1)

    if len(points['STEP_TOP']) == 2:
        cv2.line(img_display, points['STEP_TOP'][0], points['STEP_TOP'][1], (255, 0, 0), 2)
        mid_top = get_midpoint(points['STEP_TOP'][0], points['STEP_TOP'][1])
        cv2.circle(img_display, mid_top, 6, (0, 255, 255), -1) 
    for p in points['STEP_TOP']: cv2.circle(img_display, p, 5, (255, 0, 0), -1)

    if mid_bottom and mid_top:
        cv2.line(img_display, mid_bottom, mid_top, (0, 255, 255), 3)
        cv2.putText(img_display, "CENTER LINE", (mid_top[0] + 10, mid_top[1] + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # UI Hướng dẫn
    menu_text = [
        f"MODE: {drawing_mode}",
        "[1] ROI Safe Zone (Màu Xanh lá - Chấm 4 điểm)",
        "[2] Lan can TRÁI (Màu Cam - Chấm dọc lan can)",
        "[3] Lan can PHẢI (Màu Hồng - Chấm dọc mép tường)",
        "[4] Bậc THẤP NHẤT (Chấm 2 điểm GẦN camera)",
        "[5] Bậc CAO NHẤT (Chấm 2 điểm XA camera)",
        "[C] Xóa vùng vẽ hiện tại | [S] LƯU & THOÁT"
    ]
    
    for i, text in enumerate(menu_text):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(img_display, text, (20, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow("Smart Config UI", img_display)

# ================= 4. KHỞI CHẠY =================
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Lỗi: Không tìm thấy file ảnh '{IMAGE_PATH}'!")
    exit()

img = cv2.imread(IMAGE_PATH)
img_clone = img.copy()

cv2.namedWindow("Smart Config UI", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Smart Config UI", mouse_handler)
render_ui()

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('1'): drawing_mode = 'ROI'; render_ui()
    elif key == ord('2'): drawing_mode = 'HANDRAIL_LEFT_POLY'; render_ui()
    elif key == ord('3'): drawing_mode = 'HANDRAIL_RIGHT_POLY'; render_ui()
    elif key == ord('4'): drawing_mode = 'STEP_BOTTOM'; render_ui()
    elif key == ord('5'): drawing_mode = 'STEP_TOP'; render_ui()
    elif key == ord('c'):
        points[drawing_mode] = []
        render_ui()
    elif key == ord('s'):
        if len(points['STEP_BOTTOM']) == 2 and len(points['STEP_TOP']) == 2:
            points['CENTER_LINE'] = [
                get_midpoint(points['STEP_BOTTOM'][0], points['STEP_BOTTOM'][1]),
                get_midpoint(points['STEP_TOP'][0], points['STEP_TOP'][1])
            ]
        with open(CONFIG_FILE, 'w') as f:
            json.dump(points, f, indent=4)
        print("✅ Đã lưu cấu hình không gian vào camera_config.json thành công!")
        break
    elif key == ord('q'):
        break

cv2.destroyAllWindows()