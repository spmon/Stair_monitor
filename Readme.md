# VNA Demo

Repo này chứa các bài toán computer vision phục vụ demo an toàn lao động:

- Giám sát hành vi lên/xuống cầu thang.
- Kiểm tra PPE (mũ bảo hộ, áo phản quang).
- Cảnh báo người đi vào vùng nguy hiểm.
- Xem camera RTSP qua WebSocket và ghi video từ trình duyệt.

## 1. Mục tiêu bài toán

Các nhóm chức năng chính trong repo:

- Đếm người trong vùng camera.
- Nhận dạng người có đội mũ bảo hộ / mặc áo phản quang hay không.
- Phát hiện hành vi đi cầu thang không an toàn:
  - Đi sai làn.
  - Không vịn tay vịn.
  - Vịn sai bên.
  - Mang vác vật khi lên/xuống cầu thang.
  - Đi lùi.
  - Đứng yên trên cầu thang.
- Cảnh báo khi người đi vào vùng cấm / vùng nguy hiểm.

Phần cứng dự kiến: Jetson Orin Nano + camera cố định.

## 2. Cấu trúc repo

### 2.1. Thư mục chính

| Đường dẫn | Vai trò |
| --- | --- |
| `stair_monitor/` | Lõi phân tích hành vi trên cầu thang. |
| `ppe_monitor_core/` | Lõi PPE monitor toàn khung hình. |
| `ppe_monitor_ROI/` | PPE monitor chỉ hoạt động trong ROI. |
| `Dataset/` | Dataset YOLO dùng để train / validate model PPE. |
| `runs/` | Kết quả train / predict từ Ultralytics YOLO. |
| `video/` | Video đầu vào và video output cho các demo. |
| `venv/` | Virtual environment cục bộ. Không nên push lên git. |

### 2.2. File chạy chính

| File | Vai trò |
| --- | --- |
| `main.py` | Backend FastAPI + WebSocket để stream RTSP và nhận lệnh bắt đầu / dừng ghi video. |
| `viewer.html` | Giao diện web xem luồng camera qua WebSocket. |
| `test-cauthang.py` | Entrypoint chạy demo giám sát hành vi cầu thang. |
| `ppe.py` | Entrypoint chạy PPE monitor theo ROI. |
| `ppe_monitor.py` | Entrypoint chạy PPE monitor toàn khung hình. |
| `danger_zone_monitor.py` | Demo cảnh báo người bước vào vùng nguy hiểm. |

### 2.3. File hỗ trợ / tiện ích

| File | Vai trò |
| --- | --- |
| `draw_camera_points.py` | Vẽ lại `camera_config.json` lên ảnh để kiểm tra ROI, tay vịn, bậc thang, center line. |
| `veline.py` | Công cụ click trực tiếp lên ảnh để tạo / sửa file cấu hình không gian camera. |
| `trainyolo.py` | Script train model PPE bằng Ultralytics YOLO. |
| `yolotest.py` | Script test / predict model PPE trên video. |
| `pose-test.py` | Script test YOLO pose tracking trên video. |
| `tachframe.py` | Tách frame từ video ra ảnh để làm dataset hoặc kiểm tra dữ liệu. |
| `camera_config.json` | Cấu hình không gian chính cho bài toán cầu thang. |
| `camera_config2.json` | Một cấu hình camera phụ / thử nghiệm. |

## 3. Chi tiết từng module

### 3.1. `stair_monitor/`

Đây là module lớn nhất cho bài toán giám sát hành vi trên cầu thang.

| File | Vai trò |
| --- | --- |
| `settings.py` | Toàn bộ đường dẫn input/output, cờ debug, ngưỡng hình học và ngưỡng xác nhận hành vi. |
| `analyzer.py` | Lớp `BehaviorAnalyzer`, đóng vai trò điều phối toàn bộ phân tích hành vi. |
| `geometry.py` | Các hàm hình học cơ bản: góc tay, hướng cơ thể, quan hệ vị trí. |
| `handrail_analysis.py` | Suy luận trạng thái vịn tay vịn: đúng bên, sai bên, không vịn. |
| `carry.py` | Luật phát hiện tư thế mang vác. |
| `carry_analysis.py` | Ghép logic mang vác theo lịch sử nhiều frame. |
| `behavior_history.py` | Lưu lịch sử để xác nhận hướng đi, sai làn, đứng yên, đi lùi. |
| `result_builder.py` | Tổng hợp trạng thái cuối cùng thành nhãn, màu, cảnh báo hiển thị. |
| `rendering.py` | Vẽ bbox, nhãn, guide line, people count, debug info. |

### 3.2. `ppe_monitor_core/`

Đây là bản PPE monitor toàn khung hình, không lọc theo ROI.

| File | Vai trò |
| --- | --- |
| `config.py` | Đường dẫn video/model, ngưỡng confidence, ngưỡng overlap, ngưỡng temporal smoothing. |
| `geometry.py` | Tạo `head_bbox`, `torso_bbox`, tính overlap. |
| `detection.py` | Tách bbox PPE từ model, lọc false positive tóc đen, match mũ/áo với người. |
| `tracking.py` | Gán track bằng IoU, lưu history mũ/áo, ra trạng thái ổn định theo thời gian. |
| `rendering.py` | Vẽ bbox debug và bảng đếm số người theo trạng thái PPE. |
| `pipeline.py` | Luồng chạy chính: load model, đọc video, infer, tracking, render, lưu output. |
| `__init__.py` | Export `run_ppe_monitor`. |

### 3.3. `ppe_monitor_ROI/`

Đây là biến thể PPE monitor chỉ đánh giá người ở trong ROI.

| File | Vai trò |
| --- | --- |
| `config.py` | Cấu hình input/output, threshold và tọa độ ROI. |
| `roi.py` | Kiểm tra chân người có nằm trong ROI hay không; tạo `head_bbox` và `torso_bbox`. |
| `tracking.py` | Lưu history PPE theo từng người trong ROI. |
| `rendering.py` | Vẽ ROI, panel trạng thái và debug PPE. |
| `pipeline.py` | Pipeline chính cho PPE + ROI. |

### 3.4. `danger_zone_monitor.py`

`danger_zone_monitor.py` hiện là một module đơn file, không tách thành package riêng.
Chức năng của nó là phát hiện người đi vào vùng cấm bằng YOLO pose.

Các phần chính trong file:

| Thành phần | Vai trò |
| --- | --- |
| `roi_coords` | Tọa độ vùng nguy hiểm cần giám sát. |
| `foot_in_roi()` | Kiểm tra có ít nhất một bàn chân của người nằm trong ROI hay không. |
| `draw_person_debug()` | Vẽ bbox người và keypoint chân để debug trạng thái trong / ngoài ROI. |
| `draw_roi()` | Vẽ vùng ROI lên khung hình. |
| `draw_alert_text()` | Hiển thị dòng cảnh báo lớn khi có người vào vùng cấm. |
| `run_danger_zone_monitor()` | Pipeline chính: đọc video, chạy pose model, kiểm tra ROI, render cảnh báo, lưu output. |

Khi phát hiện người trong ROI:

- viền ROI đổi sang màu đỏ,
- khung hình có hiệu ứng nháy đỏ,
- xuất hiện cảnh báo `DANGER: PERSON IN RESTRICTED AREA`.

## 4. Luồng dữ liệu

### 4.1. Luồng RTSP WebSocket

`RTSP camera` -> `main.py` đọc frame bằng OpenCV -> encode JPEG base64 -> gửi qua WebSocket -> `viewer.html` hiển thị trên trình duyệt

Ngoài việc xem hình trực tiếp:

- `viewer.html` gửi `START_REC` -> `main.py` bắt đầu ghi file `record_YYYY-MM-DD_HH-MM-SS.avi`.
- `viewer.html` gửi `STOP_REC` -> `main.py` dừng ghi và đóng file video.

### 4.2. Luồng stair monitor

`video đầu vào` -> `YOLO pose tracking` -> `bbox + track_id + keypoints`

Sau đó:

- `rendering.get_feet_point()` lấy điểm đại diện cho vị trí chân.
- `rendering.get_motion_point()` lấy điểm đại diện cho chuyển động thân người.
- `BehaviorAnalyzer.analyze()` suy luận:
  - Người có nằm trong vùng cầu thang không.
  - Đang đi lên hay đi xuống.
  - Có đi sai làn không.
  - Có vịn tay vịn không.
  - Có vịn đúng bên không.
  - Có mang vác không.
  - Có đi lùi không.
  - Có đứng yên quá lâu không.
- `result_builder.py` gộp các tín hiệu trên thành trạng thái cuối.
- `rendering.py` vẽ overlay và người đếm.
- Kết quả được ghi ra video output.

### 4.3. Luồng PPE monitor toàn khung hình

`video đầu vào` -> `YOLO pose model` + `YOLO PPE model`

Sau đó:

- `parse_ppe_boxes()` lấy bbox mũ và áo.
- `get_head_bbox()` và `get_torso_bbox()` sinh vùng mục tiêu cho từng người.
- `match_ppe_item()` match mũ / áo vào từng người bằng center check + overlap.
- `tracking.py` lưu history ngắn hạn theo track.
- `update_stable_state()` dùng nhiều frame liên tiếp để giảm rung trạng thái.
- `rendering.draw_count_panel()` hiển thị số người đủ PPE / thiếu mũ / thiếu áo.
- Ghi video output.

### 4.4. Luồng PPE monitor theo ROI

`video đầu vào` -> `pose model` + `PPE model`

Khác biệt so với bản toàn khung hình:

- Chỉ xử lý người có chân nằm trong ROI.
- Có vẽ ROI và panel trạng thái tổng thể cho vùng đó.
- Dùng temporal smoothing để giảm flicker giữa các frame.

### 4.5. Luồng danger zone monitor

`video đầu vào` -> `YOLO pose model`

Sau đó:

- Lấy keypoint chân.
- Kiểm tra chân có nằm trong ROI cấm không.
- Nếu có người trong ROI:
  - ROI đổi màu đỏ.
  - Khung hình chớp đỏ.
  - Hiện cảnh báo `DANGER: PERSON IN RESTRICTED AREA`.

## 5. Cấu hình đầu vào cần chuẩn bị

### 5.1. Model

Repo hiện đang dùng các model sau:

- Pose model: `yolo11x-pose.pt`, `yolo11m-pose.pt` hoặc model pose YOLO tương đương.
- PPE detection model: `runs/detect/ppe-2class-6/weights/best.pt`.

Nếu thiếu file model, script sẽ lỗi khi load `YOLO(...)`.

### 5.2. Video và camera

Cần sửa đường dẫn input/output đúng với máy của bạn:

- `stair_monitor/settings.py`
- `ppe_monitor_core/config.py`
- `ppe_monitor_ROI/config.py`
- `danger_zone_monitor.py`
- `main.py` nếu dùng RTSP thật

### 5.3. Cấu hình không gian cầu thang

`camera_config.json` hiện chứa các key chính:

- `ROI`
- `HANDRAIL_LEFT_POLY`
- `HANDRAIL_RIGHT_POLY`
- `STEP_BOTTOM`
- `STEP_TOP`
- `CENTER_LINE`

Các key này được `stair_monitor` dùng để suy luận:

- vùng cầu thang cần theo dõi,
- tay vịn trái / phải,
- bậc dưới cùng / trên cùng,
- trục giữa để xác định làn và hướng đi.

## 6. Cài đặt môi trường

Khuyến nghị Python `3.10` đến `3.12`.

### 6.1. Tạo môi trường ảo

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 6.2. Cài thư viện

```powershell
pip install -r requirements.txt
```

Nếu bạn không dùng `requirements.txt`, có thể cài trực tiếp:

```powershell
pip install ultralytics opencv-python numpy shapely fastapi "uvicorn[standard]"
```

### 6.3. Ghi chú cho Jetson

- Trên Jetson, `cv2` và `torch` thường nên cài theo bản phù hợp với JetPack trước.
- Sau khi có `torch` và `cv2`, cài tiếp `ultralytics`, `fastapi`, `uvicorn`, `shapely`.
- Nếu `opencv-python` cài bằng `pip` không ổn trên Jetson, dùng OpenCV hệ thống là hợp lý.

## 7. Cách chạy

### 7.1. Chạy web viewer RTSP

Sửa `RTSP_URL` trong `main.py`, sau đó chạy:

```powershell
python main.py
```

Mở `viewer.html` trên trình duyệt.

Lưu ý:

- Trong `viewer.html`, ô WebSocket mặc định đang là `ws://100.64.0.17:8080/ws`.
- Nếu backend chạy trên máy khác, sửa đúng IP trước khi bấm connect.

### 7.2. Chạy demo hành vi cầu thang

Sửa `VIDEO_INPUT_PATH`, `VIDEO_OUTPUT_PATH`, `CAMERA_CONFIG_PATH` trong `stair_monitor/settings.py`, sau đó chạy:

```powershell
python test-cauthang.py
```

### 7.3. Chạy PPE monitor theo ROI

Sửa `VIDEO_INPUT_PATH`, `VIDEO_OUTPUT_PATH`, `ROI_COORDS` trong `ppe_monitor_ROI/config.py`, sau đó chạy:

```powershell
python ppe.py
```

### 7.4. Chạy PPE monitor toàn khung hình

Sửa `VIDEO_INPUT_PATH`, `VIDEO_OUTPUT_PATH` trong `ppe_monitor_core/config.py`, sau đó chạy:

```powershell
python ppe_monitor.py
```

### 7.5. Chạy danger zone monitor

Sửa ROI và đường dẫn video ngay trong `danger_zone_monitor.py`, sau đó chạy:

```powershell
python danger_zone_monitor.py
```

### 7.6. Vẽ lại cấu hình camera lên ảnh

```powershell
python draw_camera_points.py --image snapshot_2026-05-27T01-39-57.jpg --config camera_config.json --show
```

### 7.7. Tạo / chỉnh cấu hình camera bằng thao tác click

```powershell
python veline.py
```

### 7.8. Train model PPE

Sửa đường dẫn model / dataset trong `trainyolo.py`, sau đó chạy:

```powershell
python trainyolo.py
```

### 7.9. Test nhanh model

```powershell
python yolotest.py
python pose-test.py
```

## 8. Output được tạo ở đâu

| Chức năng | Output mặc định |
| --- | --- |
| WebSocket recorder | File `record_*.avi` ở thư mục gốc repo. |
| Stair monitor | Video trong `video/stair_demo/`. |
| PPE monitor | Video trong `video/ppe_demo/`. |
| Danger zone monitor | `video/roi_warning_output.mp4`. |
| YOLO train / predict | Thư mục `runs/`. |
| Tách frame | Thư mục output do `tachframe.py` cấu hình, hiện tại là `val2/`. |

## 9. Những file nên sửa đầu tiên khi chạy trên máy mới

- `main.py`
- `viewer.html`
- `stair_monitor/settings.py`
- `ppe_monitor_core/config.py`
- `ppe_monitor_ROI/config.py`
- `danger_zone_monitor.py`
- `camera_config.json`

## 10. Gợi ý khi push lên git

Không nên push các file sinh ra trong quá trình chạy nếu không cần thiết:

- `venv/`
- `runs/`
- video record output lớn
- model weight lớn nếu repo không chủ đích lưu model

Hiện `.gitignore` đã bỏ qua một số thư mục sinh tự động như `runs/`, `venv/`, `__pycache__/`, `video/`.
