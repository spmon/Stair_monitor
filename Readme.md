# HƯỚNG DẪN SỬ DỤNG

> Phạm vi tài liệu: hướng dẫn này áp dụng cho bản Windows/demo ở thư mục gốc của repo.

## 1. Giới thiệu Tổng quan

Đây là hệ thống Demo An toàn lao động ứng dụng Computer Vision sử dụng YOLO để giám sát hành vi, PPE và vùng nguy hiểm từ video/camera.

Các tính năng chính:

- `Stair Monitor`: giám sát hành vi di chuyển trên cầu thang
- `PPE ROI`: kiểm tra PPE trong vùng quan tâm
- `PPE Full-frame`: kiểm tra PPE trên toàn khung hình
- `Danger Zone`: cảnh báo người đi vào vùng nguy hiểm

## 2. Cấu trúc Thư mục & Chuẩn bị Tài nguyên Local

### 2.1. Sơ đồ cây source code

```text
vnaDemo/
├── Dataset/
│   ├── data.yaml
│   ├── data1/
│   ├── data2/
│   └── data6/
├── stair_monitor/
│   ├── analyzer.py
│   ├── behavior_history.py
│   ├── carry.py
│   ├── carry_analysis.py
│   ├── geometry.py
│   ├── handrail_analysis.py
│   ├── rendering.py
│   ├── result_builder.py
│   └── settings.py
├── ppe_monitor_core/
│   ├── config.py
│   ├── detection.py
│   ├── geometry.py
│   ├── pipeline.py
│   ├── rendering.py
│   └── tracking.py
├── ppe_monitor_ROI/
│   ├── config.py
│   ├── pipeline.py
│   ├── rendering.py
│   ├── roi.py
│   └── tracking.py
├── camera_config.json
├── camera_config2.json
├── danger_zone_monitor.py
├── draw_camera_points.py
├── ppe.py
├── ppe_monitor.py
├── test-cauthang.py
├── trainyolo.py
├── veline.py
├── viewer.html
└── requirements.txt
```

### 2.2. Tài nguyên Local cần chuẩn bị

Chuẩn bị các tài nguyên sau trước khi chạy dự án:

| Tài nguyên | Bắt buộc cho | Cách chuẩn bị | Vị trí đặt khuyến nghị |
| --- | --- | --- | --- |
| `video/` | Tất cả các bài toán chạy video offline | Tự tạo thư mục chứa video input và video output | Tạo tại thư mục gốc repo |
| Pose weights `*.pt` | Stair, PPE ROI, PPE Full-frame, Danger Zone | Chuẩn bị file YOLO pose như `yolo11x-pose.pt` hoặc `yolo11m-pose.pt` | Đặt ở thư mục gốc repo hoặc sửa lại path trong code |
| PPE weights `best.pt` | PPE ROI, PPE Full-frame | Chuẩn bị file weight PPE đã train | Có thể đặt tại `runs/detect/ppe-2class-6/weights/best.pt` để khớp path mặc định |
| `main.py` | Web Viewer RTSP/WebSocket | Chỉ cần nếu được bàn giao backend RTSP | Đặt ở thư mục gốc repo |

### 2.3. Hướng dẫn tải Video Test

Tạo cấu trúc thư mục local trước khi chạy:

#### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force -Path video, video\raw_video, video\stair_demo, video\ppe_demo
```

#### Linux

```bash
mkdir -p video/raw_video video/stair_demo video/ppe_demo
```

video raw từ Cloud:

- Link: `https://cloud.vnatechlab.com/apps/files/files/156235?dir=/Prj_Nguyen_Chi_Minh/video/raw_video`

Thực hiện theo các bước:

1. Mở link Cloud ở trên.
2. Tải các video test cần dùng về máy.
3. Lưu các video vào thư mục `video/raw_video/` trong repo.
4. Khi cấu hình các script, trỏ `VIDEO_INPUT_PATH` tới đúng file video vừa tải.

Ví dụ:

```text
video/raw_video/your_test_video.avi
```

## 3. Hướng dẫn Cài đặt Môi trường

### Bước 1. Clone repo

```bash
git clone <YOUR_REPO_URL>
cd vnaDemo
```

### Bước 2. Khởi tạo và kích hoạt môi trường ảo

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

#### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Bước 3. Cài đặt thư viện

```powershell
pip install -r requirements.txt
```

> **WARNING - PYTORCH + CUDA 12.1**
>
> Dự án này đang dùng:
>
> - `torch==2.5.1+cu121`
> - `torchvision==0.20.1+cu121`
>
> Nếu bạn chạy bằng GPU NVIDIA, hãy cài đúng bản CUDA 12.1 để tránh bị cài nhầm bản CPU và làm inference/train chạy rất chậm.
>
> Cách cài khuyến nghị:
>
> ```powershell
> pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```
>
> Kiểm tra lại GPU:
>
> ```powershell
> python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
> ```
>
> Kết quả mong muốn: `torch.cuda.is_available()` trả về `True`.

## 4. Hướng dẫn Vận hành chi tiết

### 4.1. Giám sát hành vi cầu thang

**Mục đích:** Phát hiện người trong vùng cầu thang, xác định hướng `UP/DOWN`, phát hiện sai làn, không vịn tay, vịn sai bên, mang vác, đi lùi và đứng yên.

**Bước 1 - Cấu hình:**

Mở `stair_monitor/settings.py` và sửa các biến sau:

- `VIDEO_INPUT_PATH`
- `VIDEO_OUTPUT_PATH`
- `CAMERA_CONFIG_PATH`
- `USE_CURRENT_CAMERA_ANGLE`

Gợi ý cấu hình:

- `VIDEO_INPUT_PATH` trỏ tới video vừa tải trong `video/raw_video/`
- `VIDEO_OUTPUT_PATH` trỏ tới file output trong `video/stair_demo/`

Mở thêm `test-cauthang.py` và kiểm tra đường dẫn model pose:

- `YOLO("yolo11x-pose.pt")`

Nếu file model không nằm ở thư mục gốc repo, hãy sửa lại path cho đúng.

**Bước 2 - Lệnh chạy:**

```powershell
python test-cauthang.py
```

### 4.2. Kiểm tra PPE theo ROI

**Mục đích:** Chỉ đánh giá PPE cho người nằm trong vùng ROI đã cấu hình sẵn.

**Bước 1 - Cấu hình:**

Mở `ppe_monitor_ROI/config.py` và sửa:

- `VIDEO_INPUT_PATH`
- `VIDEO_OUTPUT_PATH`
- `ROI_COORDS`

Gợi ý cấu hình:

- `VIDEO_INPUT_PATH` trỏ tới file trong `video/raw_video/`
- `VIDEO_OUTPUT_PATH` trỏ tới file trong `video/ppe_demo/`

Mở `ppe_monitor_ROI/pipeline.py` và kiểm tra 2 model đang được nạp trực tiếp:

- `YOLO("yolo11m-pose.pt")`
- `YOLO("runs/detect/ppe-2class-6/weights/best.pt")`

Nếu đặt model ở vị trí khác, sửa lại đúng path trong file này.

**Bước 2 - Lệnh chạy:**

```powershell
python ppe.py
```

### 4.3. Kiểm tra PPE toàn khung hình

**Mục đích:** Kiểm tra mũ bảo hộ và áo phản quang trên toàn bộ khung hình, không giới hạn theo ROI.

**Bước 1 - Cấu hình:**

Mở `ppe_monitor_core/config.py` và sửa:

- `VIDEO_INPUT_PATH`
- `VIDEO_OUTPUT_PATH`
- `POSE_MODEL_PATH`
- `PPE_MODEL_PATH`

Gợi ý cấu hình:

- `VIDEO_INPUT_PATH` trỏ tới file trong `video/raw_video/`
- `VIDEO_OUTPUT_PATH` trỏ tới file trong `video/ppe_demo/`
- `POSE_MODEL_PATH` trỏ tới file pose model `.pt`
- `PPE_MODEL_PATH` trỏ tới file PPE model `best.pt`

**Bước 2 - Lệnh chạy:**

```powershell
python ppe_monitor.py
```

### 4.4. Cảnh báo vùng nguy hiểm

**Mục đích:** Phát hiện người đi vào vùng cấm và hiển thị cảnh báo trực tiếp trên video output.

**Bước 1 - Cấu hình:**

Mở `danger_zone_monitor.py` và sửa:

- `VIDEO_INPUT_PATH`
- `VIDEO_OUTPUT_PATH`
- `POSE_MODEL_PATH`
- `roi_coords`

Gợi ý cấu hình:

- `VIDEO_INPUT_PATH` trỏ tới file trong `video/raw_video/`
- `VIDEO_OUTPUT_PATH` trỏ tới file output mong muốn
- `POSE_MODEL_PATH` trỏ tới file pose model `.pt`

**Bước 2 - Lệnh chạy:**

```powershell
python danger_zone_monitor.py
```

### 4.5. Bộ công cụ cấu hình Camera & Tool hỗ trợ

#### 4.5.1. `veline.py`

**Mục đích:** Tạo hoặc cập nhật file cấu hình camera bằng thao tác click trực tiếp trên ảnh.

**Bước 1 - Cấu hình:**

Mở `veline.py` và sửa:

- `IMAGE_PATH`
- `CONFIG_FILE`

Khuyến nghị:

- `IMAGE_PATH`: trỏ tới ảnh snapshot `.jpg`
- `CONFIG_FILE`: chọn `camera_config.json` hoặc `camera_config2.json`

**Bước 2 - Lệnh chạy:**

```powershell
python veline.py
```

#### 4.5.2. `draw_camera_points.py`

**Mục đích:** Vẽ ROI, handrail, step và các đường tham chiếu từ file config camera lên ảnh để kiểm tra nhanh cấu hình.

**Bước 1 - Cấu hình:**

Chuẩn bị:

- 1 ảnh đầu vào `.jpg`
- 1 file config camera, ví dụ `camera_config.json`

Bạn có thể truyền trực tiếp bằng tham số:

- `--image`
- `--config`
- `--output`
- `--show`

**Bước 2 - Lệnh chạy:**

```powershell
python draw_camera_points.py --image path/to/frame.jpg --config camera_config.json --show
```

#### 4.5.3. `trainyolo.py`

**Mục đích:** Train PPE model bằng Ultralytics YOLO trên dataset trong repo.

**Bước 1 - Cấu hình:**

Mở `trainyolo.py` và kiểm tra:

- Model đầu vào, hiện đang là `runs/detect/ppe-2class-5/weights/last.pt`
- Dataset YAML, hiện đang là `Dataset/data6/data.yaml`

Nếu bạn train từ đầu hoặc dùng weight khác, sửa lại path trước khi chạy.

**Bước 2 - Lệnh chạy:**

```powershell
python trainyolo.py
```

## 5. Hướng dẫn sử dụng Web Viewer (RTSP / WebSocket)

Mục này chỉ áp dụng khi bạn được bàn giao file backend `main.py`.

### Bước 1. Cài thêm thư viện cho backend

```powershell
pip install fastapi "uvicorn[standard]"
```

### Bước 2. Cấu hình nguồn RTSP

Mở `main.py` và sửa:

- `RTSP_URL`

Ví dụ:

```python
RTSP_URL = "rtsp://admin:password@IP:554/cam/realmonitor?channel=1&subtype=0"
```

### Bước 3. Khởi động Uvicorn backend

Chạy:

```powershell
python main.py
```

Backend mặc định mở WebSocket tại:

```text
ws://0.0.0.0:8080/ws
```

### Bước 4. Mở Web Viewer

Mở file `viewer.html` bằng trình duyệt.

Trong giao diện:

1. Nhập địa chỉ WebSocket, ví dụ `ws://<IP-máy-backend>:8080/ws`
2. Nhấn kết nối
3. Theo dõi video realtime
4. Nếu cần, dùng các nút trên giao diện để gửi lệnh ghi hình `START_REC` / `STOP_REC`

Lưu ý:

- Giá trị mặc định trong `viewer.html` hiện là `ws://100.64.0.17:8080/ws`
- Nếu IP backend khác, hãy nhập lại đúng địa chỉ trước khi kết nối

## 6. Xử lý lỗi thường gặp

| Lỗi | Dấu hiệu | Cách xử lý |
| --- | --- | --- |
| Không tìm thấy file model | Lỗi khi gọi `YOLO("...")` hoặc lúc khởi tạo model | Kiểm tra đã chuẩn bị đủ file `*.pt` chưa, đặt đúng vị trí chưa, và sửa lại path trong `test-cauthang.py`, `ppe_monitor_ROI/pipeline.py`, `ppe_monitor_core/config.py` hoặc `danger_zone_monitor.py` |
| OpenCV không đọc được video | `cv2.VideoCapture(...)` không mở được file hoặc báo lỗi không mở được video | Kiểm tra đã tạo `video/` và `video/raw_video/` chưa, đã tải video từ Cloud về chưa, và `VIDEO_INPUT_PATH` có đang trỏ đúng file không |
| Mismatch CUDA / PyTorch | Chạy rất chậm, không dùng GPU, `torch.cuda.is_available()` trả về `False` | Cài lại đúng `torch==2.5.1+cu121` và `torchvision==0.20.1+cu121`, kiểm tra driver NVIDIA, rồi xác minh lại bằng lệnh kiểm tra CUDA trong phần cài đặt |
