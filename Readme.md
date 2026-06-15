# HƯỚNG DẪN SỬ DỤNG

> **Phạm vi tài liệu:** Hướng dẫn này áp dụng cho bản Windows/demo ở thư mục gốc của repo. Không áp dụng cho `jetson_orin_nano/`.

## 1. Giới thiệu Tổng quan

Đây là hệ thống Demo An toàn lao động ứng dụng Computer Vision dựa trên YOLO, dùng để giám sát hành vi và phát hiện vi phạm từ video hoặc camera.

Các tính năng chính:

- Giám sát hành vi di chuyển trên cầu thang (`Stair Monitor`)
- Kiểm tra PPE theo vùng quan tâm (`PPE ROI`)
- Kiểm tra PPE trên toàn khung hình (`PPE Full-frame`)
- Cảnh báo người đi vào vùng nguy hiểm (`Danger Zone`)

## 2. Cấu trúc Thư mục & Chuẩn bị Tài nguyên

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

### 2.2. Tài nguyên local cần chuẩn bị

Chuẩn bị các tài nguyên sau trước khi chạy dự án:

| Tài nguyên | Dùng cho chức năng | Yêu cầu | Vị trí đặt khuyến nghị |
| --- | --- | --- | --- |
| `video/` | Tất cả các bài toán chạy video offline | Chứa video đầu vào `.avi` hoặc `.mp4` và thư mục lưu output | Nên tạo các nhánh như `video/raw_video/`, `video/stair_demo/`, `video/ppe_demo/` |
| Pose model `*.pt` | Stair Monitor, PPE Full-frame, PPE ROI, Danger Zone | File YOLO pose, ví dụ `yolo11x-pose.pt` hoặc `yolo11m-pose.pt` | Repo root hoặc đổi lại path trong code |
| PPE model `best.pt` | PPE ROI, PPE Full-frame | File weight đã train cho bài toán PPE | Có thể đặt tại `runs/detect/ppe-2class-6/weights/best.pt` để khớp path mặc định |
| Ảnh `.jpg` để cấu hình camera | `veline.py`, `draw_camera_points.py` | Ảnh snapshot từ camera cần hiệu chỉnh | Repo root hoặc đường dẫn bất kỳ nếu truyền bằng tham số |
| `main.py` | Web Viewer RTSP/WebSocket | File backend FastAPI/Uvicorn nếu muốn xem realtime qua `viewer.html` | Đặt tại thư mục gốc repo |

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

> **LƯU Ý RẤT QUAN TRỌNG VỀ PYTORCH + CUDA 12.1**
>
> Repo này đang dùng:
>
> - `torch==2.5.1+cu121`
> - `torchvision==0.20.1+cu121`
>
> Nếu bạn chạy bằng GPU NVIDIA, hãy bảo đảm PyTorch được cài đúng bản `CUDA 12.1`. Không dùng nhầm bản CPU nếu bạn cần tăng tốc GPU.
>
> Cách cài khuyến nghị:
>
> ```powershell
> pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```
>
> Kiểm tra lại sau khi cài:
>
> ```powershell
> python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
> ```
>
> Nếu kết quả `torch.cuda.is_available()` trả về `False`, hãy kiểm tra lại driver NVIDIA, CUDA runtime và phiên bản wheel PyTorch đang dùng.

## 4. Hướng dẫn Vận hành chi tiết

### 4.1. Giám sát cầu thang

**Mục đích:** Phát hiện người di chuyển trên cầu thang, xác định hướng `UP/DOWN`, phát hiện sai làn, không vịn tay, vịn sai bên, mang vác, đi lùi và đứng yên.

**Bước 1 - Cấu hình:**

Mở `stair_monitor/settings.py` và sửa các biến quan trọng sau:

- `VIDEO_INPUT_PATH`
- `VIDEO_OUTPUT_PATH`
- `CAMERA_CONFIG_PATH`
- `USE_CURRENT_CAMERA_ANGLE`

Mở thêm `test-cauthang.py` và kiểm tra đường dẫn pose model tại dòng khởi tạo:

- `YOLO("yolo11x-pose.pt")`

Nếu thay đổi camera hoặc góc quay, kiểm tra lại các file:

- `camera_config.json`
- `camera_config2.json`

**Bước 2 - Lệnh chạy:**

```powershell
python test-cauthang.py
```

### 4.2. Kiểm tra PPE theo ROI

**Mục đích:** Chỉ kiểm tra PPE cho người nằm trong một vùng ROI xác định trước.

**Bước 1 - Cấu hình:**

Mở `ppe_monitor_ROI/config.py` và sửa:

- `VIDEO_INPUT_PATH`
- `VIDEO_OUTPUT_PATH`
- `ROI_COORDS`

Mở `ppe_monitor_ROI/pipeline.py` và kiểm tra 2 weight model đang được nạp trực tiếp:

- `YOLO("yolo11m-pose.pt")`
- `YOLO("runs/detect/ppe-2class-6/weights/best.pt")`

Nếu bạn đặt model ở vị trí khác, hãy sửa lại đúng path tại file này.

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

**Bước 2 - Lệnh chạy:**

```powershell
python ppe_monitor.py
```

### 4.4. Cảnh báo vùng nguy hiểm

**Mục đích:** Phát hiện người đi vào vùng cấm và sinh cảnh báo trực tiếp trên video output.

**Bước 1 - Cấu hình:**

Mở `danger_zone_monitor.py` và sửa:

- `VIDEO_INPUT_PATH`
- `VIDEO_OUTPUT_PATH`
- `POSE_MODEL_PATH`
- `roi_coords`

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

**Bước 2 - Lệnh chạy:**

```powershell
python veline.py
```

#### 4.5.2. `draw_camera_points.py`

**Mục đích:** Vẽ các vùng và đường tham chiếu từ `camera_config.json` lên ảnh để kiểm tra ROI, handrail, step và center line.

**Bước 1 - Cấu hình:**

Chuẩn bị:

- 1 file ảnh đầu vào `.jpg`
- 1 file config camera, ví dụ `camera_config.json`

Bạn có thể truyền trực tiếp bằng tham số dòng lệnh:

- `--image`
- `--config`
- `--output`
- `--show`

**Bước 2 - Lệnh chạy:**

```powershell
python draw_camera_points.py --image path/to/frame.jpg --config camera_config.json --show
```

#### 4.5.3. `trainyolo.py`

**Mục đích:** Train model PPE bằng Ultralytics YOLO trên dataset trong repo.

**Bước 1 - Cấu hình:**

Mở `trainyolo.py` và kiểm tra:

- Model khởi tạo đầu vào, hiện tại đang là `runs/detect/ppe-2class-5/weights/last.pt`
- Dataset YAML, hiện tại đang là `Dataset/data6/data.yaml`

Nếu bạn không dùng đúng cấu trúc này, hãy sửa lại path model hoặc path dataset trước khi train.

**Bước 2 - Lệnh chạy:**

```powershell
python trainyolo.py
```

## 5. Hướng dẫn sử dụng Web Viewer (RTSP / WebSocket)

Mục này chỉ áp dụng khi bạn có file backend `main.py`.

### Bước 1. Cài thêm thư viện cho backend

```powershell
pip install fastapi "uvicorn[standard]"
```

### Bước 2. Cấu hình nguồn RTSP

Mở `main.py` và sửa:

- `RTSP_URL`

Ví dụ trong code hiện tại:

```python
RTSP_URL = "rtsp://admin:password@IP:554/cam/realmonitor?channel=1&subtype=0"
```

### Bước 3. Chạy backend WebSocket

```powershell
python main.py
```

Backend mặc định sẽ mở WebSocket tại:

```text
ws://0.0.0.0:8080/ws
```

### Bước 4. Mở Web Viewer

Mở file `viewer.html` bằng trình duyệt.

Trong giao diện:

- Nhập địa chỉ WebSocket, ví dụ `ws://<IP-máy-chạy-backend>:8080/ws`
- Nhấn kết nối để xem realtime
- Dùng các nút điều khiển trên giao diện nếu muốn gửi lệnh ghi hình `START_REC` / `STOP_REC`

Lưu ý:

- Trong `viewer.html`, giá trị mặc định hiện đang là `ws://100.64.0.17:8080/ws`
- Nếu IP máy backend khác, hãy nhập lại đúng địa chỉ trước khi kết nối

## 6. Xử lý lỗi thường gặp

| Lỗi | Dấu hiệu | Cách xử lý |
| --- | --- | --- |
| Không tìm thấy file model | Lỗi phát sinh tại `YOLO("...")` hoặc khi khởi tạo model | Kiểm tra file `.pt` đã tồn tại chưa, kiểm tra đúng tên file, và sửa lại path trong `test-cauthang.py`, `ppe_monitor_ROI/pipeline.py`, `ppe_monitor_core/config.py` hoặc `danger_zone_monitor.py` |
| `cv2` không đọc được video | `cv2.VideoCapture(...)` không mở được file, hoặc code báo không mở được video | Kiểm tra `VIDEO_INPUT_PATH`, bảo đảm file `.avi/.mp4` tồn tại, tạo sẵn thư mục `video/` nếu đang dùng path bên trong repo, và kiểm tra quyền đọc file |
| Mismatch CUDA / PyTorch | GPU không được nhận, `torch.cuda.is_available()` trả về `False`, hoặc lỗi khi chạy inference deep learning | Cài lại đúng `torch==2.5.1+cu121` và `torchvision==0.20.1+cu121`, kiểm tra driver NVIDIA, sau đó xác minh lại bằng lệnh `python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"` |
