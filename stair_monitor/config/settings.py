from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from stair_monitor.common.types import CameraConfigDict, ColorBGR


@dataclass(frozen=True)
class VideoConfig:
    # Video dau vao cua ban Windows/demo.
    input_path: str = field(
        default_factory=lambda: str(
            Path("video") / "raw_video" / "record_2026-06-11_16-23-21.avi"
        )
    )
    # Video output sau khi da ve overlay.
    output_path: str = field(
        default_factory=lambda: str(Path("video") / "stair_demo" / "demo40.mp4")
    )
    # JSON chua line/polygon ROI, lane va handrail.
    camera_config_path: str = field(default_factory=lambda: str(Path("camera_config.json")))
    # Luu video output sau khi ve overlay.
    save_output_video: bool = True
    # Luu snapshot input/final de debug model input.
    save_model_input_debug: bool = False
    # Moi N frame moi luu 1 snapshot debug.
    save_model_input_debug_every: int = 60


@dataclass(frozen=True)
class DemoOverlayConfig:
    # Bat giao dien demo gon hoac giao dien debug day du.
    demo_mode: bool = False
    # Flag tong de bat/tat thong tin debug phuc vu quan sat demo.
    enable_debug_overlay: bool = True
    # Bat debug text chi tiet cho tung nguoi.
    enable_verbose_person_debug: bool = False
    # Bat ve skeleton tay/than de quan sat pose.
    enable_skeleton_draw: bool = True
    # Bat ve text tieng Viet qua PIL/font Windows.
    enable_vietnamese_text: bool = True
    # Neu an toan thi co hien "An Toan" hay khong.
    draw_safe_status: bool = False
    # Demo badge canh bao se duoc giu them N giay.
    demo_alert_hold_seconds: float = 1.0

    @property
    def draw_debug(self) -> bool:
        return self.enable_debug_overlay

    @property
    def draw_debug_detail(self) -> bool:
        return self.enable_verbose_person_debug

    @property
    def draw_skeleton(self) -> bool:
        return self.enable_skeleton_draw

    @property
    def draw_keypoints(self) -> bool:
        return self.enable_debug_overlay


@dataclass(frozen=True)
class ViolationDisplayConfig:
    # Mapping nhan noi bo -> nhan hien thi tren overlay/alert.
    display_names: dict[str, str] = field(
        default_factory=lambda: {
            "Sai Lan": "Sai làn",
            "Khong Vin": "Không vịn tay",
            "Vin Sai Ben": "Vịn sai bên",
            "Mang Vac": "Mang đồ",
            "Di Lui": "Đi lùi",
            "Dung Yen": "Không di chuyển",
        }
    )
    # Thu tu uu tien khi hien thi alert vi pham.
    display_order: list[str] = field(
        default_factory=lambda: [
            "Khong Vin",
            "Vin Sai Ben",
            "Sai Lan",
            "Mang Vac",
            "Di Lui",
            "Dung Yen",
        ]
    )
    # Mau cho trang thai an toan.
    safe_color: ColorBGR = (0, 255, 0)
    # Mau cho trang thai vi pham.
    violation_color: ColorBGR = (0, 0, 255)
    # Mau cho trang thai chua du bang chung.
    unknown_color: ColorBGR = (0, 255, 255)
    # Mau cho nguoi dang ngoai vung cau thang.
    outside_color: ColorBGR = (0, 255, 255)
    # Mau cho giai doan dang thu thap direction/history.
    analyzing_color: ColorBGR = (255, 255, 0)

    @property
    def count_labels(self) -> list[str]:
        return list(self.display_order)


@dataclass(frozen=True)
class CameraProfileConfig:
    # True:
    # dung goc camera hien tai, giu nguyen logic cu
    #
    # False:
    # dung goc camera moi dat cao o chan cau thang
    # DOWN: y tang, lane dung LEFT, handrail dung LEFT
    # UP:   y giam, lane dung RIGHT, handrail dung RIGHT
    use_current_camera_angle: bool = True
    # side_value > 0 => LEFT, side_value < 0 => RIGHT
    # Dat = -1 neu muon dao nguoc quy uoc nay.
    lane_left_side_sign: int = 1


@dataclass(frozen=True)
class DirectionConfig:
    # True/False dao quy uoc dy -> UP/DOWN cho profile camera cu.
    sign_normal: bool = False
    # So frame history motion giu lai de tinh dy.
    history_len: int = 7
    # Toi thieu bao nhieu frame moi duoc ket luan huong di.
    min_frames: int = 5
    # Bien do pixel toi thieu tren truc y de khong con la IDLE.
    pixel_threshold: int = 20


@dataclass(frozen=True)
class LaneConfig:
    # True/False dao mapping side_value -> sai lan theo direction.
    sign_normal: bool = True
    # Chieu dai history sai lan theo track_id.
    history_len: int = 5
    # So hit sai lan toi thieu de xac nhan canh bao.
    min_wrong_hits: int = 4
    # So frame duoc giu lai lane cu khi mat chan.
    missing_feet_grace_frames: int = 15


@dataclass(frozen=True)
class HandrailConfig:
    # Signed distance toi da de wrist duoc xem la nam trong cua so lan can trai.
    left_max_distance: int = 120
    # Signed distance toi da de wrist duoc xem la nam trong cua so lan can phai.
    right_max_distance: int = 120
    # Gioi han khoang cach toi doan line khi check lan can dung ben.
    segment_max_distance: int = 100
    # Gioi han segment cho check lan can sai ben.
    wrong_side_segment_max_distance: int = 120
    # Chieu dai history hold final/raw theo track_id.
    hold_history_len: int = 15
    # So hit khong vin co bang chung toi thieu de xac nhan.
    hold_min_not_hold_evidence_hits: int = 13
    # So hit vin sai ben toi thieu de xac nhan.
    hold_min_wrong_side_hits: int = 10


@dataclass(frozen=True)
class CarryConfig:
    # Nguong goc tay duoc xem la dang gap theo logic Mang Vac.
    carry_arm_angle_threshold: int = 150
    # Chieu dai history carry 2 tay.
    front_carry_history_len: int = 15
    # So hit toi thieu de xac nhan carry 2 tay.
    front_carry_min_hits: int = 12
    # Chieu dai history carry 1 tay ro rang.
    front_carry_one_arm_history_len: int = 15
    # So hit toi thieu de xac nhan carry 1 tay.
    front_carry_one_arm_min_hits: int = 12
    # Nguong ngang dong theo shoulder width cho "hai co tay gan nhau".
    wrist_together_x_ratio: float = 1.2
    # Nguong doc dong theo torso height cho "hai co tay gan nhau".
    wrist_together_y_ratio: float = 0.75
    # Nguong mo rong ngang cua torso box dong.
    wrist_to_torso_x_ratio: float = 0.1
    # Nguong mo rong doc cua torso box dong.
    wrist_to_torso_y_ratio: float = 0.25
    # Margin them quanh torso cho vung check carry.
    carry_region_margin_ratio: float = 0.15
    # San pixel toi thieu cho nguong ngang wrist together.
    min_wrist_together_x_px: int = 20
    # San pixel toi thieu cho nguong doc wrist together.
    min_wrist_together_y_px: int = 20
    # San pixel toi thieu cho mo rong ngang torso box.
    min_wrist_to_torso_x_px: int = 10
    # San pixel toi thieu cho mo rong doc torso box.
    min_wrist_to_torso_y_px: int = 10
    # San pixel toi thieu cho carry region.
    min_carry_region_margin_px: int = 10
    # One-arm carry manh duoc check voi nguong goc nghiem hon.
    strong_one_arm_angle_threshold: int = 145


@dataclass(frozen=True)
class BackwardConfig:
    # Chieu dai history Di Lui.
    history_len: int = 15
    # So hit backward toi thieu de xac nhan.
    min_hits: int = 12
    # So bang chung body-facing toi thieu de xem la hop le.
    min_valid_evidence: int = 2


@dataclass(frozen=True)
class StandingConfig:
    # Chieu dai history motion cho Dung Yen.
    still_history_len: int = 18
    # So hit standing toi thieu de xac nhan.
    still_min_hits: int = 14
    # Bien do dao dong p_motion toi da de coi la dung yen.
    still_pixel_threshold: int = 25


@dataclass(frozen=True)
class PerformanceConfig:
    # Bat log thong ke thoi gian tung block lon.
    enable_perf_log: bool = True
    # So frame moi lan in thong ke perf.
    perf_log_interval: int = 30


@dataclass(frozen=True)
class AppSettings:
    video: VideoConfig = field(default_factory=VideoConfig)
    demo: DemoOverlayConfig = field(default_factory=DemoOverlayConfig)
    violation: ViolationDisplayConfig = field(default_factory=ViolationDisplayConfig)
    camera: CameraProfileConfig = field(default_factory=CameraProfileConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    lane: LaneConfig = field(default_factory=LaneConfig)
    handrail: HandrailConfig = field(default_factory=HandrailConfig)
    carry: CarryConfig = field(default_factory=CarryConfig)
    backward: BackwardConfig = field(default_factory=BackwardConfig)
    standing: StandingConfig = field(default_factory=StandingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


SETTINGS = AppSettings()


def load_camera_config(path: str | None = None) -> CameraConfigDict:
    """Doc cau hinh camera cho ban Windows/demo.

    Args:
        path: Duong dan toi file JSON chua line/polygon ROI, lane va handrail.

    Returns:
        CameraConfigDict: Noi dung camera config da duoc parse tu JSON.

    Notes:
        settings.py hien tai tap trung vao SETTINGS, mot AppSettings dataclass
        gom nhieu group config. Ham nay chi nap camera JSON, khong doi mapping
        direction/lane/handrail cua runtime.
    """
    config_path = Path(path or SETTINGS.video.camera_config_path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = json.load(file)
    except FileNotFoundError:
        print("Khong tim thay file camera_config.json!")
        raise SystemExit(1)

    if not isinstance(raw_config, dict):
        print("camera_config.json khong dung dinh dang dict!")
        raise SystemExit(1)

    return cast(CameraConfigDict, raw_config)
