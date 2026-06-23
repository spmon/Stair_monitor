from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from loguru import logger

from stair_monitor.common.types import (
    CameraConfigDict,
    ColorBGR,
    DebugFocusMode,
    StepLineJson,
)

# File nay gom toan bo SETTINGS cua ban Windows/demo.
# FLOW: dataclass group config -> SETTINGS singleton -> test-cauthang/analyzer/rules/rendering cung doc chung.
# WHY: Tach setting theo nhom giup lead mo dung khu vuc can giai thich: input, lane, handrail, carry, standing, perf.

def should_show_debug_focus(
    mode: DebugFocusMode,
    target: DebugFocusMode,
) -> bool:
    """Cho biet focus mode hien tai co cho phep ve nhom debug `target` hay khong."""
    if mode == "all":
        return target != "none"
    if mode == "none":
        return False
    return mode == target


def is_specific_debug_focus_mode(mode: DebugFocusMode) -> bool:
    return mode not in ("all", "none")


@dataclass(frozen=True)
class VideoConfig:
    """Nhom setting dau vao/dau ra video cua demo Windows."""

    # Video dau vao cua ban Windows/demo.
    # Co the la file path, RTSP URL hoac sentinel "RTSP_TUNNEL" de test-cauthang.py map sang URL localhost.
    input_path: str = field(
        default_factory=lambda: "RTSP_TUNNEL"
    )
    # Video output sau khi da ve overlay.
    output_path: str = field(
        default_factory=lambda: str(Path("video") / "stair_demo" / "demo50.mp4")
    )
    # JSON chua line/polygon ROI, lane va handrail.
    camera_config_path: str = field(default_factory=lambda: str(Path("camera_config5.json")))
    # Luu video output sau khi ve overlay.
    save_output_video: bool = False
    # Luu snapshot input/final de debug model input.
    save_model_input_debug: bool = False
    # Moi N frame moi luu 1 snapshot debug.
    save_model_input_debug_every: int = 60
    # Bat cua so live stream tren Windows/demo.
    show_live_window: bool = True
    # Ti le resize overlay chi de hien thi live window.
    live_window_scale: float = 0.5
    # Bat logger su kien violation ra terminal + CSV.
    alert_log_enabled: bool = True
    # Thu muc chua file CSV alert cho Windows/demo.
    alert_log_dir: str = "logs"
    # Cooldown log cung mot person/warning tinh theo giay.
    alert_log_cooldown_seconds: float = 3.0


@dataclass(frozen=True)
class DemoOverlayConfig:
    """Nhom setting giao dien demo va debug overlay."""

    # Bat giao dien demo gon hoac giao dien debug day du.
    demo_mode: bool = True
    # Flag tong de bat/tat thong tin debug phuc vu quan sat demo.
    enable_debug_overlay: bool = False
    # Bat debug text chi tiet cho tung nguoi.
    enable_verbose_person_debug: bool = False
    # Bat ve skeleton tay/than de quan sat pose.
    enable_skeleton_draw: bool = False
    # Bat ve text tieng Viet qua PIL/font Windows.
    enable_vietnamese_text: bool = True
    # Neu an toan thi co hien "An Toan" hay khong.
    draw_safe_status: bool = False
    # Demo badge canh bao se duoc giu them N giay.
    demo_alert_hold_seconds: float = 1.0
    # Demo live chi hien frame goc + badge loi lon, khong ve bbox/keypoint/ROI/debug.
    demo_alert_only_display: bool = True
    # Chon nhom debug can tap trung. "all" chi co tac dung khi enable_debug_overlay bat.
    debug_focus_mode: DebugFocusMode = "handrail"
    # Legacy compatibility flag; neu bat thi runtime map sang focus mode handrail.
    show_handrail_debug_only: bool = False
    # Ve line tu wrist toi diem gan nhat tren handrail khi debug handrail.
    show_handrail_distance_lines: bool = True
    # Ve text khoang cach ngan gon cho wrist khi debug handrail.
    show_handrail_distance_text: bool = True

    @property
    def draw_debug(self) -> bool:
        return (
            self.enable_debug_overlay
            or self.focus_debug_only
        )

    @property
    def draw_debug_detail(self) -> bool:
        return self.enable_verbose_person_debug

    @property
    def draw_skeleton(self) -> bool:
        return self.enable_skeleton_draw

    @property
    def draw_keypoints(self) -> bool:
        return self.draw_debug

    @property
    def handrail_debug_only(self) -> bool:
        return self.effective_debug_focus_mode == "handrail"

    @property
    def effective_debug_focus_mode(self) -> DebugFocusMode:
        if self.show_handrail_debug_only:
            return "handrail"
        return self.debug_focus_mode

    @property
    def focus_debug_only(self) -> bool:
        return is_specific_debug_focus_mode(self.effective_debug_focus_mode)

    @property
    def alert_only_display_active(self) -> bool:
        return self.demo_mode and self.demo_alert_only_display and not self.focus_debug_only


@dataclass(frozen=True)
class ViolationDisplayConfig:
    """Nhom setting nhan hien thi va mau sac cua warning/status."""

    # Mapping nhan noi bo -> nhan hien thi tren overlay/alert.
    display_names: dict[str, str] = field(
        default_factory=lambda: {
            "Sai Lan": "Sai làn",
            "Khong Vin": "Không vịn tay",
            "Vin Sai Ben": "Vịn sai bên",
            "Mang Vac": "Mang đồ",
            "Di Lui": "Đi lùi",
            "Dung Yen": "Không di chuyển",
            "Buoc 2 Bac": "Bước 2 Bậc",
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
            "Buoc 2 Bac",
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
    """Nhom setting profile camera anh huong mapping direction/lane/backward."""

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
    """Nhom setting history va nguong suy ra direction."""

    # True/False dao quy uoc dy -> UP/DOWN cho profile camera cu.
    sign_normal: bool = False
    # So frame history motion giu lai de tinh dy.
    history_len: int = 8
    # Toi thieu bao nhieu frame moi duoc ket luan huong di.
    min_frames: int = 7
    # Bien do pixel toi thieu tren truc y de khong con la IDLE.
    pixel_threshold: int = 20


@dataclass(frozen=True)
class LaneConfig:
    """Nhom setting history sai lan va grace frame khi mat feet."""

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
    """Nhom setting cua so signed-distance va history cho logic vin tay."""

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
    """Nhom setting nguong tu the tay/torso va history cho logic Mang Vac."""

    # Nguong goc tay duoc xem la dang gap theo logic Mang Vac.
    carry_arm_angle_threshold: int = 150
    # Chieu dai history carry 2 tay.
    front_carry_history_len: int = 15
    # So hit toi thieu de xac nhan carry 2 tay.
    front_carry_min_hits: int = 12
    # Chieu dai history carry 1 tay ro rang.
    front_carry_one_arm_history_len: int = 18
    # So hit toi thieu de xac nhan carry 1 tay.
    front_carry_one_arm_min_hits: int = 18
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
    """Nhom setting history va do tin cay cho logic Di Lui."""

    # Chieu dai history Di Lui.
    history_len: int = 15
    # So hit backward toi thieu de xac nhan.
    min_hits: int = 12
    # So bang chung body-facing toi thieu de xem la hop le.
    min_valid_evidence: int = 2


@dataclass(frozen=True)
class StandingConfig:
    """Nhom setting history motion cho logic Dung Yen."""

    # Chieu dai history motion cho Dung Yen.
    still_history_len: int = 18
    # So hit standing toi thieu de xac nhan.
    still_min_hits: int = 14
    # Bien do dao dong p_motion toi da de coi la dung yen.
    still_pixel_threshold: int = 25


@dataclass(frozen=True)
class VirtualFeetConfig:
    """Nhom setting suy virtual feet khi mat ankle."""

    # Scale trung tinh cho virtual feet shoulder + hip.
    shoulder_hip_scale: float = 0.9


@dataclass(frozen=True)
class PersonIdentityConfig:
    """Nhom setting stable identity/relink/enter-exit cho demo Windows."""

    # Bat lop person_uid on dinh de giam reset history khi YOLO doi track_id.
    enable_stable_identity: bool = True
    # So frame trusted feet can nam trong ROI de promote candidate thanh person_id.
    entry_confirm_frames: int = 2
    # Bat duong promote an toan cho nguoi bi che khuat luc vao ROI.
    allow_occluded_entry_promotion: bool = True
    # Candidate inside khong co outside proof phai du tuoi frame moi duoc promote.
    occluded_entry_min_age_frames: int = 10
    # Candidate inside khong co outside proof phai on dinh trong ROI du so frame nay.
    occluded_entry_min_inside_frames: int = 8
    # Candidate inside khong co outside proof phai co it nhat so frame motion hop ly nay.
    occluded_entry_min_motion_frames: int = 4
    # Block occluded promote neu van co LOST_INSIDE ghost hop ly o gan de uu tien relink.
    occluded_entry_block_if_any_lost_inside_ghost: bool = True
    # Block occluded promote neu candidate con overlap/qua gan voi active person khac.
    occluded_entry_block_if_near_active_person: bool = True
    # Nguong IoU de xem candidate con dang dinh vao active person khac.
    occluded_entry_near_active_iou_threshold: float = 0.15
    # Nguong khoang cach de xem candidate con qua gan active person khac.
    occluded_entry_near_active_distance_px: float = 80.0
    # Chi cho occluded promote khi feet source la trusted feet that/safe.
    occluded_entry_require_trusted_feet: bool = True
    # So frame trusted feet can nam ngoai ROI de xac nhan EXIT.
    exit_confirm_frames: int = 2
    # Candidate ngoai ROI duoc giu toi da bao nhieu frame truoc khi bo.
    candidate_timeout_frames: int = 45
    # So frame toi da cho phep session o LOST truoc khi xem xet complete.
    max_lost_frames: int = 30
    # So frame toi da de ghost con duoc xem xet relink an toan.
    relink_max_lost_frames: int = 30
    # So frame ghost duoc giu them neu mat giua cau thang va chua exit hop le.
    lost_inside_extra_frames: int = 45
    # Nguong khoang cach toi da de relink detection moi vao ghost session cu.
    relink_max_distance_px: int = 120
    # Do lech ty le bbox toi da cho relink.
    relink_max_bbox_size_ratio_diff: float = 0.5
    # Score tong hop toi da de chap nhan relink; score cang thap cang tot.
    relink_score_threshold: float = 1.0
    # Muc chenh toi thieu giua best va second-best de xem match la ro rang.
    relink_ambiguity_margin: float = 0.35
    # So frame can giu temp candidate truoc khi commit relink.
    relink_min_confirm_frames: int = 3
    # Khoang cach toi thieu tren truc cau thang de xet dao thu tu la conflict.
    relink_order_min_separation_px: int = 24
    # Margin top/bottom cua vung cau thang de xem la exit hop le.
    exit_zone_margin_px: int = 100
    # Bat log cac event identity quan trong.
    log_events: bool = True


@dataclass(frozen=True)
class StepBandConfig:
    """Nhom setting map ankle vao step band de debug va ho tro Buoc 2 Bac."""

    # Margin an toan quanh ranh gioi STEP_BAND khi map ankle vao step_index.
    boundary_margin_px: int = 2
    # Cho phep point hoi lech ngoai polygon van bam vao band gan nhat de debug on dinh hon.
    outside_tolerance_px: int = 20
    # Bat hien thi reason raw step debug tren overlay.
    debug_show_reason: bool = True
    # Bat ve outline nhe cua STEP_BANDS khi debug overlay dang bat.
    debug_show_bands: bool = True


@dataclass(frozen=True)
class TwoStepSkipConfig:
    """Nhom setting canh bao Buoc 2 Bac."""

    # Bat/tat logic canh bao "Buoc 2 Bac".
    enabled: bool = True
    # Hai ankle lech it nhat bao nhieu bac moi xem la vi pham.
    min_step_gap: int = 2
    # So frame lien tiep can giu gap >= nguong de xac nhan.
    confirm_frames: int = 1
    # Bat bo loc monotonic theo direction cho step cua tung chan.
    two_step_use_monotonic_filter: bool = True
    # True neu step_index tang dan theo chieu di len.
    two_step_step_index_increases_when_up: bool = True
    # So frame duoc phep giu last_valid_step khi raw step bi reject/mat tam thoi.
    two_step_hold_last_valid_step_frames: int = 3
    # Cho phep step giu nguyen cung 1 bac ma van hop le theo filter.
    two_step_allow_same_step: bool = True
    # Nguong confidence toi thieu cua moi ankle that de tham gia check.
    ankle_conf_threshold: float = 0.5
    # So raw step frame gan nhat duoc giu lai de lam muot planted state.
    foot_step_history_window: int = 3
    # So frame toi thieu de xem 1 step candidate da du on dinh.
    foot_planted_confirm_frames: int = 1
    # Nguong speed toi da de 1 ankle duoc xem la dang dat on dinh tren bac.
    foot_planted_max_speed_px_per_frame: float = 30.0
    # Cho phep tat speed check neu video 15 FPS khien planted bi miss qua nhieu.
    foot_planted_use_speed_check: bool = False
    # Bat/tat offset theo direction cho diem map ankle vao STEP_BAND.
    use_directional_ankle_offset: bool = True
    # Offset mac dinh khi direction la UP.
    up_ankle_step_offset_x_px: int = 0
    up_ankle_step_offset_y_px: int = 20
    # Offset mac dinh khi direction la DOWN.
    down_ankle_step_offset_x_px: int = 0
    down_ankle_step_offset_y_px: int = 0
    # Offset fallback khi direction chua xac dinh hoac khong co last valid.
    unknown_ankle_step_offset_x_px: int = 0
    unknown_ankle_step_offset_y_px: int = 0


@dataclass(frozen=True)
class PerformanceConfig:
    """Nhom setting log hieu nang cua pipeline demo."""

    # Bat log thong ke thoi gian tung block lon.
    enable_perf_log: bool = False
    # So frame moi lan in thong ke perf.
    perf_log_interval: int = 300


@dataclass(frozen=True)
class AppSettings:
    """Root settings gom cac nhom config lon cua toan bo demo."""

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
    virtual_feet: VirtualFeetConfig = field(default_factory=VirtualFeetConfig)
    identity: PersonIdentityConfig = field(default_factory=PersonIdentityConfig)
    step_band: StepBandConfig = field(default_factory=StepBandConfig)
    two_step_skip: TwoStepSkipConfig = field(default_factory=TwoStepSkipConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


SETTINGS = AppSettings()


def _normalize_step_line_point(raw_point: object) -> list[int] | None:
    if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
        return None

    x_value, y_value = raw_point
    if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
        return None
    return [int(x_value), int(y_value)]


def _normalize_step_line_id(raw_id: object, fallback_id: int) -> int:
    if raw_id is None:
        return fallback_id
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, float) and raw_id.is_integer():
        return int(raw_id)
    return fallback_id


def _normalize_step_line_entry(
    raw_step_line: object,
    fallback_id: int,
) -> StepLineJson | None:
    if not isinstance(raw_step_line, dict):
        return None

    point_1 = _normalize_step_line_point(raw_step_line.get("p1"))
    point_2 = _normalize_step_line_point(raw_step_line.get("p2"))
    if point_1 is None or point_2 is None:
        return None

    return {
        "id": _normalize_step_line_id(raw_step_line.get("id"), fallback_id),
        "p1": point_1,
        "p2": point_2,
    }


def _normalize_step_lines(raw_step_lines: object) -> list[StepLineJson]:
    if not isinstance(raw_step_lines, list):
        return []

    normalized_lines: list[StepLineJson] = []
    for fallback_id, raw_step_line in enumerate(raw_step_lines):
        normalized_step_line = _normalize_step_line_entry(
            raw_step_line,
            fallback_id,
        )
        if normalized_step_line is None:
            continue
        normalized_lines.append(normalized_step_line)

    normalized_lines.sort(key=lambda step_line: step_line["id"])
    return normalized_lines


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
        logger.error("Khong tim thay file camera_config.json!")
        raise SystemExit(1)

    if not isinstance(raw_config, dict):
        logger.error("camera_config.json khong dung dinh dang dict!")
        raise SystemExit(1)

    raw_config["STEP_LINES"] = _normalize_step_lines(raw_config.get("STEP_LINES"))
    return cast(CameraConfigDict, raw_config)
