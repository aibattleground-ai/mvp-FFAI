# app.py
# FormFoundry AI — A4 Marker (PnP/Homography) Calibrated Measurement MVP
# 안정화 패치 포함: ArUco(OpenCV) 호환, assets 상대경로, Streamlit image width 호환



from __future__ import annotations

import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import streamlit as st

# --- Optional / heavy deps (guarded) ---
try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# =========================
# Config
# =========================
st.set_page_config(
    page_title="FormFoundry AI — A4 Calibrated Measurement MVP",
    layout="wide",
)

A4_W_MM = 210.0
A4_H_MM = 297.0

# Marker sheet design assumptions (your PDF preview 기준)
MARKER_DICT_NAME = "DICT_4X4_50"
MARKER_IDS_REQUIRED = {0, 1, 2, 3}
MARKER_SIDE_MM = 45.0
MARGIN_MM = 10.0

DEFAULT_RATIOS = {"chest": 2.25, "waist": 2.15, "hip": 2.20}

GARMENT_PRIORS_CM_PER_SIDE = {
    "sleeveless": (0.2, 0.5),
    "short_sleeve": (0.2, 0.6),
    "long_sleeve": (0.3, 0.8),
    "hoodie": (0.6, 1.5),
    "coat": (1.2, 2.5),
    "puffer": (2.0, 4.0),
}
MARKER_PDF_PATH = Path(__file__).resolve().parent / "assets" / "FormFoundry_A4_MarkerSheet_v1_1.pdf"

# --- Preflight: required asset check ---
import streamlit as st

if not MARKER_PDF_PATH.exists():
    st.error(f"Marker PDF not found: {MARKER_PDF_PATH}")
    st.stop()

if not MARKER_PDF_PATH.is_file():
    st.error(f"Marker PDF path is not a file: {MARKER_PDF_PATH}")
    st.stop()

MATERIAL_ADJ_CM = {
    "cotton blend": -0.1,
    "thin cotton": -0.2,
    "knit": +0.3,
    "wool": +0.4,
    "leather": +0.6,
    "down": +0.8,
    "synthetic": +0.1,
    "unknown": 0.0,
}

# assets 위치: main/main/assets/*
ASSET_DIR = Path(__file__).resolve().parent / "assets"
MARKER_PDF = ASSET_DIR / "FormFoundry_A4_MarkerSheet_v1_1.pdf"
DEMO_IMAGE_CANDIDATES = [
    ASSET_DIR / "demo_front.jpg",
    ASSET_DIR / "demo_front.jpeg",
    ASSET_DIR / "demo_front.png",
]


# =========================
# Data classes
# =========================
@dataclass
class PoseLandmarks:
    xy: Dict[int, Tuple[float, float]]
    z: Dict[int, float]


@dataclass
class A4Calib:
    ok: bool
    H_img2mm: Optional[np.ndarray] = None
    H_img2px: Optional[np.ndarray] = None
    px_per_mm: Optional[float] = None
    a4_size_px: Optional[Tuple[int, int]] = None
    debug: Optional[dict] = None


@dataclass
class MeasurementResult:
    px_per_cm: float
    rotation_deg: float
    garment_class: str
    material: str
    offset_per_side_cm: float
    width_deduct_cm: float

    shoulder_width_cm: Optional[float]
    chest_circ_cm: Optional[float]
    waist_circ_cm: Optional[float]
    hip_circ_cm: Optional[float]
    arm_len_cm: Optional[float]
    leg_len_cm: Optional[float]

    debug: dict


# =========================
# Streamlit compatibility helpers
# =========================
def st_image_compat(img, caption: str = ""):
    """
    Streamlit 버전에 따라 st.image 파라미터가 달라서
    width='stretch' 지원 안 하는 환경에서도 안 죽게 보호.
    """
    try:
        st.image(img, caption=caption, width="stretch")
        return
    except TypeError:
        pass

    # 구버전 fallback
    try:
        st.image(img, caption=caption, use_container_width=True)  # deprecated지만 구버전 호환용
        return
    except TypeError:
        st.image(img, caption=caption)


# =========================
# Utility
# =========================
def _require_cv2():
    if cv2 is None:
        st.error("OpenCV(cv2) import failed. requirements에 opencv-contrib-python(-headless) 포함 여부를 확인하세요.")
        st.stop()
    if not hasattr(cv2, "aruco"):
        st.error("cv2.aruco가 없습니다. opencv-contrib-python(-headless)가 필요합니다.")
        st.stop()


def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _read_image(file) -> np.ndarray:
    data = file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image.")
    return bgr


def _apply_homography_pts(pts_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = pts_xy.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def _safe_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


# =========================
# ArUco detection (OpenCV 호환)
# =========================
def _aruco_dict_by_name(name: str):
    aruco = cv2.aruco
    if hasattr(aruco, name):
        return aruco.getPredefinedDictionary(getattr(aruco, name))
    raise ValueError(f"Unknown aruco dict name: {name}")


def _aruco_detect_markers(img_bgr: np.ndarray, dict_name: str):
    """
    OpenCV 버전에 따라:
    - 최신: cv2.aruco.ArucoDetector
    - 구버전: cv2.aruco.detectMarkers + DetectorParameters_create
    둘 다 지원.
    """
    aruco = cv2.aruco
    d = _aruco_dict_by_name(dict_name)

    # parameters
    if hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
    else:
        params = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(d, params)
        corners, ids, rej = detector.detectMarkers(img_bgr)
        return corners, ids, rej

    corners, ids, rej = aruco.detectMarkers(img_bgr, d, parameters=params)
    return corners, ids, rej


def _marker_centers_mm() -> Dict[int, Tuple[float, float]]:
    cx = MARGIN_MM + MARKER_SIDE_MM / 2.0
    cy = MARGIN_MM + MARKER_SIDE_MM / 2.0
    right_cx = A4_W_MM - cx
    bot_cy = A4_H_MM - cy
    return {
        0: (cx, cy),
        1: (right_cx, cy),
        2: (right_cx, bot_cy),
        3: (cx, bot_cy),
    }


def _marker_object_corners_mm(marker_id: int) -> np.ndarray:
    centers = _marker_centers_mm()
    cx, cy = centers[marker_id]
    h = MARKER_SIDE_MM / 2.0
    return np.array(
        [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]],
        dtype=np.float32,
    )


def detect_a4_calibration(img_bgr: np.ndarray, px_per_mm: float, dict_name: str) -> A4Calib:
    _require_cv2()

    corners, ids, _rej = _aruco_detect_markers(img_bgr, dict_name=dict_name)
    if ids is None or len(ids) == 0:
        return A4Calib(ok=False, debug={"reason": "no_markers"})

    ids_list = [int(i) for i in ids.flatten().tolist()]
    found = set(ids_list)
    missing = list(MARKER_IDS_REQUIRED - found)
    if missing:
        return A4Calib(ok=False, debug={"reason": "missing_markers", "missing": missing, "found": ids_list})

    img_pts, obj_pts = [], []
    for c, mid in zip(corners, ids.flatten()):
        mid = int(mid)
        if mid not in MARKER_IDS_REQUIRED:
            continue
        c2 = c.reshape(-1, 2).astype(np.float32)      # tl,tr,br,bl
        obj = _marker_object_corners_mm(mid)          # tl,tr,br,bl (mm)
        img_pts.append(c2)
        obj_pts.append(obj)

    img_pts = np.vstack(img_pts).astype(np.float32)
    obj_pts = np.vstack(obj_pts).astype(np.float32)

    H_img2mm, inliers = cv2.findHomography(img_pts, obj_pts, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if H_img2mm is None:
        return A4Calib(ok=False, debug={"reason": "homography_failed"})

    # mm -> px scale on A4 plane
    S = np.array([[px_per_mm, 0, 0], [0, px_per_mm, 0], [0, 0, 1]], dtype=np.float64)
    H_img2px = S @ H_img2mm

    a4_w_px = int(round(A4_W_MM * px_per_mm))
    a4_h_px = int(round(A4_H_MM * px_per_mm))

    return A4Calib(
        ok=True,
        H_img2mm=H_img2mm,
        H_img2px=H_img2px,
        px_per_mm=px_per_mm,
        a4_size_px=(a4_w_px, a4_h_px),
        debug={"markers_found": ids_list, "px_per_mm": px_per_mm, "a4_size_px": (a4_w_px, a4_h_px)},
    )


def warp_to_a4(img_bgr: np.ndarray, calib: A4Calib) -> np.ndarray:
    w_px, h_px = calib.a4_size_px
    return cv2.warpPerspective(img_bgr, calib.H_img2px, (w_px, h_px))


def warp_mask_to_a4(mask01: np.ndarray, calib: A4Calib) -> np.ndarray:
    w_px, h_px = calib.a4_size_px
    warped = cv2.warpPerspective(mask01.astype(np.uint8) * 255, calib.H_img2px, (w_px, h_px), flags=cv2.INTER_NEAREST)
    return (warped > 127).astype(np.uint8)


# =========================
# Pose
# =========================
@st.cache_resource
def _get_pose_model():
    if mp is None:
        return None
    Pose = mp.solutions.pose.Pose
    return Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )


def infer_pose_landmarks(img_bgr: np.ndarray, calib: Optional[A4Calib]) -> Optional[PoseLandmarks]:
    if mp is None:
        return None
    model = _get_pose_model()
    if model is None:
        return None

    img_rgb = _to_rgb(img_bgr)
    h, w = img_rgb.shape[:2]
    res = model.process(img_rgb)
    if not res.pose_landmarks:
        return None

    xy, z = {}, {}
    for idx, lm in enumerate(res.pose_landmarks.landmark):
        xy[idx] = (float(lm.x * w), float(lm.y * h))
        z[idx] = float(lm.z)

    if calib is not None and calib.ok:
        pts = np.array(list(xy.values()), dtype=np.float32)
        pts2 = _apply_homography_pts(pts, calib.H_img2px)
        xy = {k: (float(pts2[i, 0]), float(pts2[i, 1])) for i, k in enumerate(list(xy.keys()))}

    return PoseLandmarks(xy=xy, z=z)


def estimate_rotation_deg(pose: PoseLandmarks) -> float:
    if 11 not in pose.xy or 12 not in pose.xy:
        return 0.0
    zL = pose.z.get(11, 0.0)
    zR = pose.z.get(12, 0.0)
    xL, _ = pose.xy[11]
    xR, _ = pose.xy[12]
    shoulder_span = max(1e-6, abs(xR - xL))
    dz = (zR - zL)
    rot = math.degrees(math.atan2(dz, shoulder_span / 1000.0))
    return float(max(-30.0, min(30.0, rot)))


# =========================
# Segmentation (person mask)
# =========================
@st.cache_resource
def _get_yolo_seg_model():
    if YOLO is None:
        return None
    return YOLO("yolov8n-seg.pt")


def infer_person_mask_yolo(img_bgr: np.ndarray, conf: float = 0.25) -> Optional[np.ndarray]:
    if YOLO is None:
        return None
    model = _get_yolo_seg_model()
    if model is None:
        return None

    results = model.predict(img_bgr, conf=conf, iou=0.5, verbose=False)
    if not results:
        return None
    r = results[0]
    if r.masks is None or r.boxes is None:
        return None

    names = r.names
    cls = r.boxes.cls.detach().cpu().numpy().astype(int)
    confs = r.boxes.conf.detach().cpu().numpy()

    best_i, best_c = None, -1.0
    for i, (ci, cf) in enumerate(zip(cls, confs)):
        if names.get(int(ci), "") == "person" and float(cf) > best_c:
            best_c = float(cf)
            best_i = i
    if best_i is None:
        return None

    mask = r.masks.data[best_i].detach().cpu().numpy()
    return (mask > 0.5).astype(np.uint8)


# =========================
# Measurement helpers
# =========================
def row_edges_from_x(mask01: np.ndarray, y: int, x0: int, direction: int) -> Optional[int]:
    h, w = mask01.shape
    y = int(max(0, min(h - 1, y)))
    row = mask01[y]
    ones = np.where(row > 0)[0]
    if ones.size == 0:
        return None

    x0 = int(max(0, min(w - 1, x0)))
    if row[x0] == 0:
        x0 = int(ones[np.argmin(np.abs(ones - x0))])

    x = x0
    if direction < 0:
        while x > 0 and row[x] > 0:
            x -= 1
        return x + 1
    else:
        while x < w - 1 and row[x] > 0:
            x += 1
        return x - 1


def center_run_width(mask01: np.ndarray, y: int, center_x: int) -> Optional[Tuple[int, int, int]]:
    h, w = mask01.shape
    y = int(max(0, min(h - 1, y)))
    row = mask01[y]
    ones = np.where(row > 0)[0]
    if ones.size == 0:
        return None

    center_x = int(max(0, min(w - 1, center_x)))
    if row[center_x] == 0:
        center_x = int(ones[np.argmin(np.abs(ones - center_x))])

    left = row_edges_from_x(mask01, y, center_x, -1)
    right = row_edges_from_x(mask01, y, center_x, +1)
    if left is None or right is None or right <= left:
        return None
    width = right - left + 1
    return left, right, width


def estimate_offset_per_side_cm(garment_class: str, material: str, looseness: float, mode: str = "mid") -> float:
    lo, hi = GARMENT_PRIORS_CM_PER_SIDE.get(garment_class, (0.3, 0.8))
    base = {"min": lo, "mid": (lo + hi) / 2.0, "max": hi}.get(mode, (lo + hi) / 2.0)
    adj = MATERIAL_ADJ_CM.get(material, 0.0)

    loose_boost = 0.0
    if looseness >= 1.55:
        loose_boost = 0.6
    elif looseness >= 1.40:
        loose_boost = 0.35
    elif looseness >= 1.25:
        loose_boost = 0.15

    return float(max(0.0, base + adj + loose_boost))


def maybe_erode_mask(mask01: np.ndarray, offset_per_side_cm: float, px_per_cm: float, enabled: bool) -> np.ndarray:
    if not enabled:
        return mask01
    k = int(round(offset_per_side_cm * px_per_cm))
    if k <= 1:
        return mask01
    k = k if k % 2 == 1 else k + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    eroded = cv2.erode(mask01.astype(np.uint8), kernel, iterations=1)
    return (eroded > 0).astype(np.uint8)


def measure_lengths_from_pose(pose: PoseLandmarks, px_per_cm: float):
    def dist(a: int, b: int) -> Optional[float]:
        if a not in pose.xy or b not in pose.xy:
            return None
        ax, ay = pose.xy[a]
        bx, by = pose.xy[b]
        return float(math.hypot(ax - bx, ay - by))

    parts_arm = []
    for (s, e, w) in [(11, 13, 15), (12, 14, 16)]:
        d1, d2 = dist(s, e), dist(e, w)
        if d1 is not None and d2 is not None:
            parts_arm.append(d1 + d2)
    arm_px = float(np.mean(parts_arm)) if parts_arm else None

    parts_leg = []
    for (h, k, a) in [(23, 25, 27), (24, 26, 28)]:
        d1, d2 = dist(h, k), dist(k, a)
        if d1 is not None and d2 is not None:
            parts_leg.append(d1 + d2)
    leg_px = float(np.mean(parts_leg)) if parts_leg else None

    arm_cm = None if arm_px is None else arm_px / px_per_cm
    leg_cm = None if leg_px is None else leg_px / px_per_cm
    return arm_cm, leg_cm


def compute_measurements(
    mask01: np.ndarray,
    pose: Optional[PoseLandmarks],
    px_per_cm: float,
    garment_class: str,
    material: str,
    ratio_mode: str,
    offset_mode: str,
    erosion_enabled: bool,
) -> MeasurementResult:
    debug = {}
    rotation_deg = estimate_rotation_deg(pose) if pose else 0.0
    debug["rotation_deg"] = rotation_deg

    h, w = mask01.shape

    def get_xy(i: int):
        return pose.xy.get(i) if pose and i in pose.xy else None

    shoulderL, shoulderR = get_xy(11), get_xy(12)
    hipL, hipR = get_xy(23), get_xy(24)

    if shoulderL and shoulderR and hipL and hipR:
        y_sh = (shoulderL[1] + shoulderR[1]) / 2.0
        y_hp = (hipL[1] + hipR[1]) / 2.0
        x_center = (hipL[0] + hipR[0]) / 2.0
    else:
        ys, xs = np.where(mask01 > 0)
        if ys.size == 0:
            return MeasurementResult(
                px_per_cm=px_per_cm, rotation_deg=rotation_deg,
                garment_class=garment_class, material=material,
                offset_per_side_cm=0.0, width_deduct_cm=0.0,
                shoulder_width_cm=None, chest_circ_cm=None, waist_circ_cm=None, hip_circ_cm=None,
                arm_len_cm=None, leg_len_cm=None,
                debug={"reason": "empty_mask"},
            )
        y_sh = float(np.percentile(ys, 20))
        y_hp = float(np.percentile(ys, 60))
        x_center = float(np.median(xs))

    y_chest = y_sh + 0.28 * (y_hp - y_sh)
    y_waist = y_sh + 0.55 * (y_hp - y_sh)
    y_hip = y_sh + 0.82 * (y_hp - y_sh)
    debug["y_levels_px"] = {"shoulder": y_sh, "chest": y_chest, "waist": y_waist, "hip": y_hip}

    # Shoulder width: 관절 주변에서 바깥 경계까지 스캔
    shoulder_width_px = None
    if shoulderL and shoulderR:
        yS = _safe_int(y_sh, 0, h - 1)
        xLS = _safe_int(shoulderL[0], 0, w - 1)
        xRS = _safe_int(shoulderR[0], 0, w - 1)
        left_edge = row_edges_from_x(mask01, yS, xLS, -1)
        right_edge = row_edges_from_x(mask01, yS, xRS, +1)
        if left_edge is not None and right_edge is not None and right_edge > left_edge:
            shoulder_width_px = right_edge - left_edge + 1
            debug["shoulder_edges_px"] = {"y": yS, "left": left_edge, "right": right_edge}

    # looseness = clothed chest width / pose shoulder span (proxy)
    looseness = 1.25
    pose_shoulder_span_cm = None
    if shoulderL and shoulderR:
        pose_span_px = float(abs(shoulderR[0] - shoulderL[0]))
        pose_shoulder_span_cm = pose_span_px / px_per_cm

    chest_run = center_run_width(mask01, _safe_int(y_chest, 0, h - 1), _safe_int(x_center, 0, w - 1))
    if chest_run is not None and pose_shoulder_span_cm and pose_shoulder_span_cm > 1e-6:
        _, _, chest_w_px = chest_run
        chest_w_cm = chest_w_px / px_per_cm
        looseness = float(chest_w_cm / pose_shoulder_span_cm)
    debug["looseness"] = looseness
    debug["pose_shoulder_span_cm"] = pose_shoulder_span_cm

    offset_per_side_cm = estimate_offset_per_side_cm(garment_class, material, looseness, mode=offset_mode)
    width_deduct_cm = 2.0 * offset_per_side_cm

    mask_use = maybe_erode_mask(mask01, offset_per_side_cm, px_per_cm, enabled=erosion_enabled)

    def width_cm_at(yf: float) -> Optional[float]:
        y = _safe_int(yf, 0, h - 1)
        cx = _safe_int(x_center, 0, w - 1)
        run = center_run_width(mask_use, y, cx)
        if run is None:
            return None
        _, _, ww = run
        return float(ww / px_per_cm)

    chest_width_cm = width_cm_at(y_chest)
    waist_width_cm = width_cm_at(y_waist)
    hip_width_cm = width_cm_at(y_hip)
    debug["widths_cm_post_erosion"] = {
        "chest_width_cm": chest_width_cm,
        "waist_width_cm": waist_width_cm,
        "hip_width_cm": hip_width_cm,
    }

    shoulder_width_cm = None
    if shoulder_width_px is not None:
        shoulder_width_cm = float(shoulder_width_px / px_per_cm)

    def corrected_width(width_cm: Optional[float]) -> Optional[float]:
        if width_cm is None:
            return None
        if erosion_enabled:
            return float(max(0.0, width_cm))
        return float(max(0.0, width_cm - width_deduct_cm))

    chest_w_body = corrected_width(chest_width_cm)
    waist_w_body = corrected_width(waist_width_cm)
    hip_w_body = corrected_width(hip_width_cm)

    shoulder_w_body = None
    if shoulder_width_cm is not None:
        shoulder_w_body = float(max(0.0, shoulder_width_cm - (0.5 * width_deduct_cm)))

    ratios = DEFAULT_RATIOS.copy()

    rot_factor = 1.0 + (abs(rotation_deg) / 90.0) * 0.04
    loose_factor = 1.0 + max(0.0, looseness - 1.3) * 0.03

    if ratio_mode == "fixed+signals":
        ratios["chest"] *= rot_factor * loose_factor
        ratios["waist"] *= (1.0 + max(0.0, looseness - 1.3) * 0.02)
        ratios["hip"] *= (1.0 + max(0.0, looseness - 1.3) * 0.02)

    debug["ratios_used"] = ratios
    debug["rot_factor"] = rot_factor
    debug["loose_factor"] = loose_factor

    chest_circ = None if chest_w_body is None else float(chest_w_body * ratios["chest"])
    waist_circ = None if waist_w_body is None else float(waist_w_body * ratios["waist"])
    hip_circ = None if hip_w_body is None else float(hip_w_body * ratios["hip"])

    arm_len_cm, leg_len_cm = (None, None)
    if pose:
        arm_len_cm, leg_len_cm = measure_lengths_from_pose(pose, px_per_cm)

    return MeasurementResult(
        px_per_cm=px_per_cm,
        rotation_deg=rotation_deg,
        garment_class=garment_class,
        material=material,
        offset_per_side_cm=offset_per_side_cm,
        width_deduct_cm=width_deduct_cm,
        shoulder_width_cm=shoulder_w_body,
        chest_circ_cm=chest_circ,
        waist_circ_cm=waist_circ,
        hip_circ_cm=hip_circ,
        arm_len_cm=arm_len_cm,
        leg_len_cm=leg_len_cm,
        debug=debug,
    )


# =========================
# UI
# =========================
st.title("FormFoundry AI — A4 Calibrated Measurement MVP")

with st.sidebar:
    st.header("Mode")
    mode = st.radio("측정 모드", ["고정밀 (A4 마커 필수)", "대체 (A4 없이: 대략)"], index=0)

    st.header("A4 Calibration")
    px_per_mm = st.slider("A4 워프 해상도 (px/mm)", min_value=3.0, max_value=10.0, value=6.0, step=0.5)

    st.header("Segmentation")
    yolo_conf = st.slider("YOLO conf", 0.10, 0.80, 0.25, 0.05)

    st.header("Clothing Signals (MVP)")
    garment_class = st.selectbox(
        "Garment Class",
        ["short_sleeve", "long_sleeve", "hoodie", "coat", "puffer", "sleeveless"],
        index=0,
    )
    material = st.selectbox(
        "Material",
        ["cotton blend", "thin cotton", "knit", "wool", "leather", "down", "synthetic", "unknown"],
        index=0,
    )

    st.header("Volume-Offset")
    offset_mode = st.selectbox("두께 범위 선택", ["min", "mid", "max"], index=1)
    erosion_enabled = st.checkbox("마스크 erosion 적용(시각 데모용)", value=False)

    st.header("Circumference Ratio")
    ratio_mode = st.selectbox("ratio 모드", ["fixed", "fixed+signals"], index=1)

    st.header("Export")
    export_json = st.checkbox("결과 JSON 출력", value=True)

    # Marker sheet download (repo relative assets)
    if MARKER_PDF.exists():
        st.download_button(
            "A4 마커 시트 PDF 다운로드",
            data=MARKER_PDF.read_bytes(),
            file_name=MARKER_PDF.name,
            mime="application/pdf",
        )
    else:
        st.caption("assets 폴더에 마커 PDF가 없으면 다운로드 버튼이 숨겨집니다.")


colL, colR = st.columns([1.1, 0.9])

with colL:
    st.subheader("Input")

    # Demo button (repo relative)
    demo_path = next((p for p in DEMO_IMAGE_CANDIDATES if p.exists()), None)
    use_demo = False
    if demo_path is not None:
        use_demo = st.button("Demo 이미지로 테스트 (assets/demo_front.*)")

    uploaded = None if use_demo else st.file_uploader("전신 사진 업로드 (A4 마커 포함 권장)", type=["jpg", "jpeg", "png", "webp"])

# --- Input guard: stop until an image is provided ---
if (not use_demo) and (uploaded is None):
    st.info("Please upload a full-body photo with the A4 marker to start.")
    st.stop()
    use_camera = False if use_demo else st.checkbox("카메라로 촬영 (가능하면)", value=False)
    camera_img = st.camera_input("Capture") if use_camera else None

    _require_cv2()

    if not use_demo and uploaded is None and camera_img is None:
        st.info("이미지를 업로드하거나 카메라 촬영을 해주세요.")
        st.stop()

    try:
        if use_demo:
            img_bgr = cv2.imread(str(demo_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("demo image read failed")
        elif camera_img is not None:
            img_bgr = _read_image(camera_img)
        else:
            img_bgr = _read_image(uploaded)
    except Exception as e:
        st.error(f"이미지 로드 실패: {e}")
        st.stop()

    st_image_compat(_to_rgb(img_bgr), caption="Original")


# --- Calibration ---
calib = None
px_per_cm = 14.0  # fallback init

if mode.startswith("고정밀"):
    calib = detect_a4_calibration(img_bgr, px_per_mm=px_per_mm, dict_name=MARKER_DICT_NAME)
    if not calib.ok:
        with colR:
            st.subheader("A4 Calibration")
            st.error(f"A4 마커 인식 실패: {calib.debug}")
            st.write("팁: 마커(0,1,2,3)가 네 모서리에 모두 보이게, 반사 없이, 너무 작지 않게 촬영하세요.")
        st.stop()

    warped_bgr = warp_to_a4(img_bgr, calib)
    px_per_cm = float(px_per_mm * 10.0)

    with colR:
        st.subheader("A4 Rectified")
        st_image_compat(_to_rgb(warped_bgr), caption="Warped to A4 plane")
        st.caption(f"px/mm = {calib.px_per_mm:.2f}  |  A4(px) = {calib.a4_size_px}")
else:
    with colR:
        st.subheader("Fallback Mode")
        height_cm = st.number_input("키 입력(cm) — A4 없을 때만", min_value=120.0, max_value=220.0, value=175.0, step=0.5)


# --- Pose ---
pose = infer_pose_landmarks(img_bgr, calib if mode.startswith("고정밀") else None)
if pose is None:
    st.warning("Pose 추론 실패(또는 mediapipe 미설치). 일부 출력(팔/다리/회전)이 제한될 수 있습니다.")

# --- Segmentation ---
mask01 = infer_person_mask_yolo(img_bgr, conf=yolo_conf)
if mask01 is None:
    st.error("인물 마스크 추론 실패. (YOLO/seg 모델 또는 person 검출 문제)")
    st.stop()

# Warp mask if calibrated
if mode.startswith("고정밀") and calib and calib.ok:
    work_mask = warp_mask_to_a4(mask01, calib)
else:
    work_mask = mask01

# Fallback px_per_cm: pose + height
if not mode.startswith("고정밀"):
    if pose is not None:
        def get(i): return pose.xy.get(i) if pose and i in pose.xy else None
        top = get(0) or get(11) or get(12)
        aL = get(27) or get(29)
        aR = get(28) or get(30)
        if top and aL and aR:
            y_top = top[1]
            y_bot = max(aL[1], aR[1])
            pix_h = max(1.0, y_bot - y_top)
            px_per_cm = float(pix_h / float(height_cm))

with colR:
    st.subheader("Mask Preview")
    vis = np.dstack([work_mask * 255, work_mask * 255, work_mask * 255]).astype(np.uint8)
    st_image_compat(vis, caption="Person mask (working plane)")
    st.caption(f"px/cm = {px_per_cm:.2f}")

# --- Measurements ---
result = compute_measurements(
    mask01=work_mask,
    pose=pose,
    px_per_cm=px_per_cm,
    garment_class=garment_class,
    material=material,
    ratio_mode=ratio_mode,
    offset_mode=offset_mode,
    erosion_enabled=erosion_enabled,
)

# =========================
# Output
# =========================
st.subheader("Module Summary (시연용)")
rot_txt = f"{result.rotation_deg:.1f}° rotation"
offset_txt = f"Width deduct: -{result.width_deduct_cm:.1f} cm"
thick_txt = f"offset(per-side): {result.offset_per_side_cm:.2f} cm"

summary_lines = [
    "Module 02 — Pose",
    rot_txt,
    "Shoulder depth 기반 회전 보정값이 측정에 반영됩니다.",
    "",
    "Module 03 — Person Mask",
    "YOLOv8-seg 기반 person 마스크로 신체 외곽을 픽셀 단위로 확보합니다.",
    "",
    "Module 04 — Material (MVP)",
    f"{result.material}",
    "confidence: (MVP에서는 UI 선택/placeholder)",
    "",
    "Module 05 — Volume-Offset",
    offset_txt,
    "의복/air-gap/소재 두께 신호를 합산해 body contour를 안쪽으로 복원(erosion 또는 width 차감)합니다.",
    f"({thick_txt})",
]
st.code("\n".join(summary_lines), language="text")

st.subheader("Body Specs (Output)")

def fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.1f} cm"

c1, c2 = st.columns(2)
with c1:
    st.metric("Shoulder Width", fmt(result.shoulder_width_cm))
    st.metric("Chest Circumference", fmt(result.chest_circ_cm))
    st.metric("Waist Circumference", fmt(result.waist_circ_cm))
with c2:
    st.metric("Hip Circumference", fmt(result.hip_circ_cm))
    st.metric("Arm Length", fmt(result.arm_len_cm))
    st.metric("Leg Length", fmt(result.leg_len_cm))

st.subheader("계산 로직(짧게)")
st.write(
    "\n".join(
        [
            f"- 스케일(px/cm): {result.px_per_cm:.2f}",
            "- 고정밀 모드: A4 마커를 Homography로 A4 평면에 정렬 → 동일 좌표계에서 Pose/Mask 측정",
            "- chest/waist/hip: 마스크 y 라인의 center-run 폭(px) → cm 변환 → ratio로 둘레 추정",
            "- volume-offset: garment class + material + looseness로 per-side 두께(cm) 추정 → 폭에서 2배 차감 또는 erosion",
        ]
    )
)

if export_json:
    st.subheader("Export JSON")
    payload = {
        "px_per_cm": result.px_per_cm,
        "rotation_deg": result.rotation_deg,
        "garment_class": result.garment_class,
        "material": result.material,
        "offset_per_side_cm": result.offset_per_side_cm,
        "width_deduct_cm": result.width_deduct_cm,
        "shoulder_width_cm": result.shoulder_width_cm,
        "chest_circ_cm": result.chest_circ_cm,
        "waist_circ_cm": result.waist_circ_cm,
        "hip_circ_cm": result.hip_circ_cm,
        "arm_len_cm": result.arm_len_cm,
        "leg_len_cm": result.leg_len_cm,
        "debug": result.debug,
    }
    st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
