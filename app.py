import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO
import mediapipe as mp

import timm
from torchvision import transforms

# MediaPipe Tasks
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


# =========================================================
# Config
# =========================================================
CIRC_RATIO_CHEST = 2.25   # 2.35 -> 과대/과소 흔들림 커서 완화
CIRC_RATIO_WAIST = 2.15
CIRC_RATIO_HIP = 2.20

# "공제(deduction)"는 body_mask를 안쪽으로 복원할 때 쓰는 값(전체 너비 기준)
GARMENT_DB = {
    "Short Sleeve": {"base_width_deduct_cm": 0.6, "desc": "Light / Thin Top"},
    "Long Sleeve":  {"base_width_deduct_cm": 1.0, "desc": "Standard Long Sleeve"},
    "Hoodie/Sweat": {"base_width_deduct_cm": 2.2, "desc": "Heavy Knit / Fleece"},
    "Outer/Jacket": {"base_width_deduct_cm": 2.8, "desc": "Outerwear / Thick"},
}

# Fit의 air-gap 공제를 MVP 수준으로 낮춤(기존 1.5~4.0은 반팔에서 과도)
FIT_DB = {
    "Tight":   {"ease_width_deduct_cm": 0.2, "msg": "Skin Fit (Low air-gap)"},
    "Regular": {"ease_width_deduct_cm": 0.7, "msg": "Standard Drape"},
    "Loose":   {"ease_width_deduct_cm": 1.6, "msg": "Air-gap / Gravity Drape"},
}

POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (3, 7), (6, 8), (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

MODELS_DIR = Path(__file__).parent / "models"
POSE_TASK_PATH = MODELS_DIR / "pose_landmarker_heavy.task"
POSE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)


# =========================================================
# Data container
# =========================================================
@dataclass
class Lm:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


# =========================================================
# Cached loaders
# =========================================================
def _ensure_pose_task() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if POSE_TASK_PATH.exists():
        return
    # 다운로드(웹 배포/Streamlit Cloud에서도 동작)
    urllib.request.urlretrieve(POSE_TASK_URL, str(POSE_TASK_PATH))


@st.cache_resource(show_spinner=False)
def load_pose_landmarker() -> vision.PoseLandmarker:
    _ensure_pose_task()
    base = mp_tasks.BaseOptions(model_asset_path=str(POSE_TASK_PATH))
    opts = vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(opts)


@st.cache_resource(show_spinner=False)
def load_yolo() -> YOLO:
    return YOLO("yolov8n-seg.pt")


@st.cache_resource(show_spinner=False)
def load_material_backbone(name: str = "swin_tiny_patch4_window7_224") -> torch.nn.Module:
    # MVP 데모: "실행"이 핵심이므로 pretrained=True 유지
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.eval()
    return model


# =========================================================
# Utils
# =========================================================
def pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    pil_img = ImageOps.exif_transpose(pil_img)
    return np.array(pil_img.convert("RGB"))


def safe_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(int(v), hi))


def lm_xy(lm: List[Lm], idx: int, w: int, h: int) -> Tuple[int, int]:
    return safe_int(lm[idx].x * w, 0, w - 1), safe_int(lm[idx].y * h, 0, h - 1)


def draw_skeleton(img_bgr: np.ndarray, lm: List[Lm]) -> None:
    h, w = img_bgr.shape[:2]
    for a, b in POSE_EDGES:
        ax, ay = lm_xy(lm, a, w, h)
        bx, by = lm_xy(lm, b, w, h)
        cv2.line(img_bgr, (ax, ay), (bx, by), (255, 255, 255), 2, cv2.LINE_AA)
    for i in range(len(lm)):
        x, y = lm_xy(lm, i, w, h)
        cv2.circle(img_bgr, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)


def rotation_compensation(lm: List[Lm]) -> Tuple[float, float]:
    lz, rz = lm[11].z, lm[12].z
    z_diff = abs(lz - rz)
    theta = min(z_diff * 2.0, 1.0)
    deg = math.degrees(theta)
    factor = 1.0 / max(math.cos(theta), 0.55)
    return factor, deg


def geodesic_height_px(lm: List[Lm], w: int, h: int) -> float:
    def dist(i1: int, i2: int) -> float:
        x1, y1 = lm[i1].x * w, lm[i1].y * h
        x2, y2 = lm[i2].x * w, lm[i2].y * h
        return math.hypot(x2 - x1, y2 - y1)

    # head: nose->mid-shoulder (보정은 낮춤: 1.6 -> 1.25)
    mid_sh_x = (lm[11].x + lm[12].x) / 2
    mid_sh_y = (lm[11].y + lm[12].y) / 2
    nose_x, nose_y = lm[0].x * w, lm[0].y * h
    head = math.hypot(mid_sh_x * w - nose_x, mid_sh_y * h - nose_y) * 1.25

    torso = (dist(11, 23) + dist(12, 24)) / 2
    left_leg = dist(23, 25) + dist(25, 27)
    right_leg = dist(24, 26) + dist(26, 28)
    leg = (left_leg + right_leg) / 2
    return head + torso + leg


def vertical_span_height_px(lm: List[Lm], h: int) -> float:
    # 상단: nose/eyes/ears 중 가장 위
    top = min(lm[i].y for i in [0, 1, 2, 3, 4, 5, 6, 7, 8])
    # 하단: heels/foot-index 중 가장 아래
    bottom = max(lm[i].y for i in [29, 30, 31, 32, 27, 28])
    span = max(0.0, (bottom - top) * h)
    return span


def estimate_px_per_cm(lm: List[Lm], w: int, h: int, height_cm: float) -> Tuple[float, Dict[str, float]]:
    """
    1) geodesic + vertical span을 섞어서 스케일 안정화
    2) 어깨/키 비율 sanity calibration(약하게)로 과소/과대 스케일을 완화
    """
    px_geo = geodesic_height_px(lm, w, h)
    px_span = vertical_span_height_px(lm, h)

    # 둘 중 하나가 실패하면 다른 쪽 사용
    px_h = 0.55 * px_span + 0.45 * px_geo if (px_span > 10 and px_geo > 10) else max(px_span, px_geo)
    px_per_cm = px_h / max(height_cm, 1e-6)

    # shoulder sanity (약한 보정)
    sh_px = math.hypot((lm[11].x - lm[12].x) * w, (lm[11].y - lm[12].y) * h)
    shoulder_cm_raw = sh_px / max(px_per_cm, 1e-6)
    ratio = shoulder_cm_raw / max(height_cm, 1e-6)

    # 남성 기준 대략 0.215~0.285 범위에 약하게 맞춤(데모용)
    target_min = 0.215
    target_max = 0.285
    adj = 1.0
    if ratio < target_min:
        adj = ratio / target_min  # <1 => px_per_cm 줄여서(cm 증가)
        px_per_cm *= max(adj, 0.82)  # 과보정 방지
    elif ratio > target_max:
        adj = ratio / target_max  # >1 => px_per_cm 늘려서(cm 감소)
        px_per_cm *= min(adj, 1.18)

    dbg = {
        "px_geo": float(px_geo),
        "px_span": float(px_span),
        "px_h_used": float(px_h),
        "px_per_cm": float(px_per_cm),
        "shoulder_cm_raw": float(shoulder_cm_raw),
        "shoulder_height_ratio": float(ratio),
        "shoulder_sanity_adj": float(adj),
    }
    return float(px_per_cm), dbg


def yolo_person_mask(rgb: np.ndarray, yolo: YOLO) -> Tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    conf = 0.0
    mask_bin = np.zeros((h, w), dtype=np.uint8)

    res = yolo(rgb, verbose=False, classes=[0], conf=0.25)
    if not res or res[0] is None:
        return mask_bin, conf
    r0 = res[0]
    if r0.masks is None or r0.boxes is None or len(r0.boxes) == 0:
        return mask_bin, conf

    c = r0.boxes.conf.detach().cpu()
    best = int(torch.argmax(c).item())
    conf = float(c[best].item())

    m = r0.masks.data[best].detach().cpu().numpy()
    if m.ndim == 2:
        m = cv2.resize(m, (w, h))
        mask_bin = (m > 0.5).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k, iterations=2)
    return mask_bin, conf


def build_adaptive_skin_mask(rgb: np.ndarray, person_mask: np.ndarray, lm: List[Lm]) -> Tuple[np.ndarray, Dict[str, float]]:
    h, w = rgb.shape[:2]
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)

    face_idx = [0, 1, 2, 3, 4, 5, 6, 9, 10]
    samples = []
    for idx in face_idx:
        x, y = lm_xy(lm, idx, w, h)
        x1, x2 = max(0, x - 6), min(w, x + 6)
        y1, y2 = max(0, y - 6), min(h, y + 6)
        patch = ycrcb[y1:y2, x1:x2, :]
        if patch.size > 0:
            samples.append(patch.reshape(-1, 3))

    if len(samples) == 0:
        lower = np.array([0, 135, 85], dtype=np.uint8)
        upper = np.array([255, 180, 135], dtype=np.uint8)
    else:
        s = np.vstack(samples)
        mean = s.mean(axis=0)
        std = s.std(axis=0) + 1e-6

        y_lo = max(25, int(mean[0] - 2.5 * std[0]))
        y_hi = min(255, int(mean[0] + 2.5 * std[0]))
        cr_lo = max(120, int(mean[1] - 2.8 * std[1]))
        cr_hi = min(205, int(mean[1] + 2.8 * std[1]))
        cb_lo = max(70, int(mean[2] - 2.8 * std[2]))
        cb_hi = min(170, int(mean[2] + 2.8 * std[2]))

        lower = np.array([y_lo, cr_lo, cb_lo], dtype=np.uint8)
        upper = np.array([y_hi, cr_hi, cb_hi], dtype=np.uint8)

    skin = cv2.inRange(ycrcb, lower, upper)
    skin = (skin > 0).astype(np.uint8)
    skin = cv2.bitwise_and(skin, skin, mask=person_mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k, iterations=1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k, iterations=2)

    stats = {
        "lower_Y": float(lower[0]), "lower_Cr": float(lower[1]), "lower_Cb": float(lower[2]),
        "upper_Y": float(upper[0]), "upper_Cr": float(upper[1]), "upper_Cb": float(upper[2]),
    }
    return skin, stats


def arm_skin_exposure(skin_mask: np.ndarray, lm: List[Lm], w: int, h: int) -> float:
    def segment_box(i1: int, i2: int, pad: int = 12) -> Tuple[int, int, int, int]:
        x1, y1 = lm_xy(lm, i1, w, h)
        x2, y2 = lm_xy(lm, i2, w, h)
        xa, xb = sorted([x1, x2])
        ya, yb = sorted([y1, y2])
        xa = max(0, xa - pad); xb = min(w - 1, xb + pad)
        ya = max(0, ya - pad); yb = min(h - 1, yb + pad)
        return xa, ya, xb, yb

    rois = [segment_box(11, 13), segment_box(13, 15), segment_box(12, 14), segment_box(14, 16)]
    skin_pixels = 0
    total_pixels = 0
    for xa, ya, xb, yb in rois:
        roi = skin_mask[ya:yb, xa:xb]
        if roi.size == 0:
            continue
        skin_pixels += int(roi.sum())
        total_pixels += int(roi.size)
    return float(skin_pixels / max(total_pixels, 1))


def hood_score(garment_mask: np.ndarray, skin_mask: np.ndarray, lm: List[Lm], w: int, h: int) -> float:
    _, sh_y = lm_xy(lm, 11, w, h)
    _, nose_y = lm_xy(lm, 0, w, h)
    top = max(0, nose_y - int(0.15 * h))
    bottom = min(h - 1, sh_y + int(0.05 * h))

    left_ear_x, _ = lm_xy(lm, 7, w, h)
    right_ear_x, _ = lm_xy(lm, 8, w, h)
    xa = max(0, min(left_ear_x, right_ear_x) - int(0.20 * w))
    xb = min(w - 1, max(left_ear_x, right_ear_x) + int(0.20 * w))

    gm = garment_mask[top:bottom, xa:xb]
    sm = skin_mask[top:bottom, xa:xb]
    if gm.size == 0:
        return 0.0

    garment_ratio = gm.sum() / gm.size
    skin_ratio = sm.sum() / max(sm.size, 1)
    return float(np.clip((garment_ratio * 1.4) - (skin_ratio * 0.6), 0.0, 1.0))


def sobel_wrinkle_ratios(gray: np.ndarray) -> Tuple[float, float]:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    mx = float(np.sum(np.abs(gx)))
    my = float(np.sum(np.abs(gy)))
    total = mx + my + 1e-6
    return mx / total, my / total


def fft_highfreq_ratio(gray: np.ndarray) -> float:
    gray = gray.astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) + 1e-6

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / (max(h, w) / 2 + 1e-6)

    hf = mag[r_norm > 0.55].sum()
    total = mag.sum()
    return float(hf / max(total, 1e-6))


def backbone_signature(rgb_roi: np.ndarray, model: torch.nn.Module, device: str = "cpu") -> Dict[str, float]:
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    x = tfm(rgb_roi).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.ndim == 4:
            feat_std = float(out.std().item())
            feat_norm = float(out.flatten().norm().item())
        else:
            feat_std = float(out.std().item())
            feat_norm = float(out.norm().item())
    return {"bb_feat_std": feat_std, "bb_feat_norm": feat_norm}


def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return sorted(cnts, key=cv2.contourArea, reverse=True)[0]


def width_run_at_y(mask: np.ndarray, y: int, center_x: int) -> Tuple[int, int, int]:
    h, w = mask.shape[:2]
    y = max(0, min(y, h - 1))
    row = mask[y, :]
    if row.sum() == 0:
        return 0, 0, 0
    idx = np.where(row > 0)[0]
    if len(idx) == 0:
        return 0, 0, 0

    runs = []
    start = idx[0]
    prev = idx[0]
    for v in idx[1:]:
        if v == prev + 1:
            prev = v
        else:
            runs.append((start, prev))
            start = v
            prev = v
    runs.append((start, prev))

    best = None
    best_dist = 1e9
    for a, b in runs:
        if a <= center_x <= b:
            best = (a, b)
            break
        dist = min(abs(center_x - a), abs(center_x - b))
        if dist < best_dist:
            best_dist = dist
            best = (a, b)
    if best is None:
        return 0, 0, 0
    a, b = best
    return (b - a), a, b


# =========================================================
# Engine
# =========================================================
class FormFoundryMVP:
    def __init__(self) -> None:
        self.pose = load_pose_landmarker()
        self.yolo = load_yolo()
        self.backbone_name = "swin_tiny_patch4_window7_224"
        self.backbone = load_material_backbone(self.backbone_name)

    def detect_pose(self, rgb: np.ndarray) -> List[Lm]:
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        r = self.pose.detect(mp_image)
        if not r.pose_landmarks:
            return []
        lms = r.pose_landmarks[0]
        out: List[Lm] = []
        for p in lms:
            out.append(Lm(x=p.x, y=p.y, z=getattr(p, "z", 0.0), visibility=getattr(p, "visibility", 1.0)))
        return out

    def garment_classify(
        self, lm: List[Lm], garment_mask: np.ndarray, skin_mask: np.ndarray
    ) -> Tuple[str, Dict[str, float], str]:
        h, w = garment_mask.shape[:2]
        arm_ratio = arm_skin_exposure(skin_mask, lm, w, h)
        hood = hood_score(garment_mask, skin_mask, lm, w, h)

        # Decision rules (t-shirt 오판 줄이기: 팔 skin 우선)
        if arm_ratio >= 0.08:
            g = "Short Sleeve"
            reason = f"Arm-skin exposure {arm_ratio*100:.0f}% (short sleeve prior)"
        else:
            if hood >= 0.38:
                g = "Hoodie/Sweat"
                reason = f"Hood/neck coverage score {hood:.2f} (hood prior)"
            else:
                g = "Long Sleeve"
                reason = f"Low arm-skin {arm_ratio*100:.0f}% and no hood dominance (hood={hood:.2f})"

        dbg = {"arm_skin_ratio": float(arm_ratio), "hood_score": float(hood)}
        return g, dbg, reason

    def material_infer(self, rgb: np.ndarray, garment_mask: np.ndarray, lm: List[Lm]) -> Dict[str, Any]:
        h, w = rgb.shape[:2]

        # ROI = torso bbox (shoulder~hip)
        lsx, lsy = lm_xy(lm, 11, w, h)
        rsx, rsy = lm_xy(lm, 12, w, h)
        lhx, lhy = lm_xy(lm, 23, w, h)
        rhx, rhy = lm_xy(lm, 24, w, h)

        xa = max(0, min(lsx, rsx, lhx, rhx) - int(0.10 * w))
        xb = min(w - 1, max(lsx, rsx, lhx, rhx) + int(0.10 * w))
        ya = max(0, min(lsy, rsy) - int(0.02 * h))
        yb = min(h - 1, max(lhy, rhy) + int(0.03 * h))

        roi = rgb[ya:yb, xa:xb]
        mroi = garment_mask[ya:yb, xa:xb]
        if roi.size == 0:
            roi = rgb[max(0, lsy - 60):min(h, lsy + 80), max(0, lsx - 80):min(w, lsx + 80)]
            mroi = None

        roi_masked = roi.copy()
        if mroi is not None and mroi.size > 0:
            roi_masked[mroi == 0] = 0

        gray = cv2.cvtColor(roi_masked, cv2.COLOR_RGB2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        rough_0_10 = float(np.clip(lap / 200.0, 0.0, 10.0))
        v_ratio, h_ratio = sobel_wrinkle_ratios(gray)
        hf = fft_highfreq_ratio(gray)

        bb = backbone_signature(roi, self.backbone, device="cpu")

        # thickness score (0..1)
        thick_score = 0.50 * (rough_0_10 / 10.0) + 0.35 * hf + 0.15 * np.clip(bb["bb_feat_std"], 0.0, 1.0)
        thick_score = float(np.clip(thick_score, 0.0, 1.0))

        if thick_score >= 0.70:
            thickness_level = "thick"
            thickness_cm = 2.4
            bucket = "knit/fleece"
        elif thick_score >= 0.40:
            thickness_level = "medium"
            thickness_cm = 1.4
            bucket = "cotton blend"
        else:
            thickness_level = "thin"
            thickness_cm = 0.8
            bucket = "smooth cotton/synthetic"

        # confidence (MVP)
        conf = float(np.clip(0.55 + 0.25 * abs(hf - 0.25) + 0.10 * (rough_0_10 / 10.0), 0.0, 0.95))

        return {
            "roi_rgb": roi,
            "roughness_0_10": rough_0_10,
            "wrinkle_v_ratio": float(v_ratio),
            "wrinkle_h_ratio": float(h_ratio),
            "fft_highfreq_ratio": float(hf),
            "backbone": {"name": self.backbone_name, **bb},
            "material_bucket": bucket,
            "thickness_level": thickness_level,
            "thickness_cm": float(thickness_cm),
            "confidence": conf,
        }

    def fit_infer(self, mat: Dict[str, Any]) -> Tuple[str, Dict[str, float], str]:
        v = float(mat["wrinkle_v_ratio"])
        h = float(mat["wrinkle_h_ratio"])
        hf = float(mat["fft_highfreq_ratio"])
        rough = float(mat["roughness_0_10"])

        if v >= 0.62 and hf >= 0.22:
            fit = "Loose"
            reason = f"Vertical drape dominates (v={v:.2f}) + highfreq wrinkles (hf={hf:.2f})"
        elif h >= 0.58 and hf <= 0.22 and rough <= 4.8:
            fit = "Tight"
            reason = f"Horizontal tension dominates (h={h:.2f}) + smoother surface (hf={hf:.2f})"
        else:
            fit = "Regular"
            reason = f"Balanced texture vectors (v={v:.2f}, h={h:.2f})"

        dbg = {"v_ratio": v, "h_ratio": h, "hf_ratio": hf, "rough_0_10": rough}
        return fit, dbg, reason

    def estimate_body_torso_mask(self, person_mask: np.ndarray, lm: List[Lm], px_per_cm: float, width_deduct_cm: float) -> np.ndarray:
        h, w = person_mask.shape[:2]
        _, sh_y = lm_xy(lm, 11, w, h)
        _, hip_y = lm_xy(lm, 23, w, h)
        y1 = max(0, min(sh_y, hip_y) - int(0.03 * h))
        y2 = min(h - 1, max(sh_y, hip_y) + int(0.06 * h))

        torso = np.zeros_like(person_mask)
        torso[y1:y2, :] = person_mask[y1:y2, :]

        # width_deduct_cm 는 "전체 너비 감소" 개념 -> erosion radius는 절반
        deduct_px = max(0, int(width_deduct_cm * px_per_cm))
        radius = max(1, int(deduct_px / 2))
        ksz = radius * 2 + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        body = cv2.erode(torso, k, iterations=1)
        return body

    def measure(self, body_mask: np.ndarray, lm: List[Lm], height_cm: float) -> Tuple[Dict[str, Any], Dict[str, str]]:
        h, w = body_mask.shape[:2]

        px_per_cm, scale_dbg = estimate_px_per_cm(lm, w, h, height_cm)
        rot_factor, rot_deg = rotation_compensation(lm)

        center_x = int(((lm[23].x + lm[24].x) / 2) * w)

        y_sh = int(((lm[11].y + lm[12].y) / 2) * h)
        y_ch = int(((lm[11].y * 2 + lm[23].y) / 3) * h)
        y_wa = int(((lm[11].y + lm[23].y * 2) / 3) * h)
        y_hi = int(((lm[23].y + lm[24].y) / 2) * h)

        sh_px = math.hypot((lm[11].x - lm[12].x) * w, (lm[11].y - lm[12].y) * h)
        sh_cm = (sh_px / px_per_cm) + 1.2  # bias 줄임(1.5->1.2)

        def width_cm_at(y: int) -> Tuple[float, int, int, int]:
            wp, x1, x2 = width_run_at_y(body_mask, y, center_x)
            if wp <= 0:
                return 0.0, 0, 0, 0
            wp_corr = int(wp * rot_factor)
            return float(wp_corr / px_per_cm), x1, x2, wp_corr

        ch_w_cm, ch_x1, ch_x2, ch_wp_corr = width_cm_at(y_ch)
        wa_w_cm, wa_x1, wa_x2, wa_wp_corr = width_cm_at(y_wa)
        hi_w_cm, hi_x1, hi_x2, hi_wp_corr = width_cm_at(y_hi)

        chest = ch_w_cm * CIRC_RATIO_CHEST
        waist = wa_w_cm * CIRC_RATIO_WAIST
        hip = hi_w_cm * CIRC_RATIO_HIP

        def dist(i1: int, i2: int) -> float:
            return math.hypot((lm[i1].x - lm[i2].x) * w, (lm[i1].y - lm[i2].y) * h)

        arm = (dist(11, 13) + dist(13, 15) + dist(12, 14) + dist(14, 16)) / 2 / px_per_cm
        leg = (dist(23, 25) + dist(25, 27) + dist(24, 26) + dist(26, 28)) / 2 / px_per_cm

        meas = {
            "px_per_cm": float(px_per_cm),
            "rotation_deg": float(rot_deg),
            "scale_dbg": scale_dbg,
            "lines": {
                "y_chest": y_ch, "y_waist": y_wa, "y_hip": y_hi,
                "ch_x1": ch_x1, "ch_x2": ch_x2,
                "wa_x1": wa_x1, "wa_x2": wa_x2,
                "hi_x1": hi_x1, "hi_x2": hi_x2,
            },
            "meas_cm": {
                "Shoulder_Width": float(sh_cm),
                "Chest_Circ": float(chest),
                "Waist_Circ": float(waist),
                "Hip_Circ": float(hip),
                "Arm_Length": float(arm),
                "Leg_Length": float(leg),
            }
        }

        explain = {
            "Shoulder_Width": (
                f"Module02(Pose): dist(shoulder11-12)={sh_px:.0f}px ÷ px_per_cm({px_per_cm:.2f}) + bias(1.2cm). "
                f"Scale uses mix(span+geodesic) + weak shoulder sanity."
            ),
            "Chest_Circ": (
                f"Module05(BodyMask): at y_chest, center-run width={ch_wp_corr}px ÷ px_per_cm({px_per_cm:.2f}) "
                f"→ width_cm({ch_w_cm:.1f}) × ratio({CIRC_RATIO_CHEST}). "
                f"Arms excluded by center-run; rotation corrected."
            ),
            "Waist_Circ": (
                f"Module05(BodyMask): at y_waist, center-run width={wa_wp_corr}px ÷ px_per_cm({px_per_cm:.2f}) "
                f"→ width_cm({wa_w_cm:.1f}) × ratio({CIRC_RATIO_WAIST})."
            ),
            "Hip_Circ": (
                f"Module05(BodyMask): at y_hip, center-run width={hi_wp_corr}px ÷ px_per_cm({px_per_cm:.2f}) "
                f"→ width_cm({hi_w_cm:.1f}) × ratio({CIRC_RATIO_HIP})."
            ),
            "Arm_Length": (
                f"Module02(Pose): avg(LS arm + RS arm) "
                f"= (dist(11-13)+dist(13-15)+dist(12-14)+dist(14-16))/2 ÷ px_per_cm({px_per_cm:.2f})."
            ),
            "Leg_Length": (
                f"Module02(Pose): avg(hip-knee-ankle left/right) "
                f"= (dist(23-25)+dist(25-27)+dist(24-26)+dist(26-28))/2 ÷ px_per_cm({px_per_cm:.2f})."
            ),
        }
        return meas, explain

    def render_overlay(
        self,
        rgb: np.ndarray,
        lm: List[Lm],
        person_mask: np.ndarray,
        garment_mask: np.ndarray,
        body_mask: np.ndarray,
        meas: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        h, w = rgb.shape[:2]
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Person overlay (blue-ish) + Garment overlay (green)
        person_col = np.zeros_like(base); person_col[:] = (255, 120, 0)   # BGR
        garment_col = np.zeros_like(base); garment_col[:] = (0, 255, 0)

        pm = (person_mask.astype(np.uint8) * 255)
        gm = (garment_mask.astype(np.uint8) * 255)

        person_col_masked = cv2.bitwise_and(person_col, person_col, mask=pm)
        garment_col_masked = cv2.bitwise_and(garment_col, garment_col, mask=gm)

        base = cv2.addWeighted(base, 1.0, person_col_masked, 0.16, 0)
        base = cv2.addWeighted(base, 1.0, garment_col_masked, 0.20, 0)

        # Body contour (red)
        cnt = largest_contour(body_mask)
        if cnt is not None and cv2.contourArea(cnt) > 2000:
            cv2.drawContours(base, [cnt], -1, (0, 0, 255), 3, cv2.LINE_AA)

        # Skeleton
        draw_skeleton(base, lm)

        # Measurement lines
        L = meas["lines"]
        font = cv2.FONT_HERSHEY_SIMPLEX

        def line(y: int, x1: int, x2: int, label: str, color: Tuple[int, int, int]) -> None:
            if x2 > x1 > 0 and 0 <= y < h:
                cv2.line(base, (x1, y), (x2, y), color, 3, cv2.LINE_AA)
                cv2.putText(base, label, (min(x2 + 8, w - 1), max(18, y - 6)), font, 0.65, color, 2, cv2.LINE_AA)

        line(L["y_chest"], L["ch_x1"], L["ch_x2"], "Chest", (0, 255, 255))
        line(L["y_waist"], L["wa_x1"], L["wa_x2"], "Waist", (255, 0, 255))
        line(L["y_hip"], L["hi_x1"], L["hi_x2"], "Hip", (0, 255, 0))

        overlay_rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

        def to_rgb_mask(m: np.ndarray) -> np.ndarray:
            return np.stack([m * 255, m * 255, m * 255], axis=-1).astype(np.uint8)

        masks_rgb = np.concatenate(
            [to_rgb_mask(person_mask), to_rgb_mask(garment_mask), to_rgb_mask(body_mask)], axis=1
        )
        return {"overlay": overlay_rgb, "masks": masks_rgb}

    def process(self, image_file: Any, height_cm: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        rgb = pil_to_rgb(Image.open(image_file))
        h, w = rgb.shape[:2]

        # Module 02: Pose
        lm = self.detect_pose(rgb)
        if len(lm) < 33:
            return None, "PoseLandmarker failed: 전신/포즈 인식이 불안정합니다. 더 밝은 배경/전신 프레이밍으로 재시도하세요."

        # Module 03: Person Seg
        person_mask, yolo_conf = yolo_person_mask(rgb, self.yolo)
        if person_mask.sum() < 5000:
            return None, "YOLO person segmentation 실패: 인물이 충분히 크게 나오거나 배경 대비가 있는 사진으로 재시도하세요."

        # Skin vs garment (person - skin)
        skin_mask, skin_stats = build_adaptive_skin_mask(rgb, person_mask, lm)
        garment_mask = cv2.bitwise_and(person_mask, (1 - skin_mask).astype(np.uint8))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_CLOSE, k, iterations=2)

        # Module 04: Material
        mat = self.material_infer(rgb, garment_mask, lm)
        fit, fit_dbg, fit_reason = self.fit_infer(mat)

        # Garment class
        garment_class, g_dbg, g_reason = self.garment_classify(lm, garment_mask, skin_mask)

        # Width deduction (Module 05)
        base_d = GARMENT_DB[garment_class]["base_width_deduct_cm"]
        ease_d = FIT_DB[fit]["ease_width_deduct_cm"]

        # 소재 두께는 "추정치"라 영향 더 줄임(0.35 -> 0.15)
        mat_d = float(mat["thickness_cm"] * 0.15)

        width_deduct_cm = float(base_d + ease_d + mat_d)

        # 반팔이면 과도 deduction을 제한(데모에서 제일 튀는 포인트)
        if garment_class == "Short Sleeve":
            width_deduct_cm = min(width_deduct_cm, 1.2)

        # scale
        px_per_cm, scale_dbg = estimate_px_per_cm(lm, w, h, height_cm)

        # body mask estimate
        body_mask = self.estimate_body_torso_mask(person_mask, lm, px_per_cm, width_deduct_cm)

        # measurements + per-metric explain
        meas, explain = self.measure(body_mask, lm, height_cm)

        # overlay
        renders = self.render_overlay(rgb, lm, person_mask, garment_mask, body_mask, meas)

        # reasoning (for expander)
        reasoning = {
            "module_02_pose": {
                "landmarks": 33,
                "rotation_deg": float(meas["rotation_deg"]),
                "scale_dbg": scale_dbg,
            },
            "module_03_seg": {
                "yolo_person_conf": float(yolo_conf),
                "person_area_px": float(person_mask.sum()),
                "garment_area_px": float(garment_mask.sum()),
                "skin_area_px": float(skin_mask.sum()),
                "skin_stats": skin_stats,
            },
            "module_04_material": {
                "bucket": mat["material_bucket"],
                "thickness_level": mat["thickness_level"],
                "thickness_cm": mat["thickness_cm"],
                "confidence": mat["confidence"],
                "roughness_0_10": mat["roughness_0_10"],
                "fft_highfreq_ratio": mat["fft_highfreq_ratio"],
                "wrinkle_v_ratio": mat["wrinkle_v_ratio"],
                "wrinkle_h_ratio": mat["wrinkle_h_ratio"],
                "backbone": mat["backbone"],
            },
            "module_05_offset": {
                "garment_class": garment_class,
                "garment_reason": g_reason,
                "garment_debug": g_dbg,
                "fit": fit,
                "fit_reason": fit_reason,
                "fit_debug": fit_dbg,
                "base_width_deduct_cm": base_d,
                "ease_width_deduct_cm": ease_d,
                "material_width_deduct_cm": mat_d,
                "width_deduct_cm_used": width_deduct_cm,
            },
        }

        return {
            "overlay": renders["overlay"],
            "masks": renders["masks"],
            "roi": mat["roi_rgb"],
            "garment_class": garment_class,
            "garment_reason": g_reason,
            "fit": fit,
            "fit_reason": fit_reason,
            "material_bucket": mat["material_bucket"],
            "material_conf": mat["confidence"],
            "deduct_cm": width_deduct_cm,
            "meas_cm": meas["meas_cm"],
            "rotation_deg": meas["rotation_deg"],
            "reasoning": reasoning,
            "explain": explain,
        }, None


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(layout="wide", page_title="FormFoundry MVP — Modules 02/03/04/05")

st.markdown(
    """
<style>
  .card { background-color: #141414; border: 1px solid #2a2a2a; padding: 14px; border-radius: 10px; margin-bottom: 10px; }
  .card-title { color: #8b8b8b; font-size: 12px; text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 6px; }
  .card-value { color: #ffffff; font-size: 22px; font-weight: 700; }
  .card-sub { color: #b8b8b8; font-size: 13px; line-height: 1.35; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("FormFoundry MVP — Pose • Segmentation • Material(ViT/Swin) • Volume-Offset")
st.caption("Module 02/03/04/05를 모두 실제 실행. 각 수치별 계산 근거를 Body Specs 카드 아래에 텍스트로 표시합니다.")

with st.sidebar:
    st.header("Config")
    height_cm = st.number_input("Height (cm)", 150, 210, 182, step=1)
    st.caption("Run locally:")
    st.code("streamlit run app.py", language="bash")
    st.caption("Pose model will auto-download into ./models/ on first run.")

uploaded = st.file_uploader("Upload full-body image", type=["jpg", "jpeg", "png"])

# init
try:
    engine = FormFoundryMVP()
except Exception as e:
    st.error("Engine initialization failed.")
    st.code(str(e))
    st.stop()

if uploaded:
    with st.spinner("Running full pipeline (02/03/04/05)..."):
        time.sleep(0.15)
        data, err = engine.process(uploaded, float(height_cm))

    if err:
        st.error(err)
        st.stop()

    col1, col2, col3 = st.columns([1.35, 0.95, 0.95])

    with col1:
        st.markdown("#### Visual Output")
        t1, t2, t3 = st.tabs(["Overlay", "Masks", "ROI"])
        with t1:
            st.image(
                data["overlay"],
                use_container_width=True,
                caption="Overlay: Person(blue) + Garment(green) + Body contour(red) + Skeleton + Measure lines",
            )
        with t2:
            st.image(
                data["masks"],
                use_container_width=True,
                caption="Masks: Person | Garment(person-skin) | Body estimate (erosion offset)",
            )
        with t3:
            st.image(data["roi"], use_container_width=True, caption="Torso ROI (Material inference input)")

    with col2:
        st.markdown("#### Module Summary (시연용)")
        rot = data["rotation_deg"]
        deduct = data["deduct_cm"]

        st.markdown(
            f"""
<div class="card">
  <div class="card-title">Module 02 — Pose</div>
  <div class="card-value">{rot:.1f}° rotation</div>
  <div class="card-sub">Shoulder depth 기반 회전 보정값이 측정에 반영됩니다.</div>
</div>
<div class="card">
  <div class="card-title">Module 03 — Garment Class</div>
  <div class="card-value">{data["garment_class"]}</div>
  <div class="card-sub">{data["garment_reason"]}</div>
</div>
<div class="card">
  <div class="card-title">Module 04 — Material (Swin 실행)</div>
  <div class="card-value" style="font-size:18px;">{data["material_bucket"]}</div>
  <div class="card-sub">confidence: {data["material_conf"]:.2f}</div>
</div>
<div class="card">
  <div class="card-title">Module 05 — Volume-Offset</div>
  <div class="card-value">Width deduct: -{deduct:.1f} cm</div>
  <div class="card-sub">의복/air-gap/소재 두께를 합산해 body contour를 안쪽으로 복원(erosion)합니다.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander("Demo용 상세 수치(원하면 펼치기)"):
            st.json(data["reasoning"])

    with col3:
        st.markdown("#### Body Specs (Output)")
        m = data["meas_cm"]
        ex = data.get("explain", {})

        def card(label: str, val: float, explain: str) -> None:
            explain = explain or "-"
            st.markdown(
                f"""
<div class="card">
  <div class="card-title">{label}</div>
  <div class="card-value">{val:.1f} cm</div>
  <div class="card-sub"><b>Calculation:</b> {explain}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        card("Shoulder Width", m["Shoulder_Width"], ex.get("Shoulder_Width", ""))
        card("Chest Circumference", m["Chest_Circ"], ex.get("Chest_Circ", ""))
        card("Waist Circumference", m["Waist_Circ"], ex.get("Waist_Circ", ""))
        card("Hip Circumference", m["Hip_Circ"], ex.get("Hip_Circ", ""))

        cA, cB = st.columns(2)
        with cA:
            card("Arm Length", m["Arm_Length"], ex.get("Arm_Length", ""))
        with cB:
            card("Leg Length", m["Leg_Length"], ex.get("Leg_Length", ""))

        st.button("Save Profile (MVP)", type="primary", use_container_width=True)
