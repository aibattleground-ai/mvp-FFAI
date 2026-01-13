from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import cv2


A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7

# Marker sheet IDs: 0=TL, 1=TR, 2=BR, 3=BL
SHEET_MARKER_IDS = (0, 1, 2, 3)


@dataclass(frozen=True)
class A4CalibrationResult:
    ok: bool
    H_img_to_a4: Optional[np.ndarray]   # 3x3 homography: image(x,y) -> A4 plane (cm)
    corners_img: Optional[np.ndarray]   # (4,2) float32: [TL,TR,BR,BL] in image
    debug: str


def _aruco_detect(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _rej = detector.detectMarkers(gray)
    else:
        corners, ids, _rej = aruco.detectMarkers(gray, dictionary, parameters=params)

    return corners, ids


def calibrate_a4_from_aruco(img_bgr: np.ndarray) -> A4CalibrationResult:
    if img_bgr is None or img_bgr.size == 0:
        return A4CalibrationResult(False, None, None, "Empty image")

    corners_list, ids = _aruco_detect(img_bgr)
    if ids is None or len(ids) == 0:
        return A4CalibrationResult(False, None, None, "No ArUco markers detected")

    ids = ids.flatten().tolist()
    id_to_corners: Dict[int, np.ndarray] = {}
    for c, mid in zip(corners_list, ids):
        id_to_corners[int(mid)] = np.asarray(c, dtype=np.float32).reshape(4, 2)

    missing = [mid for mid in SHEET_MARKER_IDS if mid not in id_to_corners]
    if missing:
        return A4CalibrationResult(
            False, None, None,
            f"Missing markers: {missing}. Detected: {sorted(id_to_corners.keys())}"
        )

    # ArUco corners are ordered (TL, TR, BR, BL) in marker coords.
    tl = id_to_corners[0][0]
    tr = id_to_corners[1][1]
    br = id_to_corners[2][2]
    bl = id_to_corners[3][3]
    corners_img = np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

    corners_a4 = np.array([
        [0.0,         A4_HEIGHT_CM],   # TL
        [A4_WIDTH_CM, A4_HEIGHT_CM],   # TR
        [A4_WIDTH_CM, 0.0],            # BR
        [0.0,         0.0],            # BL
    ], dtype=np.float32)

    H, inliers = cv2.findHomography(corners_img, corners_a4, method=0)
    if H is None:
        return A4CalibrationResult(False, None, corners_img, "findHomography failed")

    if inliers is not None and int(inliers.sum()) < 4:
        return A4CalibrationResult(False, None, corners_img, "Homography inliers < 4")

    return A4CalibrationResult(True, H, corners_img, "OK")


def draw_a4_debug_overlay(img_bgr: np.ndarray, calib: A4CalibrationResult) -> np.ndarray:
    out = img_bgr.copy()
    if calib.corners_img is not None:
        pts = calib.corners_img.astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for i, (x, y) in enumerate(pts):
            cv2.circle(out, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(
                out, ["TL", "TR", "BR", "BL"][i],
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )
    cv2.putText(
        out, f"A4 calib: {calib.debug}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA
    )
    return out
