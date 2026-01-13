from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2


@dataclass(frozen=True)
class RowSpan:
    x_left: int
    x_right: int
    x_center: int


def find_center_run_span(mask: np.ndarray, y: int) -> Optional[RowSpan]:
    """
    center-run: row에서 이미지 중앙에 가장 가까운 foreground를 잡고,
    그 픽셀을 기준으로 좌/우로 붙어있는 구간만 확장.
    (팔이 떨어져 있으면 자연스럽게 제외되는 장점)
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    h, w = mask.shape
    if y < 0 or y >= h:
        return None

    row = mask[y].astype(bool)
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return None

    xc = w // 2
    x0 = int(xs[np.argmin(np.abs(xs - xc))])

    xl = x0
    while xl - 1 >= 0 and row[xl - 1]:
        xl -= 1
    xr = x0
    while xr + 1 < w and row[xr + 1]:
        xr += 1

    return RowSpan(x_left=xl, x_right=xr, x_center=x0)


def width_cm_from_span(span: RowSpan, y: int, H_img_to_a4: Optional[np.ndarray]) -> Optional[float]:
    """
    Homography(image->A4cm)가 있으면 row 폭을 cm로 반환.
    없으면 None.
    """
    if H_img_to_a4 is None:
        return None

    p1 = np.array([[[float(span.x_left), float(y)]]], dtype=np.float32)
    p2 = np.array([[[float(span.x_right), float(y)]]], dtype=np.float32)

    q1 = cv2.perspectiveTransform(p1, H_img_to_a4)[0, 0]
    q2 = cv2.perspectiveTransform(p2, H_img_to_a4)[0, 0]

    dx = float(q2[0] - q1[0])
    dy = float(q2[1] - q1[1])
    return float((dx * dx + dy * dy) ** 0.5)
