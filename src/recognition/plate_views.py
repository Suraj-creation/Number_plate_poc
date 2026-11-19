"""Deterministic, non-generative views of the pixels inside a plate crop."""

from __future__ import annotations

import math

import cv2
import numpy as np


def deskew(crop):
    """Estimate small in-plane tilt only when two long edges agree."""
    height, width = crop.shape[:2]
    if width < 55 or height < 16 or width < 1.4 * height:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, max(14, width // 6),
                            minLineLength=max(22, width // 3), maxLineGap=max(3, width // 12))
    if lines is None:
        return crop
    angles = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = (angle + 90) % 180 - 90
        if abs(angle) <= 18:
            angles.append(angle)
    if len(angles) < 2:
        return crop
    angle = float(np.median(angles))
    if abs(angle) < 2 or max(abs(x - angle) for x in angles) > 5:
        return crop
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(crop, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def plate_view(crop, method: str):
    if method == "original":
        return crop
    if method == "deskew":
        return deskew(crop)
    if method == "bicubic":
        return cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if method == "unsharp":
        return cv2.addWeighted(crop, 1.35, cv2.GaussianBlur(crop, (0, 0), .9), -.35, 0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if method == "clahe":
        transformed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    elif method == "adaptive":
        transformed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, 9)
    elif method == "denoise":
        return cv2.bilateralFilter(crop, 5, 30, 30)
    else:
        raise ValueError(f"Unknown plate view: {method}")
    return cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)
