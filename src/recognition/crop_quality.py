from __future__ import annotations

import cv2
import numpy as np


def assess_crop(crop: np.ndarray, min_width: int = 48, blur_min: float = 18.0, min_height: int = 12) -> dict:
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if h > 2 and w > 2 else 0.0
    size_score = min(1.0, w / max(min_width * 2, 1))
    sharp_score = min(1.0, sharpness / max(blur_min * 4, 1e-6))
    quality = 0.55 * size_score + 0.45 * sharp_score
    p10, median, p90 = np.percentile(gray, [10, 50, 90]) if gray.size else (0, 0, 0)
    return {
        "width_px": w, "height_px": h, "laplacian_variance": sharpness,
        "too_small": w < min_width or h < min_height, "blurry": sharpness < blur_min,
        "quality_score": float(quality), "median_intensity": float(median),
        "contrast": float(p90 - p10),
        "shadow_fraction": float(np.mean(gray < 15)) if gray.size else 0.0,
        "highlight_fraction": float(np.mean(gray > 245)) if gray.size else 0.0,
    }
