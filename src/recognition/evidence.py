from __future__ import annotations

import re

import cv2
import numpy as np

from src.recognition.india_format import validate_indian_format
from src.recognition.normalize import normalize_text


def recovery_views(crop, quality: dict) -> list[tuple]:
    """Bound the extra inference and retain a label for every pixel transform."""
    height, width = crop.shape[:2]
    if min(width, height) < 8:
        return []
    views = []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if quality["contrast"] < 55 or quality["median_intensity"] < 65:
        # Cap contrast amplification in low light. Never change the source crop.
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        views.append(("local_contrast", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)))
    if 0.85 <= width / height <= 2.35:
        # Try flattened motorcycle-style rows only when there is a clear horizontal gap.
        gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)).mean(axis=1)
        start, stop = int(height * 0.36), int(height * 0.65)
        split = start + int(np.argmin(gradient[start:stop]))
        before, after = gradient[height // 10:split], gradient[split + 1:height - height // 10]
        if before.size and after.size and min(before.mean(), after.mean()) > max(5, gradient[split] * 1.8):
            row_height = max(split, height - split)
            rows = [cv2.resize(row, (width, row_height), interpolation=cv2.INTER_LINEAR)
                    for row in (crop[:split], crop[split:])]
            views.append((f"rows_flattened_at_{split}", cv2.hconcat(rows)))
    return views


def recognition_evidence(primary: dict, quality: dict, detection: dict, association: dict,
                         clipped: bool, alternatives: list[dict], cfg: dict) -> tuple:
    raw = primary.get("raw")
    text = normalize_text(raw)
    fmt = validate_indian_format(text)
    reasons = []
    if not text:
        reasons.append("no_primary_text")
    if not fmt["plausible"]:
        reasons.append("format_not_recognized")
    if raw and "_" in raw:
        reasons.append("internal_ocr_padding")
    if raw and re.search(r"[^A-Za-z0-9\s._-]", raw):
        reasons.append("unsupported_ocr_symbols")
    if quality["too_small"]:
        reasons.append("insufficient_source_pixels")
    if quality["blurry"]:
        reasons.append("low_sharpness")
    if quality["contrast"] < cfg["min_plate_contrast"]:
        reasons.append("low_contrast")
    if clipped:
        reasons.append("plate_at_frame_boundary")
    if detection["score"] < cfg["plate_read_threshold"]:
        reasons.append("plate_detector_below_read_gate")
    independent = [item for item in alternatives if item.get("independent")]
    disagreement = any(item["normalized"] and item["normalized"] != text and
                       (item.get("raw_score") or 0) >= cfg["paddle_evidence_threshold"] for item in independent)
    agreement = bool(text and any(item["normalized"] == text and
                     (item.get("raw_score") or 0) >= cfg["paddle_evidence_threshold"] for item in independent))
    if disagreement:
        reasons.append("independent_ocr_disagrees")
    hard_gate = not reasons
    mean, weakest = primary.get("raw_score"), primary.get("minimum_char_score")
    strong = mean is not None and weakest is not None and mean >= cfg["ocr_read_threshold"] and weakest >= cfg["ocr_min_char_threshold"]
    support = mean is not None and weakest is not None and mean >= cfg["ocr_track_mean_threshold"] and weakest >= cfg["ocr_track_min_char_threshold"]
    if not strong:
        reasons.append("primary_ocr_below_read_gate")
    if association["state"] != "ASSOCIATED":
        reasons.append(association["state"].lower())
        status = "UNASSOCIATED"
    elif not text and not any(item["normalized"] for item in alternatives):
        status = "UNREADABLE"
    elif hard_gate and (strong or (support and agreement)):
        status = "READ"
    else:
        status = "LOW_CONFIDENCE"
    evidence = {
        "reasons": reasons, "independent_agreement": agreement, "independent_disagreement": disagreement,
        "read_basis": ("original_crop" if strong else "independent_agreement") if status == "READ" else None,
        "supports_track_read": bool(hard_gate and (strong or support) and association["state"] == "ASSOCIATED"),
    }
    return text, fmt, status, evidence
