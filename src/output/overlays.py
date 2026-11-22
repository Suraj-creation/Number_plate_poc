from __future__ import annotations

import cv2


def plate_label(row: dict) -> str | None:
    if row["plate_bbox_xyxy"] is None:
        return None
    text = row.get("ocr_normalized")
    if row["recognition_status"] == "READ" or row.get("recognition_evidence", {}).get("supports_plate_read"):
        return text
    if row.get("track_recognition_status") == "READ" and text == row.get("track_consensus"):
        return text
    return None


def annotate_plate_reads(frame, rows: list[dict]):
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    color = (0, 220, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row in rows:
        label = plate_label(row)
        if not label:
            continue
        x1, y1, x2, y2 = row["plate_bbox_xyxy"]
        scale = min(max(height / 1080, 0.6), width / 270)
        thickness = max(1, round(scale * 2))
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
        padding = thickness + 3
        label_width = text_width + 2 * padding
        label_height = text_height + baseline + 2 * padding
        left = max(0, min(x1, width - label_width))
        top = y2 + thickness + 2
        if top + label_height > height:
            top = max(0, y1 - label_height - thickness - 2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(canvas, (left, top), (left + label_width, top + label_height), (24, 24, 24), cv2.FILLED)
        cv2.putText(canvas, label, (left + padding, top + padding + text_height), font,
                    scale, color, thickness, cv2.LINE_AA)
    return canvas
