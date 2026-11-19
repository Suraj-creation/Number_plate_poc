from __future__ import annotations

import math


def clip_box(box, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [max(0, min(width, int(round(x1)))), max(0, min(height, int(round(y1)))),
            max(0, min(width, int(round(x2)))), max(0, min(height, int(round(y2))))]


def expand_box(box, margin: float, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return clip_box([x1 - mx, y1 - my, x2 + mx, y2 + my], width, height)


def box_iou(a, b) -> float:
    overlap = intersection(a, b)
    return overlap / max(1, area(a) + area(b) - overlap)


def intersection(a, b) -> float:
    return (max(0, min(a[2], b[2]) - max(a[0], b[0])) *
            max(0, min(a[3], b[3]) - max(a[1], b[1])))


def area(box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def mapped_detections(detector, frame, roi, source_pass):
    x1, y1, x2, y2 = roi
    return map_detections(detector.predict(frame[y1:y2, x1:x2]), frame.shape[:2], roi, source_pass)


def map_detections(detections, frame_size, roi, source_pass):
    x1, y1, x2, y2 = roi
    for detection in detections:
        if not math.isfinite(detection["score"]) or not all(map(math.isfinite, detection["bbox"])):
            continue
        a, b, c, d = detection["bbox"]
        # Discard tile predictions cut by an artificial seam. The overlapping tiles
        # and full-frame pass can expose an intact box.
        if source_pass == "tile":
            height, width = frame_size
            if ((x1 > 0 and a <= 0) or (y1 > 0 and b <= 0) or
                    (x2 < width and c >= x2 - x1) or (y2 < height and d >= y2 - y1)):
                continue
        a, b, c, d = clip_box((a, b, c, d), x2 - x1, y2 - y1)
        if c > a and d > b:
            yield {**detection, "bbox": [a + x1, b + y1, c + x1, d + y1],
                   "source_pass": source_pass, "detector_roi_xyxy": list(roi)}


def recovery_tiles(width, height):
    if max(width, height) < 1024:
        return []
    return [(x1, y1, x2, y2)
            for y1, y2 in ((0, round(height * 0.64)), (round(height * 0.36), height))
            for x1, x2 in ((0, round(width * 0.64)), (round(width * 0.36), width))]


def deduplicate_plates(detections: list[dict]) -> list[dict]:
    kept = []
    for candidate in sorted(detections, key=lambda item: item["score"], reverse=True):
        box = candidate["bbox"]
        duplicate = next((other for other in kept if box_iou(box, other["bbox"]) >= 0.5), None)
        trace = {key: candidate[key] for key in ("source_pass", "score", "bbox", "detector_roi_xyxy")}
        if duplicate is None:
            kept.append({**candidate, "detection_passes": [trace]})
        elif len(duplicate["detection_passes"]) < 12:
            duplicate["detection_passes"].append(trace)
    return kept
