"""Replay real OCR on deterministic corruptions of labeled still crops."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import levenshtein
from eval.production_benchmark import save
from src.recognition.crop_quality import assess_crop
from src.recognition.evidence import recognition_evidence
from src.recognition.fast_plate_ocr import FastPlateOCR
from src.tracking.track_state import TrackState
from src.utils.config import load_config


def perturb(crop):
    h, w = crop.shape[:2]
    line = np.ones((1, 11), dtype=np.float32) / 11
    pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    skew = np.float32([[.12*w, .13*h], [w-1, 0], [.87*w, .80*h], [0, h-1]])
    gradient = np.linspace(0, 1, w)[None, :, None]
    dim = (crop.astype(np.float32) * (.28 + .22 * gradient)).astype(np.uint8)
    flare = np.exp(-((gradient - .6) / .14) ** 2)
    glare = np.clip(crop * (1-.80*flare) + 255*.80*flare, 0, 255).astype(np.uint8)
    small = cv2.resize(crop, (min(w, 42), max(5, min(h, round(42*h/w)))), interpolation=cv2.INTER_AREA)
    angle = cv2.warpPerspective(crop, cv2.getPerspectiveTransform(pts, skew), (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
    return [("dim", dim), ("horizontal_motion_blur", cv2.filter2D(crop, -1, line)), ("original", crop),
            ("glare", glare), ("small", small), ("perspective", angle),
            ("slight_blur", cv2.GaussianBlur(crop, (5, 5), 1.2)), ("repeat", crop)]


def grade(summary, expected, candidate, status=None):
    summary.update({"cases": 1, "exact": candidate == expected, "chars": len(expected),
                    "edits": levenshtein(candidate or "", expected), "reads": status == "READ",
                    "incorrect_reads": status == "READ" and candidate != expected})


def run(cohort, cfg):
    reference = json.loads((cohort / "private_ground_truth.json").read_text())
    reader = FastPlateOCR(device=cfg["device"])
    modes = {}
    conditions = {}
    evidence = []
    for name, label in reference.items():
        truth = label["truth"]
        crop = cv2.imread(str(cohort / "oracle" / "original" / name))
        views = perturb(crop)
        reads = reader.predict_many([image for _, image in views])
        track = TrackState(cfg["track_history_size"], cfg)
        rows = []
        for frame, ((condition, image), read) in enumerate(zip(views, reads, strict=True)):
            timestamp = frame * 333
            track.seen([{"track_id": 1, "class_name": "car"}], frame, 45, timestamp)
            quality = assess_crop(image, cfg["min_plate_width_px"], cfg["blur_laplacian_min"], cfg["min_plate_height_px"])
            # This test controls crop origin and association. It does not measure either detector.
            candidate, _, status, details = recognition_evidence(
                read, quality, {"score": 1}, {"state": "ASSOCIATED"}, False, [], cfg)
            fingerprint = hashlib.blake2b(image.tobytes(), digest_size=12).hexdigest()
            observation = {"ocr_normalized": candidate, "quality_score": quality["quality_score"],
                           "crop_readable": not (quality["too_small"] or quality["blurry"]), "fingerprint": fingerprint,
                           "supports_track_read": details["supports_track_read"], "timestamp_ms": timestamp,
                           "frame_index": frame}
            vote = track.add(1, observation)
            rows.append({**observation, "condition": condition, "status": status, "raw_score": read["raw_score"],
                         "current_track_candidate": vote["text"]})
            grade(conditions.setdefault(condition, Counter()), truth, candidate, status)
        finished = track.finish(1)
        # Deduplicate exact crop bytes for an equally scoped simple-majority baseline.
        unique = {row["fingerprint"]: row["ocr_normalized"] for row in rows if row["ocr_normalized"]}
        majority = Counter(unique.values()).most_common(1)
        best = max(rows, key=lambda row: row["quality_score"])
        for mode, candidate, status in (
            ("first_frame", rows[0]["ocr_normalized"], rows[0]["status"]),
            ("best_quality_frame", best["ocr_normalized"], best["status"]),
            ("simple_majority", majority[0][0] if majority else "", None),
            ("quality_weighted_track", finished["consensus"], finished["recognition_status"]),
        ):
            grade(modes.setdefault(mode, Counter()), truth, candidate, status)
        evidence.append({"crop": name, "truth": truth, "frames": rows, "track": finished})
    return {"modes": modes, "conditions": conditions}, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    aggregate, private = run(args.cohort, load_config())
    save(args.output, aggregate)
    save(args.private_output, private)


if __name__ == "__main__":
    main()
