from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.recognition.normalize import normalize_text


def levenshtein(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = current
    return row[-1]


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0


def parse_box(value):
    if not value:
        return None
    coords = [float(x) for x in value.replace(";", ",").strip("[]").split(",")]
    if len(coords) != 4 or not all(np.isfinite(coords)) or coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError(f"Invalid ground-truth box: {value}")
    return coords


def frame_key(row):
    return row["source"], int(row.get("frame_index") or 0)


def match_boxes(truth, predicted, threshold):
    if not truth or not predicted:
        return []
    overlaps = np.array([[iou(a, b) for b in predicted] for a in truth])
    valid = overlaps >= threshold
    # Maximize eligible match count first, then total IoU.
    scores = valid * (min(len(truth), len(predicted)) + 1 + overlaps)
    indices = zip(*linear_sum_assignment(scores, maximize=True))
    return [(a, b) for a, b in indices if valid[a, b]]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate(predictions_path: Path, ground_truth_path: Path, iou_threshold: float = 0.5) -> dict:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1].")
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    with ground_truth_path.open(newline="", encoding="utf-8-sig") as stream:
        truth = list(csv.DictReader(stream))
    if not truth:
        raise ValueError("Ground truth must contain at least one annotated frame.")
    scores = evaluate_rows(payload, truth, iou_threshold)
    scores["by_condition"] = {
        condition: evaluate_rows(payload, [row for row in truth if row.get("condition") == condition], iou_threshold)
        for condition in sorted({row["condition"] for row in truth if row.get("condition")})
    }
    return scores


def evaluate_rows(payload, truth, threshold=0.5):
    frame_truth = defaultdict(list)
    for row in truth:
        if not row.get("source"):
            raise ValueError("Ground truth requires a source for every row.")
        box = parse_box(row.get("plate_bbox_xyxy"))
        label = normalize_text(row.get("ground_truth"))
        if label and not box:
            raise ValueError("An OCR ground-truth label needs a plate box.")
        frame_truth[frame_key(row)].append((row, box, label))

    frame_predictions = defaultdict(list)
    for pred in payload.get("observations", []):
        if frame_key(pred) in frame_truth:
            frame_predictions[frame_key(pred)].append(pred)

    counts = Counter()
    trajectories = defaultdict(list)
    reference_texts = defaultdict(set)
    legacy_id_matches = 0
    for key, frame_gt in frame_truth.items():
        predictions = frame_predictions[key]
        pred_plates = [row for row in predictions if row.get("plate_bbox_xyxy") is not None]
        gt_plates = [(row, box, text) for row, box, text in frame_gt if box]
        gt_vehicles = {tuple(box) for row, _, _ in frame_gt
                       if (box := parse_box(row.get("vehicle_bbox_xyxy")))}
        pred_vehicles = {tuple(row["vehicle_bbox_xyxy"]) for row in predictions if row.get("vehicle_bbox_xyxy")}
        counts["gt_vehicles"] += len(gt_vehicles)
        counts["matched_vehicles"] += len(match_boxes(sorted(gt_vehicles), sorted(pred_vehicles), threshold))
        counts["gt"] += len(gt_plates)
        counts["predictions"] += len(pred_plates)
        counts["unreadable"] += sum(p.get("recognition_status") == "UNREADABLE" for p in pred_plates)
        counts["candidate_missing"] += sum(not p.get("ocr_normalized") for p in pred_plates)
        pairs = dict(match_boxes([box for _, box, _ in gt_plates],
                                 [p["plate_bbox_xyxy"] for p in pred_plates], threshold))
        counts["matched"] += len(pairs)
        unmatched_reads = sum(p.get("recognition_status") == "READ" for i, p in enumerate(pred_plates)
                              if i not in pairs.values())
        counts["unmatched_reads"] += unmatched_reads
        # False reads on unmatched boxes require an exhaustive box reference.
        exhaustive = all(str(row.get("boxes_exhaustive", "true")).lower() in ("1", "true", "yes")
                         for row, _, _ in frame_gt)
        if exhaustive:
            counts["confident_wrong"] += unmatched_reads
            counts["confident_total"] += unmatched_reads
        for i, (gt, _, label) in enumerate(gt_plates):
            track_label = str(gt.get("vehicle_track_id") or "").strip()
            if track_label and label:
                reference_texts[(key[0], track_label)].add(label)
            pred = pred_plates[pairs[i]] if i in pairs else None
            if label:
                candidate = normalize_text(pred.get("ocr_normalized")) if pred else ""
                edits = levenshtein(candidate, label)
                counts["ocr_gt"] += 1
                counts["ocr_end_to_end_edits"] += edits
                counts["ocr_gt_chars"] += len(label)
                if pred:
                    counts["ocr_matched"] += 1
                    counts["ocr_exact"] += candidate == label
                    counts["ocr_edits"] += edits
                    counts["ocr_matched_chars"] += len(label)
                    if pred.get("recognition_status") == "READ":
                        counts["confident_total"] += 1
                        counts["confident_wrong"] += candidate != label
            if not pred:
                continue
            predicted_track = pred.get("vehicle_track_id")
            gt_vehicle = parse_box(gt.get("vehicle_bbox_xyxy"))
            if gt_vehicle:
                counts["association"] += 1
                predicted_vehicle = pred.get("vehicle_bbox_xyxy")
                counts["association_correct"] += bool(predicted_vehicle and iou(gt_vehicle, predicted_vehicle) >= threshold)
            elif track_label and str(gt.get("track_ids_are_prediction_ids", "")).lower() in ("1", "true", "yes"):
                legacy_id_matches += 1
                counts["association"] += 1
                counts["association_correct"] += str(predicted_track) == track_label
            if track_label and predicted_track is not None:
                trajectories[(key[0], track_label)].append((key[1], str(predicted_track)))

    run_tracks = {(row.get("source"), str(row["vehicle_track_id"])): row for row in payload.get("tracks", [])}
    only_source = next(iter(frame_truth))[0] if len({key[0] for key in frame_truth}) == 1 else None
    correct_tracks = 0
    mapping = {}
    for source in {key[0] for key in trajectories}:
        ref_keys = sorted(key for key in trajectories if key[0] == source)
        pred_ids = sorted({track_id for key in ref_keys for _, track_id in trajectories[key]})
        overlap = np.array([[sum(p == pred_id for _, p in trajectories[key]) for pred_id in pred_ids] for key in ref_keys])
        for i, j in zip(*linear_sum_assignment(overlap, maximize=True)):
            if overlap[i, j]:
                mapping[ref_keys[i]] = pred_ids[j]
    eligible = {key: next(iter(texts)) for key, texts in reference_texts.items() if len(texts) == 1}
    track_read_count = track_wrong_reads = 0
    for key, label in eligible.items():
        track = run_tracks.get((key[0], mapping.get(key))) or (
            run_tracks.get((None, mapping.get(key))) if key[0] == only_source else None)
        if track:
            exact = normalize_text(track.get("consensus")) == label
            correct_tracks += exact
            if track.get("recognition_status") == "READ":
                track_read_count += 1
                track_wrong_reads += not exact
    switches = 0
    for evidence in trajectories.values():
        ids = [track for _, track in sorted(set(evidence))]
        switches += sum(a != b for a, b in zip(ids, ids[1:]))
    latency = payload.get("run_metadata", {}).get("performance_ms", {}).get("total", {}).get("mean")
    return {
        "annotated_frames": len(frame_truth), "ground_truth_vehicles": counts["gt_vehicles"],
        "vehicle_recall": ratio(counts["matched_vehicles"], counts["gt_vehicles"]),
        "ground_truth_plates": counts["gt"], "plate_predictions": counts["predictions"],
        "matched_detections": counts["matched"], "plate_recall": ratio(counts["matched"], counts["gt"]),
        "plate_precision": ratio(counts["matched"], counts["predictions"]),
        "ocr_labelled_plates": counts["ocr_gt"], "ocr_labelled_matches": counts["ocr_matched"],
        "ocr_exact_matches": counts["ocr_exact"], "ocr_exact_match": ratio(counts["ocr_exact"], counts["ocr_matched"]),
        "ocr_end_to_end_exact_match": ratio(counts["ocr_exact"], counts["ocr_gt"]),
        "character_error_rate": ratio(counts["ocr_edits"], counts["ocr_matched_chars"]),
        "end_to_end_character_error_rate": ratio(counts["ocr_end_to_end_edits"], counts["ocr_gt_chars"]),
        "evaluated_confident_reads": counts["confident_total"], "false_confident_reads": counts["confident_wrong"],
        "false_confident_read_rate": ratio(counts["confident_wrong"], counts["confident_total"]),
        "unmatched_confident_reads": counts["unmatched_reads"],
        "association_labelled_matches": counts["association"], "association_legacy_id_matches": legacy_id_matches,
        "association_accuracy": ratio(counts["association_correct"], counts["association"]),
        "track_level_exact_match": ratio(correct_tracks, len(eligible)), "track_level_evaluable_count": len(eligible),
        "track_false_confident_read_rate": ratio(track_wrong_reads, track_read_count),
        "matched_plate_track_id_switches": switches if trajectories else None,
        "unreadable_rate": ratio(counts["unreadable"], counts["predictions"]),
        "missing_candidate_rate": ratio(counts["candidate_missing"], counts["predictions"]),
        "mean_frame_latency_ms": latency, "estimated_pipeline_fps": 1000 / latency if latency else None,
        "iou_threshold": threshold,
    }
