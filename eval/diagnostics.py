"""Detection and recognition error attribution for exhaustively or partially labeled frames."""

from collections import Counter, defaultdict
from statistics import mean

import numpy as np

from eval.metrics import frame_key, iou, match_boxes, parse_box, ratio
from src.recognition.normalize import normalize_text


def edit_alignment(reference, predicted):
    """Deterministic minimum-edit alignment, preferring substitutions on tied paths."""
    table = np.zeros((len(reference) + 1, len(predicted) + 1), dtype=int)
    table[:, 0] = np.arange(len(reference) + 1)
    table[0, :] = np.arange(len(predicted) + 1)
    for i, expected in enumerate(reference, 1):
        for j, actual in enumerate(predicted, 1):
            table[i, j] = min(table[i-1, j] + 1, table[i, j-1] + 1,
                              table[i-1, j-1] + (expected != actual))
    i, j = len(reference), len(predicted)
    aligned = []
    while i or j:
        if i and j and table[i, j] == table[i-1, j-1] + (reference[i-1] != predicted[j-1]):
            aligned.append((reference[i-1], predicted[j-1]))
            i, j = i - 1, j - 1
        elif i and table[i, j] == table[i-1, j] + 1:
            aligned.append((reference[i-1], ""))
            i -= 1
        else:
            aligned.append(("", predicted[j-1]))
            j -= 1
    return list(reversed(aligned))


def _indexed(payload, truth):
    frames = defaultdict(lambda: ([], []))
    for row in truth:
        frames[frame_key(row)][0].append(row)
    for row in payload.get("observations", []):
        key = frame_key(row)
        if key in frames and row.get("plate_bbox_xyxy"):
            frames[key][1].append(row)
    return frames


def _ranked_ap(frames, cutoff, references):
    if not references:
        return None
    candidates = sorted(((row.get("plate_detection_confidence") or 0, key, index, row["plate_bbox_xyxy"])
                         for key, (_, rows) in frames.items() for index, row in enumerate(rows)),
                        key=lambda row: (-row[0], row[1], row[2]))
    available = {key: [(i, parse_box(row["plate_bbox_xyxy"])) for i, row in enumerate(rows)
                       if row.get("plate_bbox_xyxy")] for key, (rows, _) in frames.items()}
    matches = []
    for _, key, _, box in candidates:
        eligible = available[key]
        best = max(range(len(eligible)), key=lambda j: iou(box, eligible[j][1]), default=None)
        matched = best is not None and iou(box, eligible[best][1]) >= cutoff
        if matched:
            eligible.pop(best)
        matches.append(matched)
    hits = np.cumsum(matches)
    precision = hits / np.arange(1, len(hits) + 1)
    recall = hits / references
    return float(np.mean([precision[recall >= r].max(initial=0) for r in np.linspace(0, 1, 101)]))


def confidence_table(scored, labelled, thresholds=(0, .50, .70, .82, .90, .95)):
    """Selective OCR risk over reference matches with an available raw score."""
    output = []
    for threshold in thresholds:
        accepted = [(score, exact) for score, exact in scored if score >= threshold]
        correct = sum(exact for _, exact in accepted)
        output.append({"score_gte": threshold, "accepted": len(accepted), "exact": correct,
                       "coverage_all_labelled": ratio(len(accepted), labelled),
                       "precision": ratio(correct, len(accepted)), "recall_exact": ratio(correct, labelled),
                       "incorrect_rate": ratio(len(accepted) - correct, len(accepted))})
    bands = []
    for lower, upper in zip(thresholds, (*thresholds[1:], float("inf")), strict=True):
        band = [(score, exact) for score, exact in scored if lower <= score < upper]
        if band:
            bands.append({"score_gte": lower, "count": len(band), "mean_score": mean(s for s, _ in band),
                          "empirical_exact": mean(x for _, x in band)})
    return {"thresholds": output, "reliability_bins": bands, "scored": len(scored)}


def diagnose(payload, truth, threshold=.5):
    frames = _indexed(payload, truth)
    counts = Counter()
    overlap = []
    edit_pairs = Counter()
    scored = []
    cases = []
    exhaustive = all(str(row.get("boxes_exhaustive", "true")).lower() in ("1", "true", "yes") for row in truth)
    for (source, index), (annotations, predictions) in frames.items():
        annotations = [(row, parse_box(row.get("plate_bbox_xyxy"))) for row in annotations if row.get("plate_bbox_xyxy")]
        boxes = [row["plate_bbox_xyxy"] for row in predictions]
        pairs = dict(match_boxes([b for _, b in annotations], boxes, threshold))
        counts["predictions"] += len(predictions)
        for annotation_index, (reference, box) in enumerate(annotations):
            counts["references"] += 1
            candidate_index = pairs.get(annotation_index)
            best_iou = max((iou(box, pred_box) for pred_box in boxes), default=0)
            prediction = predictions[candidate_index] if candidate_index is not None else None
            matched_iou = iou(box, boxes[candidate_index]) if prediction else None
            expected = normalize_text(reference.get("ground_truth"))
            actual = normalize_text(prediction.get("ocr_normalized")) if prediction else ""
            if prediction is None:
                cause = ("overlapping_miss" if 0 < best_iou < threshold else
                         "assignment_miss" if best_iou >= threshold else "miss_no_overlap")
            else:
                overlap.append(matched_iou)
                cause = ("localized_unlabelled_text" if not expected else "exact" if expected == actual else
                         "no_candidate" if not actual else "incorrect_candidate")
            counts[cause] += 1
            if expected:
                counts["labelled_text"] += 1
                if prediction is not None:
                    alignment = edit_alignment(expected, actual)
                    edit_pairs.update(alignment)
                    score = prediction.get("ocr_raw_score")
                    if isinstance(score, (int, float)) and np.isfinite(score):
                        scored.append((score, expected == actual))
                    if prediction.get("recognition_status") == "READ":
                        counts["verified_reads"] += 1
                        counts["incorrect_verified_reads"] += actual != expected
            cases.append({"source": source, "frame_index": index, "annotation_index": annotation_index,
                          "ground_truth_box": box, "ground_truth_text": expected, "category": cause,
                          "matched_iou": matched_iou, "best_iou": best_iou, "prediction": prediction})
        used = set(pairs.values())
        for j, prediction in enumerate(predictions):
            if j in used:
                continue
            duplicate = any(iou(boxes[j], annotations[k][1]) >= threshold for k in pairs)
            category = "duplicate_overlap" if duplicate else "unmatched_proposal"
            counts[category] += 1
            if exhaustive and prediction.get("recognition_status") == "READ":
                counts["background_false_reads"] += 1
            cases.append({"source": source, "frame_index": index, "category": category, "prediction": prediction})
    hits = len(overlap)
    precision = ratio(hits, counts["predictions"])
    recall = ratio(hits, counts["references"])
    ap = [_ranked_ap(frames, .50 + i * .05, counts["references"]) for i in range(10)]
    # Rankings on partial labels quantify annotation matching. They do not estimate scene precision.
    detection = {"reference_scope": "exhaustive" if exhaustive else "partial_annotations",
                 "references": counts["references"], "proposals": counts["predictions"], "matched": hits,
                 "annotation_match_precision": precision, "recall": recall,
                 "annotation_match_f1": ratio(2 * hits, counts["references"] + counts["predictions"]),
                 "annotation_ap50": ap[0], "annotation_ap50_95": mean(ap) if ap[0] is not None else None,
                 "ranked_ap_prediction_floor": "submitted_proposals", "mean_localization_iou": mean(overlap) if hits else None,
                 "misses": counts["references"] - hits, "false_detections": counts["predictions"] - hits if exhaustive else None}
    chars = sum(n for (a, _), n in edit_pairs.items() if a)
    correct_chars = sum(n for (a, b), n in edit_pairs.items() if a == b)
    ocr = {"counts": dict(counts), "matched_character_accuracy": ratio(correct_chars, chars),
           "edit_confusion": [{"reference": a, "prediction": b, "count": n} for (a, b), n in edit_pairs.most_common()],
           "selective_risk": confidence_table(scored, counts["labelled_text"]),
           "incorrect_read_rate": ratio(counts["incorrect_verified_reads"], counts["verified_reads"]),
           "background_false_reads": counts["background_false_reads"] if exhaustive else None}
    return {"detection": detection, "ocr": ocr}, cases
