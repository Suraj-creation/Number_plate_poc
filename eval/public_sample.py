"""Score existing DataCluster Labs annotations without creating new labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from time import perf_counter
import xml.etree.ElementTree as ET

from eval.metrics import evaluate_rows, levenshtein, match_boxes, parse_box
from src.recognition.normalize import normalize_text


def publisher_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "images").glob("*.jpg")):
        document = ET.parse(root / "Annotations" / f"{path.stem}.xml").getroot()
        for plate in document.findall("object"):
            box = plate.find("bndbox")
            if box is None:
                continue
            coords = ",".join(box.findtext(key) for key in ("xmin", "ymin", "xmax", "ymax"))
            text = next((item.findtext("value") for item in plate.findall("attributes/attribute")
                         if item.findtext("name") == "number_plate_text"), None)
            rows.append({"source": path.name, "frame_index": 0, "plate_bbox_xyxy": coords, "ground_truth": text,
                         "boxes_exhaustive": "false"})
    return rows


def score_directories(gt, directories):
    intended_sources = {r["source"] for r in gt}
    results = {}
    for directory in directories:
        observations, latencies = [], []
        sources = set()
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data.get("run_metadata"):
                continue
            sources.add(Path(data["run_metadata"]["source"]).name)
            for row in data["observations"]:
                observations.append(dict(row, source=Path(row["source"]).name))
            latencies.append(data["run_metadata"]["performance_ms"]["total"]["mean"])
        if sources != intended_sources:
            raise ValueError(f"{directory.name} did not process the same {len(intended_sources)} sources.")
        result = evaluate_rows({"observations": observations}, gt)
        result["mean_frame_latency_ms"] = round(statistics.mean(latencies), 2)
        result["statuses"] = dict(Counter(row["recognition_status"] for row in observations))
        result["by_source_plate_width"] = {}
        tiers = (("under_96_px", 0, 96), ("96_to_159_px", 96, 160), ("160_px_and_up", 160, float("inf")))
        for name, minimum, maximum in tiers:
            subset = []
            for row in gt:
                x1, _, x2, _ = parse_box(row["plate_bbox_xyxy"])
                if minimum <= x2 - x1 < maximum:
                    subset.append(row)
            if subset:
                size_metrics = evaluate_rows({"observations": observations}, subset)
                result["by_source_plate_width"][name] = {
                    field: size_metrics[field] for field in ("ground_truth_plates", "matched_detections",
                        "plate_recall", "ocr_labelled_plates", "ocr_labelled_matches", "ocr_exact_matches")
                }
        result["matched_by_detection_pass"] = {}
        for name in ("full_frame", "vehicle_crop", "tile"):
            coverage = 0
            for source in sources:
                boxes = [parse_box(g["plate_bbox_xyxy"]) for g in gt if g["source"] == source]
                detected = {tuple(hit["bbox"]) for row in observations if row["source"] == source
                            for hit in row.get("plate_detection_passes", []) if hit["source_pass"] == name}
                coverage += len(match_boxes(boxes, sorted(detected), 0.5))
            result["matched_by_detection_pass"][name] = coverage if any(r.get("plate_detection_passes") for r in observations) else None
        results[directory.name] = result
    return results


def score_readers(root: Path, gt: list[dict], device: str, research: bool, paddle_enabled: bool) -> dict:
    from src.utils.config import configure_caches
    configure_caches()

    from src.ingest.images import read_image
    from src.pipeline.geometry import expand_box
    from src.recognition.crop_quality import assess_crop
    from src.recognition.evidence import recovery_views
    from src.recognition.fast_plate_ocr import FastPlateOCR

    primary = FastPlateOCR(device=device)
    paddle = None
    if paddle_enabled:
        from src.recognition.paddleocr_fallback import PaddleOCRFallback
        paddle = PaddleOCRFallback()
    if research:
        from src.recognition.indian_trocr_optional import IndianTrOCR
        challenger = IndianTrOCR(device=device)
    latencies = {}
    counts = Counter()
    widths = []
    total_chars = 0
    for row in gt:
        truth = normalize_text(row["ground_truth"])
        if not truth:
            continue
        frame = read_image(root / "images" / row["source"])
        box = parse_box(row["plate_bbox_xyxy"])
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = expand_box(box, 0.08, width, height)
        crop = frame[y1:y2, x1:x2]
        quality = assess_crop(crop)
        widths.append(round(box[2] - box[0]))
        total_chars += len(truth)
        started = perf_counter()
        candidate = normalize_text(primary.predict(crop)["raw"])
        latencies.setdefault("cct", []).append((perf_counter() - started) * 1000)
        counts["cct_exact"] += candidate == truth
        counts["cct_edits"] += levenshtein(candidate, truth)
        variants = recovery_views(crop, quality)
        recovered = False
        for label, view in variants:
            key = "flattened" if label.startswith("rows") else label
            view_text = normalize_text(primary.predict(view)["raw"])
            counts[f"{key}_attempts"] += 1
            counts[f"{key}_exact"] += view_text == truth
            counts[f"{key}_losses"] += candidate == truth and view_text != truth
            recovered |= view_text == truth
        counts["correct_alternative_when_original_wrong"] += recovered and candidate != truth
        if research:
            started = perf_counter()
            other = normalize_text(challenger.predict(crop)["raw"])
            latencies.setdefault("trocr", []).append((perf_counter() - started) * 1000)
            counts["trocr_exact"] += other == truth
            counts["trocr_edits"] += levenshtein(other, truth)
            counts["readers_agree_correct"] += other == candidate == truth
            counts["readers_agree_wrong"] += other == candidate and other != truth
        if paddle:
            started = perf_counter()
            result = paddle.predict(crop)
            other = normalize_text(result["raw"])
            latencies.setdefault("paddle", []).append((perf_counter() - started) * 1000)
            counts["paddle_exact"] += other == truth
            counts["paddle_edits"] += levenshtein(other, truth)
            counts["paddle_agrees_with_cct_correct"] += other == candidate == truth
            counts["paddle_agrees_with_cct_wrong"] += other == candidate and other != truth
            if (result["raw_score"] or 0) >= 0.85:
                counts["paddle_above_gate"] += 1
                counts["paddle_exact_above_gate"] += other == truth
    if paddle:
        paddle.close()
    return {"publisher_text_labels": len(widths), "label_characters": total_chars,
            "width_range_px": [min(widths), max(widths)] if widths else None,
            "counts": dict(counts), "mean_latency_ms": {k: round(statistics.mean(v), 2) for k, v in latencies.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--predictions", nargs="*", type=Path, default=[])
    parser.add_argument("--score-readers", action="store_true")
    parser.add_argument("--research-ocr", action="store_true")
    parser.add_argument("--paddle-ocr", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    gt = publisher_rows(args.sample)
    output = score_directories(gt, args.predictions)
    if args.score_readers:
        output["ocr_on_publisher_boxes"] = score_readers(args.sample, gt, args.device, args.research_ocr, args.paddle_ocr)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
