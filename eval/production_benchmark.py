"""Replay the pinned still-image experiment and write aggregate and private diagnostic results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np

from eval.diagnostics import diagnose
from eval.metrics import evaluate_rows, levenshtein, match_boxes, parse_box
from eval.public_sample import publisher_rows
from src.pipeline.geometry import expand_box
from src.recognition.normalize import normalize_text
from src.recognition.plate_views import plate_view

VIEWS = ("original", "tight", "wide", "clahe", "bicubic", "unsharp", "deskew", "adaptive", "denoise")
PROTOCOL = Path(__file__).with_name("protocols") / "india_round2.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path, "w", encoding="utf-8", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def infer(dataset, workspace, profile):
    from src.pipeline.frame_pipeline import FramePipeline
    from src.utils.config import apply_profile, load_config

    cfg = apply_profile(load_config(), profile)
    rows, analyzed = [], []
    started = perf_counter()
    pipe = FramePipeline(cfg)
    try:
        for path in sorted((dataset / "images").glob("*.jpg")):
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Could not read {path.name}")
            pipe.frame_index = 0
            _, predictions, _ = pipe.process(frame, path.name)
            rows.extend(predictions)
            analyzed.append(path.name)
    finally:
        pipe.close()
    save(workspace / f"{profile}.json", {"observations": rows, "sources": analyzed,
                                         "seconds": perf_counter() - started})


def prepare(dataset, workspace, split):
    manifest = json.loads(PROTOCOL.read_text())
    selected = set(manifest[split])
    baseline = json.loads((workspace / "standard.json").read_text())
    if not selected <= set(baseline["sources"]):
        raise ValueError("Baseline skipped a designated sample.")
    truth = defaultdict(list)
    observations = defaultdict(list)
    for row in publisher_rows(dataset):
        if row["source"] in selected:
            truth[row["source"]].append(row)
    if set(truth) != selected:
        raise ValueError("Dataset differs from the pinned protocol.")
    for row in baseline["observations"]:
        if row.get("plate_bbox_xyxy") and row["source"] in selected:
            observations[row["source"]].append(row)
    views = VIEWS if split == "validation" else ("original",)
    keys = {}
    for source, annotations in truth.items():
        predictions = observations[source]
        boxes = [parse_box(row["plate_bbox_xyxy"]) for row in annotations]
        paired = dict(match_boxes(boxes, [row["plate_bbox_xyxy"] for row in predictions], .5))
        frame = cv2.imread(str(dataset / "images" / source))
        height, width = frame.shape[:2]
        for i, label in enumerate(annotations):
            expected = normalize_text(label["ground_truth"])
            if not expected:
                continue
            key = f"{Path(source).stem}_{i}.png"
            keys[key] = {"truth": expected, "matched": i in paired}
            inputs = {"oracle": boxes[i]}
            if i in paired:
                inputs["detected"] = predictions[paired[i]]["plate_bbox_xyxy"]
            for basis, box in inputs.items():
                for view in views:
                    margin = {"tight": 0, "wide": .14}.get(view, .08)
                    x1, y1, x2, y2 = expand_box(box, margin, width, height)
                    crop = plate_view(frame[y1:y2, x1:x2], "original" if view in ("tight", "wide") else view)
                    target = workspace / split / basis / view / key
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(target), crop):
                        raise OSError(f"Failed to write {target.name}")
    save(workspace / split / "private_ground_truth.json", keys)


def read_variants(workspace, split, engine, device):
    base = workspace / split
    if engine == "cct":
        from src.utils.config import configure_caches
        from src.recognition.fast_plate_ocr import FastPlateOCR

        configure_caches()
        predict = FastPlateOCR(device=device).predict_many
    else:
        from src.recognition.awiros_optional import load_awiros

        predict = load_awiros()
    results = {}
    for kind in ("detected", "oracle"):
        variants = VIEWS if split == "validation" else ("original",)
        for name in variants:
            variant = base / kind / name
            files = sorted(variant.glob("*.png"))
            predictions = {}
            started = perf_counter()
            for k in range(0, len(files), 8):
                batch = files[k:k+8]
                texts = predict([cv2.imread(str(file)) for file in batch])
                predictions.update((file.name, text) for file, text in zip(batch, texts, strict=True))
            results[f"{kind}/{variant.name}"] = {"predictions": predictions, "seconds": perf_counter() - started}
    save(base / f"{engine}.json", results)


def summarize_readers(workspace, split):
    cohort = workspace / split
    reference = json.loads((cohort / "private_ground_truth.json").read_text())
    models = {}
    for model in ("cct", "awiros"):
        tables = json.loads((cohort / f"{model}.json").read_text())
        models[model] = {}
        for key, run in tables.items():
            if split == "test" and key.split("/")[-1] != "original":
                continue
            preds = run["predictions"]
            expected_keys = {k for k, target in reference.items() if key.startswith("oracle") or target["matched"]}
            if preds.keys() != expected_keys:
                raise ValueError(f"Inconsistent crop inputs for {model}/{key}.")
            matched = Counter()
            all_labels = Counter()
            for filename, target in reference.items():
                truth = target["truth"]
                actual = normalize_text(preds.get(filename, {}).get("raw"))
                edits = levenshtein(actual, truth)
                all_labels.update({"total": 1, "exact": actual == truth, "edits": edits, "chars": len(truth)})
                if filename in preds:
                    matched.update({"total": 1, "exact": actual == truth, "edits": edits, "chars": len(truth),
                                    "abstained": not actual})
            models[model][key] = {"on_crops": dict(matched), "all_text_labels": dict(all_labels),
                                  "cer_on_crops": matched["edits"] / matched["chars"],
                                  "end_to_end_cer": all_labels["edits"] / all_labels["chars"],
                                  "inference_ms_per_crop": run["seconds"] * 1000 / len(preds)}
    pairs = Counter()
    base = json.loads((cohort / "cct.json").read_text())["detected/original"]["predictions"]
    challenger = json.loads((cohort / "awiros.json").read_text())["detected/original"]["predictions"]
    for key, tag in reference.items():
        a = normalize_text(base.get(key, {}).get("raw"))
        b = normalize_text(challenger.get(key, {}).get("raw"))
        pairs[(a == tag["truth"], b == tag["truth"])] += 1
    return {"readers": models, "paired_correct": {f"cct_{a}_awiros_{b}": n for (a, b), n in pairs.items()}}


def characterize(dataset, cases):
    """Stratify attributes measurable directly from existing annotations and source pixels."""
    frames = {}
    buckets = defaultdict(Counter)
    for case in cases:
        if "ground_truth_box" not in case:
            continue
        name = case["source"]
        if name not in frames:
            frames.clear()
            frames[name] = cv2.imread(str(dataset / "images" / name), cv2.IMREAD_GRAYSCALE)
        gray = frames[name]
        height, width = gray.shape
        x1, y1, x2, y2 = expand_box(case["ground_truth_box"], 0, width, height)
        roi = gray[y1:y2, x1:x2]
        luminance = float(np.median(roi))
        sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        plate_width = x2 - x1
        attributes = [
            "width/<96" if plate_width < 96 else "width/96–159" if plate_width < 160 else "width/160+",
            "luminance/<70" if luminance < 70 else "luminance/70+",
            "laplacian/<40" if sharpness < 40 else "laplacian/40+",
            "aspect/<2.4" if plate_width / (y2 - y1) < 2.4 else "aspect/2.4+",
        ]
        case["pixel_proxies"] = {"luminance": luminance, "laplacian": sharpness, "width": plate_width,
                                 "height": y2 - y1}
        for key in attributes:
            buckets[key].update({"boxes": 1, "matched": case["matched_iou"] is not None,
                                 "text_labels": bool(case["ground_truth_text"]),
                                 "exact": case["category"] == "exact"})
    return {key: dict(counts) for key, counts in sorted(buckets.items())}


def score(dataset, workspace, output):
    baseline = json.loads((workspace / "standard.json").read_text())
    truth = publisher_rows(dataset)
    raw_results, cases = diagnose(baseline, truth)
    metrics = {"baseline": evaluate_rows(baseline, truth), "diagnostics": raw_results,
               "pixel_proxy_strata": characterize(dataset, cases), "splits": {}}
    save(workspace / "private_cases.json", cases)
    manifest = json.loads(PROTOCOL.read_text())
    for split in ("validation", "test"):
        rows = [r for r in truth if r["source"] in set(manifest[split])]
        result, _ = diagnose(baseline, rows)
        metrics["splits"][split] = {"baseline": evaluate_rows(baseline, rows), "diagnostics": result,
                                    **summarize_readers(workspace, split)}
    save(output, metrics)


def import_exports(directory, workspace):
    sources = []
    observations = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        metadata = payload.get("run_metadata", {})
        if metadata.get("media_type") != "image":
            continue
        source = Path(metadata["source"]).name
        sources.append(source)
        observations.extend(dict(row, source=source) for row in payload["observations"])
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("Provide one completed still-image export per publisher image.")
    save(workspace / "standard.json", {"sources": sources, "observations": observations})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("infer", "import-exports", "prepare", "read", "score"))
    parser.add_argument("--exports", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--engine", choices=("cct", "awiros"), default="cct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", choices=("standard", "recovery"), default="standard")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.action == "read":
        read_variants(args.workspace, args.split, args.engine, args.device)
        return
    if args.action == "import-exports":
        if args.exports is None:
            parser.error("import-exports requires --exports")
        import_exports(args.exports, args.workspace)
        return
    if args.dataset is None:
        parser.error("--dataset is required for infer, prepare and score")
    if args.action == "infer":
        infer(args.dataset, args.workspace, args.profile)
    elif args.action == "prepare":
        prepare(args.dataset, args.workspace, args.split)
    elif args.output is None:
        parser.error("score requires --output")
    else:
        score(args.dataset, args.workspace, args.output)


if __name__ == "__main__":
    main()
