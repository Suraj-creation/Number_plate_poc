"""Prepare equivalent crops and score a frozen pretrained reader against CCT."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2

from eval.metrics import levenshtein, match_boxes, parse_box
from eval.public_sample import publisher_rows
from src.pipeline.geometry import expand_box
from src.recognition.normalize import normalize_text


def prepare(root, runs, directory):
    truth = defaultdict(list)
    for row in publisher_rows(root):
        truth[row["source"]].append(row)
    for variant in ("publisher_original", "publisher_tight", "detected"):
        (directory / variant).mkdir(parents=True, exist_ok=True)
    index = []
    for name, labels in truth.items():
        raw = json.loads((runs / Path(name).with_suffix(".json")).read_text())
        plates = [row for row in raw["observations"] if row.get("plate_bbox_xyxy")]
        pairs = dict(match_boxes([parse_box(row["plate_bbox_xyxy"]) for row in labels],
                                 [row["plate_bbox_xyxy"] for row in plates], .5))
        frame = cv2.imread(str(root / "images" / name))
        height, width = frame.shape[:2]

        def crop(variant, key, box):
            x1, y1, x2, y2 = box
            if not cv2.imwrite(str(directory / variant / key), frame[y1:y2, x1:x2]):
                raise OSError(f"Couldn't write temporary crop {key}")

        for i, row in enumerate(labels):
            label = normalize_text(row.get("ground_truth"))
            if not label:
                continue
            bbox = parse_box(row["plate_bbox_xyxy"])
            key = f"{Path(name).stem}_{i}.png"
            crop("publisher_original", key, expand_box(bbox, .08, width, height))
            crop("publisher_tight", key, expand_box(bbox, 0, width, height))
            item = {"key": key, "source": name, "truth": label, "detected": i in pairs}
            if i in pairs:
                plate = plates[pairs[i]]
                crop("detected", key, plate["plate_crop_bbox_xyxy"])
                item.update(cct_candidate=plate["ocr_normalized"], cct_status=plate["recognition_status"])
            index.append(item)
    (directory / "private_index.json").write_text(json.dumps(index))


def read_predictions(path):
    items = json.loads(path.read_text())
    return {i["image"]: (normalize_text(i["prediction"]), i.get("confidence")) for i in items}


def score(directory):
    source = json.loads((directory / "private_index.json").read_text())
    providers = ["cct", "awiros", "paddle", "trocr"]
    output = {}
    for variant in ("publisher_original", "publisher_tight", "detected"):
        scores = defaultdict(Counter)
        readers = {provider: read_predictions(path) for provider in providers
                   if (path := directory / f"{provider}_{variant}.json").is_file()}
        labelled = [row for row in source if variant != "detected" or row["detected"]]
        expected = {row["key"] for row in labelled}
        for name, values in readers.items():
            if expected != values.keys():
                raise ValueError(f"{name} did not process every intended {variant} crop.")
        for row in labelled:
            label = row["truth"]
            candidates = {name: values[row["key"]] for name, values in readers.items()}
            if variant == "detected":
                candidates["cct"] = (normalize_text(row["cct_candidate"]), None)
            if "awiros" in candidates and "paddle" in candidates:
                awiros, _ = candidates["awiros"]
                other, _ = candidates["paddle"]
                candidates["awiros_paddle_agreement"] = (awiros if awiros == other else "", None)
            for name, (prediction, confidence) in candidates.items():
                scores[name].update(total=1, exact=int(prediction == label),
                                    edits=levenshtein(prediction, label), reference_chars=len(label),
                                    emitted=int(bool(prediction)), wrong_emitted=int(bool(prediction) and prediction != label))
        comparisons = {}
        for name, values in readers.items():
            if variant == "detected":
                wins = losses = 0
                for row in labelled:
                    old = normalize_text(row["cct_candidate"]) == row["truth"]
                    new = values[row["key"]][0] == row["truth"]
                    wins += new and not old
                    losses += old and not new
                comparisons[name] = {"paired_wins": wins, "paired_losses": losses}
        output[variant] = {"matched_crops": len(labelled), "all_text_labels": len(source),
                           "results": {k: dict(v) for k, v in scores.items()}, "paired_vs_cct": comparisons}
    return output


def cct(directory, device):
    from src.utils.config import configure_caches
    configure_caches()
    from src.recognition.fast_plate_ocr import FastPlateOCR

    model = FastPlateOCR(device=device)
    for variant in ("publisher_original", "publisher_tight"):
        files = sorted((directory / variant).glob("*.png"))
        results = model.predict_many([cv2.imread(str(path)) for path in files])
        out = [{"image": path.name, "prediction": pred["raw"], "confidence": pred["raw_score"]}
               for path, pred in zip(files, results, strict=True)]
        (directory / f"cct_{variant}.json").write_text(json.dumps(out))


def secondary(directory, device, engine):
    from src.utils.config import configure_caches
    configure_caches()
    if engine == "paddle":
        from src.recognition.paddleocr_fallback import PaddleOCRFallback
        model = PaddleOCRFallback()
    else:
        from src.recognition.indian_trocr_optional import IndianTrOCR
        model = IndianTrOCR(device=device)
    try:
        for variant in ("publisher_original", "detected"):
            results = []
            for path in sorted((directory / variant).glob("*.png")):
                pred = model.predict(cv2.imread(str(path)))
                results.append({"image": path.name, "prediction": pred["raw"], "confidence": pred["raw_score"]})
            (directory / f"{engine}_{variant}.json").write_text(json.dumps(results))
    finally:
        if engine == "paddle":
            model.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "cct", "paddle", "trocr", "score"))
    parser.add_argument("--crops", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.dataset, args.runs, args.crops)
    elif args.action == "cct":
        cct(args.crops, args.device)
    elif args.action in ("paddle", "trocr"):
        secondary(args.crops, args.device, args.action)
    else:
        print(json.dumps(score(args.crops), indent=2))


if __name__ == "__main__":
    main()
