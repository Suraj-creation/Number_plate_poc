"""Compare OCR readers on identical target-matched FANVID video crops."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from eval.metrics import levenshtein
from src.recognition.normalize import normalize_text


def prepare(private, media, output, index_file, resolution):
    from eval.production_benchmark import save

    data = json.loads(private.read_text())
    matched = [case for case in data["cases"] if case.get("ground_truth_text") and case.get("matched_iou")]
    by_source = defaultdict(list)
    for case in matched:
        by_source[case["source"]].append(case)
    sample_key = {}
    with index_file.open(newline="", encoding="utf-8-sig") as stream:
        clips = {f"{r['Region']}_{r['Clip ID']}": r for r in csv.DictReader(stream) if r["Split"] == "test"}
    for source, cases in by_source.items():
        clip = clips[source]
        first = int(clip["Start frame"])
        width, height = (int(clip[f"Target {resolution} {key}"]) for key in ("Width", "Height"))
        url_id = parse_qs(urlparse(clip["You_Tube_URL"]).query)["v"][0]
        filename = f"fanvid-h264-{url_id}.mp4"
        capture = cv2.VideoCapture(str(media / filename), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
        try:
            step = round(capture.get(cv2.CAP_PROP_FPS) / 6)
            if not capture.isOpened() or not step:
                raise OSError(f"Cannot read the source for {source}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            for i, case in enumerate(cases):
                position = first + case["frame_index"] * step
                while capture.get(cv2.CAP_PROP_POS_FRAMES) < position:
                    if not capture.grab():
                        raise OSError(f"Cannot decode the target in {source}")
                ok, frame = capture.read()
                if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != position + 1:
                    raise OSError(f"Cannot retrieve the exact target in {source}")
                if frame.shape[1::-1] != (width, height):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                observation = case["prediction"]
                x1, y1, x2, y2 = observation["plate_crop_bbox_xyxy"]
                crop = frame[y1:y2, x1:x2]
                key = f"{source}_{i}.png"
                output.mkdir(parents=True, exist_ok=True, mode=0o700)
                if not cv2.imwrite(str(output / key), crop):
                    raise OSError(f"Cannot save the target crop of {source}")
                sample_key[key] = {"source": source, "text": case["ground_truth_text"],
                                   "cct": observation["ocr_normalized"]}
        finally:
            capture.release()
    save(output / "index.json", {"sample_key": sample_key, "references": sum(bool(c.get("ground_truth_text"))
                               for c in data["cases"]), "reference_chars": sum(len(c.get("ground_truth_text", ""))
                               for c in data["cases"])})


def score(directory, output):
    from src.recognition.awiros_optional import load_awiros

    stats = {"cct": Counter(), "awiros": Counter()}
    by_clip = defaultdict(Counter)
    index = json.loads((directory / "index.json").read_text())
    cases = index["sample_key"]
    names = sorted(cases)
    paired = Counter()
    predict = load_awiros()
    for start in range(0, len(names), 8):
        batch = names[start:start+8]
        results = predict([cv2.imread(str(directory / name)) for name in batch])
        for name, result in zip(batch, results, strict=True):
            entry = cases[name]
            candidates = {"cct": normalize_text(entry["cct"]), "awiros": normalize_text(result["raw"])}
            paired[f"cct_{candidates['cct'] == entry['text']}_awiros_{candidates['awiros'] == entry['text']}"] += 1
            for model, candidate in candidates.items():
                expected = entry["text"]
                outcome = {"frames": 1, "exact": candidate == expected,
                           "edits": levenshtein(candidate, expected), "chars": len(expected)}
                stats[model].update(outcome)
                by_clip[f"{entry['source']}/{model}"].update(outcome)
    for result in stats.values():
        result["end_to_end_edits"] = result["edits"] + index["reference_chars"] - result["chars"]
    output.write_text(json.dumps({"reference_frames": index["references"], "reference_chars": index["reference_chars"],
                       "paired_matched_frames": len(cases), "by_reader": stats, "by_clip": by_clip,
                       "paired_correct": paired}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    for key in ("private", "media", "output", "directory", "index"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--resolution", choices=("HR", "LR"), default="HR")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.private, args.media, args.directory, args.index, args.resolution)
    else:
        score(args.directory, args.output)


if __name__ == "__main__":
    main()
