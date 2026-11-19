"""Render a local, private review gallery from the failure analysis."""

from __future__ import annotations

import argparse
from base64 import b64encode
from html import escape
import json
from pathlib import Path

import cv2

from eval.production_benchmark import PROTOCOL
from src.pipeline.geometry import expand_box
from src.recognition.normalize import normalize_text


def picture(image, label, max_width):
    if image is None or not image.size:
        return ""
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, max(1, round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Unable to encode a gallery image.")
    uri = b64encode(encoded).decode("ascii")
    return f'<figure><img loading="lazy" src="data:image/jpeg;base64,{uri}" alt="{escape(label)}"><figcaption>{escape(label)}</figcaption></figure>'


def rectangle(image, box, color):
    x1, y1, x2, y2 = map(round, box)
    cv2.rectangle(image, (x1, y1), (x2-1, y2-1), color, max(2, image.shape[1] // 380))


def render(dataset, workspace, output):
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    protocol = json.loads(PROTOCOL.read_text())
    awiros = {}
    for split in ("validation", "test"):
        run = json.loads((workspace / split / "awiros.json").read_text())
        awiros.update(run["detected/original"]["predictions"])
    cases = json.loads((workspace / "private_cases.json").read_text())
    cases.sort(key=lambda row: (row["source"], "annotation_index" not in row, row.get("annotation_index", 0)))
    counts = {"better": 0, "worse": 0, "same": 0, "other": 0}
    with output.open("x", encoding="utf-8") as stream:
        output.chmod(0o600)
        stream.write("""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Indian ANPR error review</title>
<style>body{font:16px system-ui;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#172e3d;background:#f2f5f7}
h1{font-size:2rem} h2{font-size:1.15rem} article{background:#fff;border:1px solid #c2d3dc;padding:.7rem 1.2rem;margin:1rem 0;border-radius:.6rem}
figure{margin:.45rem 0} img{max-width:100%;height:auto;background:#e8eff2} .panel{display:grid;grid-template-columns:2fr 1fr;gap:1.25rem}
b{color:#075e70} figcaption{font-size:.9rem} @media(max-width:680px){.panel{display:block}}</style>
<h1>Indian ANPR error review</h1><p>Green shows the publisher box and orange shows the matched or unmatched proposal.
The reference label is shown separately from both OCR candidates. Awiros scores do not authorize a READ.</p>
<label for="filter">Filter outcomes </label><select id="filter"><option value="all">All cases</option><option value="better">Awiros corrected CCT</option><option value="worse">CCT correct, Awiros wrong</option><option value="same">Readers tied on exact match</option><option value="other">Miss, unlabelled text or unmatched proposal</option></select>
<script>document.getElementById('filter').onchange = e => document.querySelectorAll('article').forEach(a => a.hidden = e.target.value !== 'all' && a.dataset.result !== e.target.value);</script>
""")
        for case in cases:
            source = case["source"]
            prediction = case["prediction"] or {}
            gt = case.get("ground_truth_box")
            gt_text = case.get("ground_truth_text") or ""
            candidate = prediction.get("ocr_normalized") or ""
            alternative = awiros.get(f'{Path(source).stem}_{case.get("annotation_index")}.png', {})
            awiros_text = normalize_text(alternative.get("raw"))
            comparison = "other"
            if alternative and gt_text:
                comparison = ("better" if awiros_text == gt_text and candidate != gt_text else
                              "worse" if candidate == gt_text and awiros_text != gt_text else "same")
            counts[comparison] += 1
            frame = cv2.imread(str(dataset / "images" / source))
            annotated = frame.copy()
            h, w = frame.shape[:2]
            source_crop = prediction.get("plate_crop_bbox_xyxy")
            if gt:
                reference_crop = expand_box(gt, .08, w, h)
                rectangle(annotated, gt, (70, 255, 70))
            if prediction.get("plate_bbox_xyxy"):
                rectangle(annotated, prediction["plate_bbox_xyxy"], (5, 145, 255))
            split = "validation" if source in protocol["validation"] else "held-out"
            stream.write(f'<article data-result="{comparison}"><h2>{escape(source)} · {split} · {escape(case["category"])}</h2>')
            stream.write(f'<p>Ground-truth text <b>{escape(gt_text or "Unavailable")}</b> · Best box IoU {case.get("best_iou", 0):.3f}</p>')
            stream.write(f'<p>CCT <b>{escape(candidate or "—")}</b> · raw score {prediction.get("ocr_raw_score")} · {escape(prediction.get("recognition_status") or "No matched detection")}</p>')
            stream.write(f'<p>Awiros <b>{escape(awiros_text or "—")}</b> · raw score {alternative.get("raw_score")}</p>')
            stream.write('<div class="panel">' + picture(annotated, "Original frame with box overlays", 680) + '<div>')
            if source_crop:
                x1, y1, x2, y2 = source_crop
                stream.write(picture(frame[y1:y2, x1:x2], "Original predicted crop used by both OCR readers", 360))
            if gt:
                x1, y1, x2, y2 = reference_crop
                stream.write(picture(frame[y1:y2, x1:x2], "Publisher-box crop with the same margin", 360))
            stream.write("</div></div></article>\n")
    print(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    render(args.dataset, args.workspace, args.output)


if __name__ == "__main__":
    main()
