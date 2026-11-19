"""Evaluate the fixed India test clips in the FANVID publisher index."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlparse

import cv2

from eval.diagnostics import diagnose
from eval.metrics import evaluate_rows
from eval.production_benchmark import save
from src.utils.config import load_config

REGIONS = {"Mumbai", "Bangalore"}
TRANSPOSED_COLUMNS = {f"Bangalore_{i}" for i in range(4)}


def evaluate(index_path, annotations_path, media, cfg, resolution):
    from src.pipeline.frame_pipeline import FramePipeline

    with index_path.open(newline="", encoding="utf-8-sig") as stream:
        clips = [row for row in csv.DictReader(stream) if row["Region"] in REGIONS and row["Split"] == "test"]
    if len(clips) != 9:
        raise ValueError("Expected nine India clips from the pinned FANVID test index.")
    with annotations_path.open(newline="", encoding="utf-8-sig") as stream:
        references = {(r["VideoID"], r["Clip ID"], int(r["FrameNo"])): r for r in csv.DictReader(stream)}
    per_region = defaultdict(list)
    per_vehicle_class = defaultdict(list)
    reports = []
    payload = {"observations": [], "tracks": []}
    analyzed = 0
    started = perf_counter()
    pipe = FramePipeline(cfg, enable_tracking=True)
    try:
        for clip in clips:
            from src.tracking.bytetrack import ByteTracker
            from src.tracking.track_state import TrackState

            url_id = parse_qs(urlparse(clip["You_Tube_URL"]).query)["v"][0]
            file = media / f"fanvid-h264-{url_id}.mp4"
            source = f"{clip['Region']}_{clip['Clip ID']}"
            source_id = clip["Video ID"].split("&")[0]
            pipe.tracker = ByteTracker(cfg["tracker_track_activation_threshold"], cfg["tracker_lost_track_buffer"],
                                       cfg["tracker_match_threshold"])
            pipe.track_state = TrackState(cfg["track_history_size"], cfg)
            pipe.finished_tracks.clear()
            capture = cv2.VideoCapture(str(file), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
            try:
                if not capture.isOpened():
                    raise OSError(f"Cannot open the published source for {source}.")
                fps = capture.get(cv2.CAP_PROP_FPS)
                nominal = float(clip["FPS"])
                native = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                dimensions = (int(clip[f"Target {resolution} Width"]), int(clip[f"Target {resolution} Height"]))
                if abs(fps - nominal) > 1.1 or abs(native[0] / native[1] - dimensions[0] / dimensions[1]) > .02:
                    raise ValueError(f"{source} encoding {native} @ {fps:g} FPS differs from the annotation geometry.")
                step = round(fps / 6)
                first, last = int(clip["Start frame"]), int(clip["End frame"])
                capture.set(cv2.CAP_PROP_POS_FRAMES, first)
                clip_analyzed = 0
                for frame_index in range(first, last + 1):
                    if (frame_index - first) % step:
                        if not capture.grab():
                            raise OSError(f"Decode stopped in {source}.")
                        continue
                    ok, frame = capture.read()
                    if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame_index + 1:
                        raise OSError(f"Could not align published frame {frame_index} of {source}.")
                    if frame.shape[1::-1] != dimensions:
                        frame = cv2.resize(frame, dimensions, interpolation=cv2.INTER_CUBIC)
                    pipe.frame_index = clip_analyzed
                    timestamp = round((frame_index - first) * 1000 / fps)
                    _, observations, _ = pipe.process(frame, source, timestamp)
                    payload["observations"].extend(observations)
                    key = (source_id, clip["Clip ID"], frame_index)
                    if key in references:
                        annotation = references[key]
                        coords = [float(annotation[k]) for k in ("BoxLeft", "BoxTop", "BoxRight", "BoxBottom")]
                        if source in TRANSPOSED_COLUMNS:
                            # All four Bangalore clips store the top and right columns in reverse order.
                            coords[1], coords[2] = coords[2], coords[1]
                        if not (0 <= coords[0] < coords[2] <= dimensions[0] + 1
                                and 0 <= coords[1] < coords[3] <= dimensions[1] + 1):
                            raise ValueError(f"Invalid published box in {source}.")
                        row = {"source": source, "frame_index": clip_analyzed,
                            "plate_bbox_xyxy": ",".join(str(round(x, 3)) for x in coords),
                            "ground_truth": annotation["IdentityOrText"], "vehicle_track_id": source,
                            "boxes_exhaustive": "false"}
                        per_region[clip["Region"]].append(row)
                        per_vehicle_class[clip.get("Unnamed: 17") or "unknown"].append(row)
                    analyzed += 1
                    clip_analyzed += 1
            finally:
                capture.release()
            finished = pipe.finished_tracks + [pipe.track_state.finish(i) for i in list(pipe.track_state.active)]
            payload["tracks"].extend(dict(row, source=source) for row in finished)
            reports.append({"source": source, "fps": fps, "native_size": native, "evaluation_size": dimensions,
                            "inferred_frames": clip_analyzed, "sample_every_source_frames": step,
                            "annotation_repair": source in TRANSPOSED_COLUMNS})
    finally:
        pipe.close()
    truth = [row for region in per_region.values() for row in region]
    extra, private = diagnose(payload, truth)
    result = {"scope": "publisher India test clips, partially labeled target plates",
              "resolution": resolution, "ocr": cfg["ocr_engine"],
              "clips": reports, "inferred_frames": analyzed, "wall_seconds": perf_counter() - started,
              "metrics": evaluate_rows(payload, truth), "detection": extra["detection"],
              "without_annotation_repair": evaluate_rows(payload, [r for r in truth if r["source"] not in TRANSPOSED_COLUMNS]),
              "by_region": {name: evaluate_rows(payload, rows) for name, rows in per_region.items()},
              "by_vehicle_class": {name: evaluate_rows(payload, rows) for name, rows in per_vehicle_class.items()}}
    return result, {"predictions": payload, "truth": truth, "cases": private}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("index", "annotations", "media", "output", "private-output", "config"):
        parser.add_argument("--" + key, type=Path, required=key != "config")
    parser.add_argument("--resolution", choices=("HR", "LR"), required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    results, private = evaluate(args.index, args.annotations, args.media, load_config(args.config), args.resolution)
    save(args.output, results)
    save(args.private_output, private)


if __name__ == "__main__":
    main()
