"""Check exported video, JSON, CSV and ZIP for internal consistency."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import zipfile


def check(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["run_metadata"]
    observations = data["observations"]
    video = path.with_name(f"{path.stem}_annotated.mp4")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                            "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate",
                            "-of", "json", str(video)], capture_output=True, text=True, check=True)
    stream, = json.loads(probe.stdout)["streams"]
    if int(stream["nb_read_frames"]) != meta["decoded_frames"]:
        raise ValueError("Decoded saved frame count differs from metadata.")
    with path.with_suffix(".csv").open(newline="", encoding="utf-8") as file:
        rows = sum(1 for _ in csv.DictReader(file))
    if rows != len(observations) or rows != meta["observations"]:
        raise ValueError("Observation counts differ between CSV, JSON or metadata.")
    width, height = meta["source_size"]
    previous = 0
    for item in observations:
        frame = item["frame_index"]
        if frame < previous or frame >= meta["decoded_frames"] or frame % meta["stride"]:
            raise ValueError("Invalid or out-of-order source frame index.")
        previous = frame
        for field in ("plate_bbox_xyxy", "vehicle_bbox_xyxy"):
            box = item.get(field)
            if box:
                x1, y1, x2, y2 = box
                if not 0 <= x1 <= x2 <= width or not 0 <= y1 <= y2 <= height:
                    raise ValueError(f"Out-of-bounds {field} in frame {frame}.")
    with zipfile.ZipFile(path.with_name(f"{path.stem}_results.zip")) as bundle:
        if not {video.name, path.name, path.with_suffix(".csv").name, "model_manifest.yaml"} <= set(bundle.namelist()):
            raise ValueError("Export archive lacks required artifacts.")
        if bundle.testzip():
            raise ValueError("Export archive failed CRC validation.")
    return {
        "status": meta["status"], "decoded": meta["decoded_frames"], "analyzed": meta["inferred_frames"],
        "encoded": int(stream["nb_read_frames"]), "resolution": [stream["width"], stream["height"]],
        "fps": stream["avg_frame_rate"], "rows": rows, "tracks": len(data["tracks"]),
        "frame_statuses": Counter(item["recognition_status"] for item in observations),
        "track_statuses": Counter(item["recognition_status"] for item in data["tracks"]),
        "wall_seconds": meta["duration_wall_s"], "timing": meta["performance_ms"],
        "torch_peak_mib": meta["torch_gpu_peak_allocated_mib"], "device": meta["device"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exports", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps({path.parent.name: check(path) for path in args.exports}, indent=2))


if __name__ == "__main__":
    main()
