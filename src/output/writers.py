from __future__ import annotations

import json
import logging
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2

from src.output.row_spool import RowSpool, write_csv
from src.utils.config import ROOT
from src.utils.checksums import sha256_file

log = logging.getLogger(__name__)


def safe_source(source: str) -> str:
    if not source.lower().startswith(("rtsp://", "rtsps://")):
        return source
    parts = urlsplit(source)
    host = parts.hostname or "camera"
    if ":" in host:
        host = f"[{host}]"
    try:
        if parts.port:
            host += f":{parts.port}"
    except ValueError:
        host = "camera"
    return urlunsplit((parts.scheme, f"***@{host}" if parts.username or parts.password else host, parts.path, "[redacted]" if parts.query else "", ""))


def safe_source_stem(source: str) -> str:
    if source.lower().startswith(("rtsp://", "rtsps://")):
        return Path(urlsplit(source).path).stem or "stream"
    return Path(source).stem or "stream"


def write_results(source: str, rows, annotated_path: Path, output_dir: Path, media_type: str,
                  plate_model: str, cfg: dict, pipeline_stats: list[dict] | None = None,
                  output_stem: str | None = None, *, track_summaries=None, run_info: dict | None = None,
                  crop_dir: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or safe_source_stem(source)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    tracks: dict[str, dict] = {}
    for row in rows if track_summaries is None else ():
        track_id = row.get("vehicle_track_id")
        if track_id is None:
            continue
        key = str(track_id)
        target = tracks.setdefault(key, {"vehicle_track_id": track_id, "vehicle_class": row.get("vehicle_class"), "_frames": {}})
        frame_key = (row.get("source"), row.get("frame_index"))
        previous = target["_frames"].get(frame_key)
        if previous is None or row.get("quality_score", 0) > previous.get("quality_score", 0):
            target["_frames"][frame_key] = row
    for track in tracks.values():
        weights: dict[str, float] = defaultdict(float)
        frame_rows = list(track.pop("_frames").values())
        track["observations"] = len(frame_rows)
        track["read_observations"] = sum(row.get("recognition_status") == "READ" for row in frame_rows)
        track["candidates"] = {}
        for row in frame_rows:
            if row.get("ocr_normalized"):
                candidate = row["ocr_normalized"]
                track["candidates"][candidate] = track["candidates"].get(candidate, 0) + 1
                weights[row["ocr_normalized"]] += max(0.01, float(row.get("quality_score", 0.01)))
        ranking = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        track["consensus"] = ranking[0][0] if ranking else None
        track["consensus_support"] = track["candidates"].get(track["consensus"], 0) if track["consensus"] else 0
        track["consensus_margin"] = round((ranking[0][1] - (ranking[1][1] if len(ranking) > 1 else 0)) / max(ranking[0][1], 1e-6), 4) if ranking else 0.0
    summaries = list(tracks.values()) if track_summaries is None else track_summaries
    row_count = rows.count if isinstance(rows, RowSpool) else len(rows)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "source": safe_source(source),
        "media_type": media_type, "plate_model": plate_model, "ocr_model": cfg["ocr_model"],
        "vehicle_model": "RF-DETR Large", "tracking_model": "ByteTrack" if media_type == "video" else None,
        "device": cfg["device"], "observations": row_count, "schema_version": 2, "settings": cfg,
        "accuracy_measured": False,
        "ocr_scores_are_calibrated": False,
        **(run_info or {}),
    }
    model_manifest = ROOT / "models" / "manifest.yaml"
    if model_manifest.is_file():
        import yaml
        manifest = yaml.safe_load(model_manifest.read_text(encoding="utf-8")) or {}
        metadata["model_manifest_path"] = "models/manifest.yaml"
        metadata["model_manifest_sha256"] = sha256_file(model_manifest)
        metadata["model_revisions"] = {
            key: manifest.get("models", {}).get(key, {}).get("revision")
            for key in ("plate_default", "indian_trocr_research", "awiros_optional")
            if manifest.get("models", {}).get(key, {}).get("revision")
        }
        metadata["model_artifact_checksums"] = [
            {"path": row.get("path"), "sha256": row.get("sha256")}
            for row in manifest.get("weight_files", [])
        ]
    if str(cfg.get("device", "")).startswith("cuda"):
        try:
            import torch
            device_index = int(str(cfg["device"]).split(":", 1)[1]) if ":" in str(cfg["device"]) else 0
            torch.cuda.synchronize(device_index)
            metadata["torch_gpu_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(device_index) / (1024**2), 1)
            metadata["torch_gpu_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved(device_index) / (1024**2), 1)
        except Exception:
            log.exception("Could not record peak GPU memory")
    try:
        from src.utils.device import device_report
        metadata["runtime"] = device_report(probe=False)
    except Exception:
        log.exception("Could not include runtime metadata")
    if pipeline_stats:
        import numpy as np
        keys = pipeline_stats[0].keys()
        metadata["performance_ms"] = {key: {"mean": round(float(np.mean([row[key] for row in pipeline_stats])), 3), "p95": round(float(np.percentile([row[key] for row in pipeline_stats], 95)), 3)} for key in keys}
    fields = rows.fields if isinstance(rows, RowSpool) else dict.fromkeys(
        key for row in rows for key in row)
    zip_path = output_dir / f"{stem}_results.zip"
    temporary_paths = [path.with_name("." + path.name + ".partial") for path in (json_path, csv_path, zip_path)]
    json_tmp, csv_tmp, zip_tmp = temporary_paths
    try:
        with json_tmp.open("w", encoding="utf-8") as stream:
            stream.write('{"run_metadata":')
            json.dump(metadata, stream, ensure_ascii=False, allow_nan=False)
            stream.write(',"observations":[\n')
            for index, row in enumerate(rows):
                if index:
                    stream.write(",\n")
                json.dump(row, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n],"tracks":')
            json.dump(summaries, stream, ensure_ascii=False, allow_nan=False)
            stream.write("}\n")
        write_csv(csv_tmp, rows, fields or ["source", "frame_index", "timestamp_ms", "recognition_status"])
        with zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, name in ((annotated_path, annotated_path.name), (json_tmp, json_path.name), (csv_tmp, csv_path.name)):
                if path.is_file():
                    archive.write(path, name)
            if model_manifest.is_file():
                archive.write(model_manifest, "model_manifest.yaml")
            if crop_dir and crop_dir.is_dir():
                for path in sorted(crop_dir.glob("*.png")):
                    archive.write(path, f"{crop_dir.name}/{path.name}")
        for temporary, final in zip(temporary_paths, (json_path, csv_path, zip_path)):
            temporary.replace(final)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return {"annotated": str(annotated_path) if annotated_path.is_file() else None, "json": str(json_path),
            "csv": str(csv_path), "zip": str(zip_path), "metadata": metadata, "tracks": summaries}


def make_video_writer(path: Path, fps: float, width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open MP4 writer. Check the system FFmpeg/OpenCV codec support.")
    return writer
