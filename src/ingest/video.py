from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import cv2

from src.output.run_guard import OutputSpaceError, check_space
from src.output.writers import make_video_writer
from src.ingest.ffmpeg_video import Av1Capture

log = logging.getLogger(__name__)


def run_video(source: str, output_path: Path, pipeline, stride: int = 1, progress=None,
              max_frames: int | None = None, max_duration_s: float | None = None,
              *, row_sink=None, cancelled=None, run_info: dict | None = None) -> list[dict]:
    webcam = source.isdecimal()
    rtsp = source.lower().startswith(("rtsp://", "rtsps://"))
    live = webcam or rtsp
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("Stride and frame limit must be positive.")
    if max_duration_s is not None and (not math.isfinite(max_duration_s) or max_duration_s <= 0):
        raise ValueError("Capture duration must be positive and finite.")
    if not live and not Path(source).is_file():
        raise ValueError("Recorded video must be a readable local file.")
    if live and max_frames is None and max_duration_s is None:
        max_frames = 900
    decode_options = [cv2.CAP_PROP_N_THREADS, pipeline.cfg["video_decoder_threads"]]
    capture = cv2.VideoCapture()
    av1 = None
    writer = None
    index = sampled = 0
    rows = []
    status = "COMPLETE"
    warnings = []
    requested_limit = max_frames or pipeline.cfg["video_decode_limit"]
    frame_limit = min(requested_limit, pipeline.cfg["video_decode_limit"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.stem}.partial.mp4")
    try:
        if rtsp:
            capture.open(source, cv2.CAP_FFMPEG, decode_options + [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000])
        elif webcam:
            capture.open(int(source))
        elif not capture.open(source, cv2.CAP_FFMPEG, decode_options):
            capture.open(source)
        if not capture.isOpened():
            raise RuntimeError("Cannot open video/stream. Check the source, camera permissions and codec support.")
        codec = int(capture.get(cv2.CAP_PROP_FOURCC)) if not live else 0
        if codec in {cv2.VideoWriter_fourcc(*"av01"), cv2.VideoWriter_fourcc(*"AV01")}:
            capture.release()
            av1 = Av1Capture(source, pipeline.cfg["video_decoder_threads"], pipeline.cfg["max_frame_pixels"])
            warnings.append("System FFmpeg decoded AV1. Export timestamps use the nominal frame rate.")
        raw_fps = av1.fps if av1 else capture.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if 0 < raw_fps < 300 else 25.0
        if fps != raw_fps:
            warnings.append("Source frame rate was unavailable. Output uses 25 FPS.")
        frame_count = av1.frame_count if av1 else 0 if live else int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        started = time.monotonic()
        last_timestamp = -1
        target_frames = min(frame_count, frame_limit) if frame_count > 0 else frame_limit
        while True:
            elapsed = time.monotonic() - started
            if cancelled and cancelled():
                status = "CANCELLED"
                break
            if index >= frame_limit:
                if not frame_count or frame_count > index:
                    status = "LIMIT_REACHED"
                break
            if max_duration_s is not None and elapsed >= max_duration_s:
                status = "LIMIT_REACHED"
                break
            if index and index % 30 == 0:
                try:
                    check_space(output_path.parent, pipeline.cfg["min_output_free_mib"])
                except OutputSpaceError as exc:
                    status = "RESOURCE_LIMIT"
                    warnings.append(str(exc))
                    break
            ok, frame = av1.read() if av1 else capture.read()
            if not ok:
                if (live or frame_count > index or (av1 and av1.failed)) and index:
                    status = "PARTIAL_DECODE"
                    warnings.append("Input stopped before the requested capture or advertised frame count.")
                break
            height, width = frame.shape[:2]
            if width * height > pipeline.cfg["max_frame_pixels"]:
                raise ValueError(f"Video frame is too large ({width}×{height}). Adjust max_frame_pixels if needed.")
            if writer is None:
                source_size = (width, height)
                encoded_size = (width + width % 2, height + height % 2)
                writer = make_video_writer(partial_path, fps, *encoded_size)
                if encoded_size != source_size:
                    warnings.append("One padding pixel was added to each odd output dimension for the MP4 codec.")
            elif (width, height) != source_size:
                raise ValueError("Video changes frame size during decoding.")
            if live:
                timestamp = round((time.monotonic() - started) * 1000)
            else:
                pts = round(index * 1000 / fps) if av1 else capture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = round(pts) if math.isfinite(pts) and pts >= 0 else -1
                if timestamp <= last_timestamp:
                    timestamp = max(last_timestamp + 1, round(index * 1000 / fps))
            last_timestamp = timestamp
            if index % stride == 0:
                pipeline.frame_index = index
                frame, new_rows, _ = pipeline.process(frame, source, timestamp)
                if row_sink is None:
                    rows.extend(new_rows)
                else:
                    row_sink(new_rows)
                sampled += 1
            if encoded_size != source_size:
                frame = cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
            writer.write(frame)
            index += 1
            if index % 60 == 0:
                log.info("Decoded %d input frames and analyzed %d", index, sampled)
            if progress:
                desc = f"Frame {index:,} · {sampled:,} analyzed"
                if live and max_duration_s is not None:
                    progress(min(0.99, elapsed / max_duration_s), desc=desc)
                else:
                    progress(min(0.99, index / target_frames), desc=desc)
        if not index and av1 and av1.failed:
            raise RuntimeError("System FFmpeg could not decode the AV1 file. Check that libdav1d is installed and the file is intact.")
        if not index and status != "CANCELLED":
            raise RuntimeError("Source opened, but no frames could be decoded.")
        if index:
            writer.release()
            writer = None
            check = cv2.VideoCapture(str(partial_path), cv2.CAP_FFMPEG, decode_options)
            try:
                if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != index:
                    raise OSError("Encoded MP4 failed the frame-count integrity check.")
            finally:
                check.release()
            partial_path.replace(output_path)
        if run_info is not None:
            run_info.update(status=status, decoded_frames=index, inferred_frames=sampled,
                            advertised_frames=frame_count or None, fps=fps, stride=stride,
                            source_size=list(source_size) if index else None,
                            output_size=list(encoded_size) if index else None,
                            duration_wall_s=round(time.monotonic() - started, 3), warnings=warnings,
                            decoder="ffmpeg_libdav1d" if av1 else "opencv")
    finally:
        capture.release()
        if av1:
            av1.release()
        if writer is not None:
            writer.release()
        partial_path.unlink(missing_ok=True)
    log.info("Processed %d of %d decoded frames (%s)", sampled, index, status)
    return rows
