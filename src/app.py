from __future__ import annotations

from html import escape
import json
import logging
import os
from pathlib import Path
from tempfile import mkdtemp
from threading import Event, Lock
import zipfile

import gradio as gr

from src.cli import run_images, run_video_cli
from src.output.row_spool import RowSpool, write_csv
from src.utils.config import ROOT, apply_profile, load_config, validate_config

log = logging.getLogger(__name__)
_jobs: dict[str, Event] = {}
_jobs_lock = Lock()

STATUSES = ["READ", "LOW_CONFIDENCE", "UNREADABLE", "UNASSOCIATED", "NO_PLATE_DETECTED"]
COLUMNS = ["Source", "Frame", "Vehicle", "Track", "Plate candidate", "Status", "OCR score", "Plate width"]
TRACK_COLUMNS = ["Track", "Vehicle", "Final candidate", "Status", "Support", "Frames", "Best frame"]


def _ui_run_base() -> Path:
    return (Path(os.getenv("ANPR_OUTPUT_DIR", ROOT / "output")) / "ui_runs").resolve()


def _new_run_dir() -> Path:
    base = _ui_run_base()
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    return Path(mkdtemp(prefix="run-", dir=base))


def _settings(profile, device, ocr_engine, save_crops, paddle, stride, vehicle_threshold, plate_threshold, ocr_threshold):
    cfg = apply_profile(load_config(), profile)
    review = ocr_engine == "cct-awiros-review"
    cfg.update(device=device, ocr_engine="fast-plate-ocr" if review else ocr_engine, awiros_review=review,
               save_plate_crops=save_crops, paddle_ocr_fallback=paddle,
               video_frame_stride=int(stride), vehicle_threshold=vehicle_threshold, plate_threshold=plate_threshold,
               ocr_read_threshold=ocr_threshold)
    return validate_config(cfg)


def _cancel(request: gr.Request):
    with _jobs_lock:
        event = _jobs.get(request.session_hash)
        if event:
            event.set()
    return "Stop requested. Finishing the current frame or image." if event else "No active job to stop."


def _batch(paths, directory, cfg, model, progress, stopped):
    results = run_images(Path("."), directory, cfg, model, progress, paths=paths, cancelled=stopped)
    completed = [item for item in results if "error" not in item]
    errors = [item["error"] for item in results if "error" in item]
    status = "Cancelled" if stopped() else "Completed with errors" if errors and completed else "Failed" if errors else "Complete"
    summary = f"**{status}** · {len(completed)} of {len(paths)} images processed."
    if errors:
        summary += "\n\n" + "\n".join(f"- {escape(message)}" for message in errors)
    if not completed:
        return [], _empty_state(), None, None, None, summary

    json_path, csv_path, zip_path = [directory / f"batch.{ext}" for ext in ("json", "csv", "zip")]
    with RowSpool(cfg["ui_preview_rows"]) as spool:
        for item in completed:
            spool.add(item["observations"])
        with json_path.open("w", encoding="utf-8") as stream:
            batch = {"image_count": len(completed), "status": status, "errors": errors,
                     "inputs": [item["metadata"] for item in completed]}
            stream.write('{"run_metadata":' + json.dumps(batch) + ',"observations":[\n')
            for index, row in enumerate(spool):
                stream.write((",\n" if index else "") + json.dumps(row, ensure_ascii=False))
            stream.write('\n],"tracks":[]}\n')
        write_csv(csv_path, spool, spool.fields)
        state = {"rows": spool.preview, "tracks": [], "counts": dict(spool.counts), "total": spool.count}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in completed:
            for key in ("annotated", "json", "csv"):
                file = Path(item[key])
                archive.write(file, file.name)
        for file in (json_path, csv_path):
            archive.write(file, file.name)
        for file in sorted(directory.glob("*_crops/*.png")):
            archive.write(file, file.relative_to(directory).as_posix())
        manifest = ROOT / "models" / "manifest.yaml"
        if manifest.is_file():
            archive.write(manifest, "model_manifest.yaml")
    return [item["annotated"] for item in completed], state, str(json_path), str(csv_path), str(zip_path), summary


def _empty_state():
    return {"rows": [], "tracks": [], "counts": {}, "total": 0}


def _loopback(bind):
    return bind in {"127.0.0.1", "::1", "localhost"}


def _process(kind, media, duration, detector, profile, device, ocr_engine, save_crops, paddle,
             stride, vehicle_threshold, plate_threshold, ocr_threshold, progress, request):
    empty = [] if kind == "images" else None
    if not media:
        return empty, _empty_state(), None, None, None, "Choose input media to start."
    if kind == "live" and not _loopback(os.getenv("ANPR_UI_BIND", "127.0.0.1")):
        return empty, _empty_state(), None, None, None, "Server-side camera capture is available on the workstation UI."
    if kind == "live" and not (str(media).isdecimal() or str(media).lower().startswith(("rtsp://", "rtsps://"))):
        return empty, _empty_state(), None, None, None, "Enter a webcam device index or an RTSP URL."
    session = request.session_hash
    event = Event()
    with _jobs_lock:
        _jobs[session] = event
    try:
        cfg = _settings(profile, device, ocr_engine, save_crops, paddle, stride, vehicle_threshold, plate_threshold, ocr_threshold)
        directory = _new_run_dir()
        if kind == "images":
            paths = [Path(item) for item in media]
            return _batch(paths, directory, cfg, detector, progress, event.is_set)
        if isinstance(media, dict):
            media = media.get("video") or media.get("path")
        if not media:
            raise ValueError("Choose a video file to start.")
        result = run_video_cli(str(media), directory, cfg, detector, progress=progress,
                               max_duration_s=duration if kind == "live" else None, cancelled=event.is_set)
        info = result["metadata"]
        status = {"COMPLETE": "Complete", "CANCELLED": "Cancelled", "PARTIAL_DECODE": "Partial decode", "RESOURCE_LIMIT": "Output storage limit reached",
                  "LIMIT_REACHED": "Requested frame or duration limit reached"}[info["status"]]
        summary = (f"**{status}** · {info['decoded_frames']:,} decoded · {info['inferred_frames']:,} analyzed"
                   f" · {len(result['tracks']):,} vehicle tracks")
        for warning in info.get("warnings", []):
            summary += "\n\n" + escape(warning)
        if info.get("device_fallback"):
            summary += "\n\nCUDA startup failed. This run used the configured CPU fallback."
        state = {"rows": result["observations"], "tracks": result["tracks"],
                 "counts": info.get("status_counts", {}), "total": info.get("observations", 0)}
        return result["annotated"], state, result["json"], result["csv"], result["zip"], summary
    except (ValueError, OSError, RuntimeError) as exc:
        log.warning("%s workflow failed (%s)", kind, type(exc).__name__)
        message = str(exc) if "rtsp://" not in str(exc).lower() and "rtsps://" not in str(exc).lower() else "Check stream URL and authentication."
        return empty, _empty_state(), None, None, None, f"**Processing failed.** {escape(message)}"
    except Exception as exc:
        log.error("%s workflow failed (%s)", kind, type(exc).__name__)
        return empty, _empty_state(), None, None, None, "**Processing failed.** Check the runtime and model cache in Diagnostics."
    finally:
        with _jobs_lock:
            _jobs.pop(session, None)


def _filtered(state, status, query):
    query = (query or "").strip().casefold()
    rows = state["rows"]
    return [row for row in rows if (status == "All statuses" or row.get("recognition_status") == status)
            and (not query or query in f"{row.get('ocr_normalized') or ''} {row.get('vehicle_class') or ''} "
                                       f"{row.get('vehicle_track_id')} {Path(str(row.get('source', ''))).name}".casefold())]


def _table(rows):
    return [[Path(str(r.get("source", ""))).name, r.get("frame_index"), r.get("vehicle_class"),
             r.get("vehicle_track_id"), r.get("ocr_normalized"), r.get("recognition_status"),
             round(r["ocr_raw_score"], 3) if r.get("ocr_raw_score") is not None else None, r.get("plate_width_px")]
            for r in rows]


def _view(state, status, query):
    rows = _filtered(state, status, query)
    tracks = [[t["vehicle_track_id"], t["vehicle_class"], t["consensus"], t["recognition_status"],
               t["consensus_support"], t["observations"], t["best_frame_index"]] for t in state["tracks"][:2000]]
    values = [("Plate boxes", sum(state["counts"].get(key, 0) for key in STATUSES[:-1])),
              ("Frame reads", state["counts"].get("READ", 0)),
              ("Needs review", state["counts"].get("LOW_CONFIDENCE", 0) + state["counts"].get("UNREADABLE", 0)),
              ("Unassociated", state["counts"].get("UNASSOCIATED", 0))]
    tiles = "".join(f"<div><strong>{value:,}</strong><span>{label}</span></div>" for label, value in values)
    scope = f"{len(rows):,} visible of {state['total']:,} observations."
    if len(state["rows"]) < state["total"]:
        scope += " Downloads include the full run. Filters apply to the preview."
    return f'<div class="metrics">{tiles}</div>', _table(rows), tracks, scope


def _inspect(state, status, query, event: gr.SelectData):
    rows = _filtered(state, status, query)
    index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
    if not 0 <= index < len(rows):
        return "Select a row to inspect the evidence.", {}
    row = rows[index]
    box = row.get("plate_bbox_xyxy")
    if not box:
        return f"**{row['recognition_status']}** · Frame {row['frame_index']} · {row.get('vehicle_class') or ''}", row
    evidence = row.get("recognition_evidence", {})
    association = row.get("association_evidence", {})
    raw_score = row.get("ocr_raw_score")
    minimum = row.get("ocr_minimum_char_score")
    reason_labels = {
        "no_primary_text": "Primary OCR returned no text",
        "format_not_recognized": "Registration format unrecognized",
        "internal_ocr_padding": "OCR left an internal blank",
        "unsupported_ocr_symbols": "OCR returned unsupported symbols",
        "insufficient_source_pixels": "Plate has too few source pixels",
        "low_sharpness": "Plate has low sharpness",
        "low_contrast": "Plate has low contrast",
        "plate_at_frame_boundary": "Plate touches the frame boundary",
        "plate_detector_below_read_gate": "Weak plate detection",
        "independent_ocr_disagrees": "The second reader disagrees",
        "primary_ocr_below_read_gate": "Primary OCR fell below the frame read gate",
        "primary_character_scores_unavailable": "Primary reader provided no character-level scores",
        "ambiguous": "More than one vehicle contains the plate",
        "multiple_plates": "More than one plate candidate matches the vehicle",
        "no_containing_vehicle": "No detected vehicle contains the plate",
        "insufficient_containment": "Vehicle containment is weak",
    }
    flags = [reason_labels.get(reason, reason) for reason in evidence.get("reasons", [])]
    text = row.get("ocr_normalized") or "No primary candidate"
    summary = f"**{escape(text)}** · {row['recognition_status']} · Frame {row['frame_index']}"
    summary += (f"\n\nPlate size {row.get('plate_width_px')} × {row.get('plate_height_px')} px"
                f" · Detector score {row['plate_detection_confidence']:.2f}"
                f" · Vehicle containment {association.get('containment', 0):.0%}")
    if raw_score is not None and minimum is not None:
        summary += f"\n\nOCR mean {raw_score:.2f} · Weakest character {minimum:.2f}"
    alternatives = [escape(str(candidate["normalized"])) for candidate in row.get("ocr_alternatives", [])
                    if candidate.get("normalized")]
    if alternatives:
        summary += "\n\nOther OCR candidates: " + ", ".join(alternatives)
    if flags:
        summary += "\n\n" + " · ".join(flags)
    return summary, row


def _inspect_track(state, event: gr.SelectData):
    index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
    if not 0 <= index < min(2000, len(state["tracks"])):
        return "Select a track to inspect its evidence.", {}
    track = state["tracks"][index]
    text = track["consensus"] or "No candidate"
    frames = ", ".join(map(str, track["evidence_frame_indices"])) or "none"
    first, last, stable = (track.get(key) for key in ("first_seen_ms", "last_seen_ms", "stable_read_ms"))
    timing = f"\n\nObserved {first / 1000:.2f}–{last / 1000:.2f} s." if first is not None and last is not None else ""
    if stable is not None:
        timing += f" Stable READ at {stable / 1000:.2f} s."
    if track.get("review_ocr_consensus"):
        timing += (f"\n\nAwiros review candidate **{escape(track['review_ocr_consensus'])}**"
                   f" from {track['review_ocr_support']} distinct observations.")
    return (f"**{escape(text)}** · {track['recognition_status']} · Track {track['vehicle_track_id']}\n\n"
            f"Supporting frame indices: {frames}. Candidate weight share {track['consensus_fraction']:.0%}.{timing}", track)


def _diagnostics():
    from src.utils.device import device_report
    import yaml

    report = device_report()
    manifest = ROOT / "models" / "manifest.yaml"
    if manifest.is_file():
        models = yaml.safe_load(manifest.read_text(encoding="utf-8")).get("weight_files", [])
        report["cached_weights"] = {entry["path"]: (ROOT / entry["path"]).is_file() for entry in models}
    report["offline_mode"] = os.getenv("ANPR_OFFLINE") == "1"
    return report


def _workflow(kind, settings):
    is_image = kind == "images"
    is_live = kind == "live"
    with gr.Row(elem_classes=["work-area"]):
        with gr.Column(scale=4, min_width=280, elem_classes=["work-panel"]):
            gr.Markdown("### Input media")
            if is_image:
                media = gr.File(file_count="multiple", file_types=[".jpg", ".jpeg", ".png", ".webp", ".bmp"],
                                label="Upload images", height=145)
                gr.Markdown("JPEG, PNG, WebP or BMP. Select up to 100 images.")
            elif is_live:
                media = gr.Textbox(value="0", label="Webcam index or RTSP URL", type="password")
            else:
                media = gr.Video(sources=["upload", "webcam"], label="Upload or record a video", height=205)
            duration = gr.Slider(5, 300, value=30, step=5, label="Capture duration (seconds)", visible=is_live)
            with gr.Row():
                start = gr.Button("Process images" if is_image else "Capture and process" if is_live else "Process video",
                                  variant="primary", min_width=170)
                stop = gr.Button("Stop", min_width=75)
            feedback = gr.Markdown("Ready for an upload." if not is_live else "Ready to capture.", elem_classes=["run-feedback"])
            with gr.Accordion("Download results", open=False):
                bundle = gr.File(label="Complete results (ZIP)", interactive=False)
                json_file = gr.File(label="Evidence (JSON)", interactive=False)
                csv_file = gr.File(label="Observations (CSV)", interactive=False)
        with gr.Column(scale=7, min_width=300, elem_classes=["work-panel"]):
            gr.Markdown("### Annotated preview")
            if is_image:
                preview = gr.Gallery(label="Annotated images", show_label=False, columns=2, height=345, object_fit="contain")
            else:
                preview = gr.Video(label="Annotated video", show_label=False, height=345, interactive=False)
    state = gr.State(_empty_state())
    metrics = gr.HTML(_view(_empty_state(), "All statuses", "")[0])
    gr.Markdown("### Review evidence")
    with gr.Tabs():
        with gr.Tab("Observations"):
            with gr.Row():
                status = gr.Dropdown(["All statuses", *STATUSES], value="All statuses", label="Status", min_width=170)
                query = gr.Textbox(label="Search", placeholder="Plate, vehicle, track or source", min_width=190)
            scope = gr.Markdown("Process media to inspect detections.")
            observation = gr.Dataframe(headers=COLUMNS, label="Frame observations", interactive=False, max_height=410)
            observation_summary = gr.Markdown("Select a row to inspect the evidence.")
            with gr.Accordion("Raw observation evidence", open=False):
                detail = gr.JSON(label="Selected observation evidence")
        if not is_image:
            with gr.Tab("Vehicle tracks"):
                track_table = gr.Dataframe(headers=TRACK_COLUMNS, label="Track summaries", interactive=False, max_height=410)
                track_summary = gr.Markdown("Select a track to inspect its evidence.")
                with gr.Accordion("Raw track evidence", open=False):
                    track_detail = gr.JSON(label="Selected track evidence")

    def submit(media, duration, detector, profile, device, ocr_engine, save_crops, paddle,
               stride, vehicle_threshold, plate_threshold, ocr_threshold,
               request: gr.Request, progress=gr.Progress()):
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), "Starting analysis."
        yield _process(kind, media, duration, detector, profile, device, ocr_engine, save_crops, paddle,
                       stride, vehicle_threshold, plate_threshold, ocr_threshold, progress, request)

    def refresh(state, status, query):
        cards, rows, tracks, scope_text = _view(state, status, query)
        basics = (cards, rows, scope_text, "Select a row to inspect the evidence.", {})
        return basics if is_image else (*basics, tracks, "Select a track to inspect its evidence.", {})

    start.click(submit, [media, duration, *settings], [preview, state, json_file, csv_file, bundle, feedback],
                concurrency_id="anpr-inference", concurrency_limit=1, api_visibility="private", show_progress_on=preview)
    stop.click(_cancel, outputs=feedback, queue=False, api_visibility="private")
    outputs = [metrics, observation, scope, observation_summary, detail]
    if not is_image:
        outputs.extend([track_table, track_summary, track_detail])
        track_table.select(_inspect_track, state, [track_summary, track_detail], queue=False, api_visibility="private")
    gr.on([state.change, status.change, query.change], refresh, [state, status, query], outputs,
          queue=False, api_visibility="private")
    observation.select(_inspect, [state, status, query], [observation_summary, detail], queue=False, api_visibility="private")


def build_ui():
    cfg = load_config()
    import torch

    devices = [(f"GPU {i} · {torch.cuda.get_device_name(i).replace('NVIDIA ', '').replace('GeForce ', '')}", f"cuda:{i}")
               for i in range(torch.cuda.device_count())] + [("CPU", "cpu")]
    selected = cfg["device"].replace("cuda", "cuda:0") if cfg["device"] == "cuda" else cfg["device"]
    if selected not in dict(devices).values():
        selected = devices[0][1]
    with gr.Blocks(title="India ANPR | Review workspace", analytics_enabled=False) as demo:
        gr.HTML(f"""
        <header class="anpr-header">
            <div><div class="product-name">INDIA ANPR</div>
            <h1>Plate review workspace</h1>
            <p>Analyze traffic footage and inspect the evidence behind each plate candidate.</p></div>
            <div class="runtime-pill"><span></span>Local inference · {len(devices) - 1} GPUs detected</div>
        </header>
        """)
        with gr.Accordion("Processing settings", open=False, elem_id="settings"):
            with gr.Row():
                profile = gr.Dropdown([("Standard", "standard"), ("Recovery · experimental tiles and OCR alternatives", "recovery")],
                                      value="standard", label="Profile", min_width=225)
                detector = gr.Dropdown([("RT-DETRv2 R50", "rtdetr-v2-r50"), ("YOLOv9-S 608 · evaluation only", "yolo-v9-s-608")],
                                       value=cfg["plate_model"], label="Plate detector", min_width=220)
                device = gr.Dropdown(devices, value=selected, label="Device", min_width=170)
                ocr = gr.Dropdown([("CCT-S v2 Global", "fast-plate-ocr"), ("Awiros · experimental candidates", "awiros"),
                                   ("CCT-S + Awiros suggestions for review", "cct-awiros-review"),
                                   ("Indian TrOCR · non-commercial research", "indian-trocr")],
                                  value=cfg["ocr_engine"], label="OCR engine", min_width=225)
            with gr.Row():
                vehicle_threshold = gr.Slider(0.05, 0.9, cfg["vehicle_threshold"], step=0.05, label="Vehicle threshold", min_width=170)
                plate_threshold = gr.Slider(0.05, 0.9, cfg["plate_threshold"], step=0.05, label="Plate threshold", min_width=170)
                ocr_threshold = gr.Slider(0.3, 1.0, cfg["ocr_read_threshold"], step=0.01, label="Frame OCR score gate", min_width=180)
                stride = gr.Slider(1, 10, cfg["video_frame_stride"], step=1, label="Analyze every Nth video frame", min_width=210)
            with gr.Row():
                save_crops = gr.Checkbox(value=cfg["save_plate_crops"], label="Include original plate crops in results")
                paddle = gr.Checkbox(value=cfg["paddle_ocr_fallback"], label="Use optional PaddleOCR second opinion")
        settings = [detector, profile, device, ocr, save_crops, paddle, stride, vehicle_threshold, plate_threshold, ocr_threshold]
        with gr.Tabs(elem_id="workflows"):
            with gr.Tab("Images"):
                _workflow("images", settings)
            with gr.Tab("Recorded video"):
                _workflow("video", settings)
            if _loopback(os.getenv("ANPR_UI_BIND", "127.0.0.1")):
                with gr.Tab("Webcam / RTSP"):
                    _workflow("live", settings)
        with gr.Accordion("Diagnostics", open=False):
            gr.Button("Check GPU and model cache").click(_diagnostics, outputs=gr.JSON(label="Runtime status"),
                                                         api_visibility="private")
        gr.Markdown("READ indicates model evidence met the configured gates. OCR scores are uncalibrated.", elem_classes=["footer-note"])
    return demo


def launch():
    css = Path(__file__).with_name("app.css").read_text(encoding="utf-8")
    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate", font=["system-ui", "sans-serif"])
    bind = os.getenv("ANPR_UI_BIND", "127.0.0.1")
    auth = None
    if not _loopback(bind):
        secret = os.getenv("ANPR_UI_PASSWORD_FILE")
        if not secret:
            raise ValueError("Set ANPR_UI_PASSWORD_FILE to enable a network bind.")
        password = Path(secret).read_text(encoding="utf-8").strip()
        if len(password) < 16:
            raise ValueError("The network UI password must contain at least 16 characters.")
        auth = (os.getenv("ANPR_UI_USERNAME", "operator"), password)
    build_ui().queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=bind, share=False, theme=theme, css=css, max_file_size="2gb", auth=auth,
        allowed_paths=[str(_ui_run_base())], enable_monitoring=False)


if __name__ == "__main__":
    launch()
