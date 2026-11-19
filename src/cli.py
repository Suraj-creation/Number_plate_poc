from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

from src.utils.config import PLATE_MODELS, ROOT, apply_profile, configure_caches, load_config, validate_config
from src.utils.device import device_report
from src.utils.logging import setup_logging

configure_caches()
log = logging.getLogger("india_anpr")


def make_pipeline(cfg, plate_model, video=False):
    from src.pipeline.frame_pipeline import FramePipeline
    from src.utils.device import CudaUnavailable

    try:
        pipe = FramePipeline(cfg, plate_model=plate_model, enable_tracking=video)
    except CudaUnavailable:
        if not cfg["allow_cpu_fallback"]:
            raise
        log.warning("CUDA startup probe failed for %s. Starting the explicitly permitted CPU fallback.", cfg["device"])
        pipe = FramePipeline(dict(cfg, device="cpu"), plate_model=plate_model, enable_tracking=video)
    pipe.execution = {"requested_device": cfg["device"], "effective_device": pipe.cfg["device"],
                      "device_fallback": pipe.cfg["device"] != cfg["device"]}
    return pipe


def run_images(input_path: Path, output_dir: Path, cfg: dict, plate_model: str, progress=None,
               *, paths=None, cancelled=None) -> list[dict]:
    from src.output.run_guard import output_run

    with output_run(output_dir, cfg["min_output_free_mib"]):
        return _run_images(input_path, output_dir, cfg, plate_model, progress, paths, cancelled)


def _run_images(input_path, output_dir, cfg, plate_model, progress, paths, cancelled):
    from src.ingest.images import list_images, read_image
    from src.output.run_guard import check_space
    from src.output.writers import write_results

    paths = list_images(input_path) if paths is None else paths
    if not paths:
        raise ValueError("Choose at least one supported image.")
    if len(paths) > cfg["max_batch_images"]:
        raise ValueError(f"Batch exceeds the {cfg['max_batch_images']} image limit.")
    if progress:
        progress(0, desc="Loading pretrained models")
    pipe = None
    all_results = []
    used_stems = set()
    try:
        for index, path in enumerate(paths):
            if cancelled and cancelled():
                break
            check_space(output_dir, cfg["min_output_free_mib"])
            try:
                image = read_image(path, cfg["max_frame_pixels"])
            except (ValueError, OSError, cv2.error) as exc:
                message = f"{path.name}: {exc}"
                all_results.append({"source": str(path), "error": message})
                log.warning("%s", message)
                continue
            if pipe is None:
                pipe = make_pipeline(cfg, plate_model)
            pipe.frame_index = 0
            output_stem = path.stem
            suffix = 2
            while output_stem in used_stems:
                output_stem = f"{path.stem}_{suffix}"
                suffix += 1
            used_stems.add(output_stem)
            pipe.crop_dir = output_dir / f"{output_stem}_crops" if cfg["save_plate_crops"] else None
            annotated, rows, _ = pipe.process(image, str(path), 0)
            out_path = output_dir / f"{output_stem}_annotated.png"
            if not cv2.imwrite(str(out_path), annotated):
                raise OSError(f"Could not write annotated image: {out_path}")
            result = write_results(str(path), rows, out_path, output_dir, "image", plate_model, pipe.cfg,
                                   pipe.timings[-1:], output_stem=output_stem, crop_dir=pipe.crop_dir, run_info=pipe.execution)
            all_results.append({"source": str(path), "observations": rows, **result})
            if progress:
                progress((index + 1) / len(paths), desc=f"Image {index + 1}/{len(paths)}")
    finally:
        if pipe is not None:
            pipe.close()
    return all_results


def run_video_cli(source: str, output_dir: Path, cfg: dict, plate_model: str, stride: int | None = None,
                  max_frames: int | None = None, progress=None, max_duration_s: float | None = None, *, cancelled=None):
    from contextlib import ExitStack

    from src.ingest.video import run_video
    from src.output.row_spool import RowSpool
    from src.output.run_guard import output_run
    from src.output.writers import safe_source_stem, write_results

    local_file = not (source.isdecimal() or source.lower().startswith(("rtsp://", "rtsps://")))
    if local_file and not Path(source).is_file():
        raise ValueError("Recorded video must be a readable local file.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("Frame limit must be positive.")
    stride = cfg["video_frame_stride"] if stride is None else stride
    cfg = validate_config(dict(cfg, video_frame_stride=stride))
    if progress:
        progress(0, desc="Loading pretrained models")
    stem = safe_source_stem(source)
    annotated = output_dir / f"{stem}_annotated.mp4"

    with ExitStack() as resources:
        resources.enter_context(output_run(output_dir, cfg["min_output_free_mib"]))
        pipe = make_pipeline(cfg, plate_model, video=True)
        resources.callback(pipe.close)
        pipe.crop_dir = output_dir / f"{stem}_crops" if cfg["save_plate_crops"] else None
        info = dict(pipe.execution)
        spool = resources.enter_context(RowSpool(cfg["ui_preview_rows"], directory=output_dir))
        run_video(source, annotated, pipe, stride=stride, progress=progress, max_frames=max_frames,
                  max_duration_s=max_duration_s, row_sink=spool.add, cancelled=cancelled, run_info=info)
        if info["decoded_frames"] == 0:
            return {"annotated": None, "json": None, "zip": None, "csv": None,
                    "metadata": info, "observations": [], "tracks": []}
        if progress:
            progress(0.99, desc="Writing video, evidence and downloads")
        end_reason = "END_OF_INPUT" if info["status"] == "COMPLETE" else info["status"]
        tracks = pipe.finished_tracks + [pipe.track_state.finish(key, end_reason) for key in list(pipe.track_state.active)]
        info["status_counts"] = dict(spool.counts)
        result = write_results(source, spool, annotated, output_dir, "video", plate_model, pipe.cfg, pipe.timings,
                               track_summaries=tracks, run_info=info, crop_dir=pipe.crop_dir)
        log.info("Video result saved", extra={"anpr_metrics": {
            "status": info["status"], "device": info["effective_device"],
            "decoded_frames": info["decoded_frames"], "inferred_frames": info["inferred_frames"],
            "duration_wall_s": info["duration_wall_s"], "observations": spool.count, "vehicle_tracks": len(tracks),
        }})
        return {"observations": spool.preview, **result}


def run_fastalpr_baseline(input_path: Path, output_dir: Path, device: str = "cuda:0"):
    """Run FastALPR's documented detector+OCR wrapper as the still-image baseline."""
    from src.utils.onnx_runtime import prepare_onnx_runtime
    from src.utils.device import require_cuda
    from src.ingest.images import read_image
    import cv2

    require_cuda(device)
    prepare_onnx_runtime(device)
    from open_image_models.detection.core import hub as plate_hub
    from fast_plate_ocr.inference import hub as ocr_hub
    plate_hub.MODEL_CACHE_DIR = ROOT / "models" / "open-image-models"
    ocr_hub.MODEL_CACHE_DIR = ROOT / "models" / "fast-plate-ocr"
    from src.utils.model_manifest import release_artifact
    model_path = release_artifact("cct_s_v2")
    config_path = release_artifact("cct_s_v2_config")
    release_artifact("yolo_v9_s")
    from fast_alpr import ALPR
    providers = [("CUDAExecutionProvider", {"device_id": int(device.split(":", 1)[1]) if ":" in device else 0}), "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
    baseline = ALPR(
        detector_model="yolo-v9-s-608-license-plate-end2end",
        detector_conf_thresh=0.25,
        detector_providers=providers,
        ocr_model=None,
        ocr_device="cuda" if device.startswith("cuda") else "cpu",
        ocr_providers=providers,
        ocr_model_path=model_path,
        ocr_config_path=config_path,
    )
    frame = read_image(input_path)
    result = baseline.draw_predictions(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = output_dir / f"{input_path.stem}_fastalpr.png"
    if not cv2.imwrite(str(annotated), result.image):
        raise RuntimeError(f"Could not write FastALPR annotated image: {annotated}")
    rows = []
    for item in result.results:
        box = item.detection.bounding_box
        rows.append({"plate_bbox_xyxy": [box.x1, box.y1, box.x2, box.y2], "plate_detection_confidence": item.detection.confidence, "ocr_text": item.ocr.text if item.ocr else None, "ocr_character_scores": item.ocr.confidence if item.ocr else None})
    payload = {"source": str(input_path), "baseline": "FastALPR + YOLOv9-S-608 + CCT-S-v2-Global", "device": device, "observations": rows, "annotated_image": str(annotated), "ocr_scores_are_calibrated": False}
    json_path = output_dir / f"{input_path.stem}_fastalpr.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"annotated": str(annotated), "json": str(json_path), "observations": rows}


def download_models(include_yolo=True, smoke=False, research_ocr=False, paddle_ocr=False, awiros_ocr=False):
    """Fetch pretrained model artifacts only and create a reproducible manifest."""
    from huggingface_hub import snapshot_download
    from src.detectors.plate import HF_MODEL, HF_REVISION
    from src.utils.checksums import sha256_file
    cfg = load_config()
    models_dir = ROOT / "models"
    hf_id = HF_MODEL
    local_snapshot = Path(snapshot_download(hf_id, revision=HF_REVISION, local_files_only=os.getenv("ANPR_OFFLINE") == "1"))
    def project_path(path: str | Path) -> str:
        try:
            return Path(path).absolute().relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host_gpu": device_report(),
        "models": {
            "vehicle": {"id": "RF-DETR Large", "repository": "https://github.com/roboflow/rf-detr", "model_license": "Apache-2.0 per model documentation", "license_url": "https://roboflow.com/model-licenses/rf-detr", "weight_path": project_path(models_dir / "vehicle" / "rf-detr-large-2026.pth")},
            "plate_default": {"id": hf_id, "revision": HF_REVISION, "declared_license": "Apache-2.0", "model_card": f"https://huggingface.co/{hf_id}", "directory": project_path(local_snapshot), "files": []},
            "ocr": {"id": "cct-s-v2-global-model", "repository": "https://github.com/ankandrew/fast-plate-ocr", "code_license": "MIT", "weight_license": "separate artifact/data provenance requires review", "cache_path": project_path(models_dir / "fast-plate-ocr" / "cct-s-v2-global-model")},
        },
    }
    for path in sorted(local_snapshot.rglob("*")):
        if path.is_file() and path.name != ".cache":
            manifest["models"]["plate_default"]["files"].append({"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    if include_yolo:
        manifest["models"]["plate_challenger"] = {"id": "yolo-v9-s-608-license-plate-end2end", "repository": "https://github.com/ankandrew/open-image-models", "code_license": "MIT", "weight_license": "unresolved; internal evaluation only"}
    if research_ocr:
        research_id = "subash1652007/indian-plate-ocr"
        from src.recognition.indian_trocr_optional import MODEL_REVISION
        research_dir = Path(snapshot_download(research_id, revision=MODEL_REVISION, allow_patterns=["trocr_plate/**", "README.md"], local_files_only=os.getenv("ANPR_OFFLINE") == "1"))
        manifest["models"]["indian_trocr_research"] = {"id": research_id, "revision": MODEL_REVISION, "license": "non-commercial research and education only", "directory": project_path(research_dir / "trocr_plate"), "excluded_artifacts": ["yolo_best.pt (AGPL-3.0) not downloaded"]}
    from src.utils.model_manifest import release_artifact
    active_assets = ["rf_detr_large", "cct_s_v2", "cct_s_v2_config"]
    if include_yolo:
        active_assets.append("yolo_v9_s")
    catalog = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    for asset in active_assets:
        release_artifact(asset)
    manifest["release_assets"] = {key: catalog["artifacts"][key] for key in active_assets}
    if paddle_ocr:
        from src.recognition.paddleocr_fallback import paddle_assets
        paddle = paddle_assets(download=True)
        manifest["models"]["paddle_optional"] = {
            key: {**entry, "directory": project_path(entry["directory"])} for key, entry in paddle.items()
        }
    if awiros_ocr:
        from src.recognition.awiros_optional import awiros_assets
        awiros = awiros_assets(download=True)
        manifest["models"]["awiros_optional"] = {**awiros, "directory": project_path(awiros["directory"])}
    if smoke:
        # Actual model executions on a small neutral image confirm load/inference APIs and device placement.
        blank = 255 * (0 * __import__("numpy").ones((480, 640, 3), dtype="uint8"))
        from src.detectors.vehicle import VehicleDetector
        from src.detectors.plate import PlateDetector
        from src.recognition.fast_plate_ocr import FastPlateOCR
        detector = VehicleDetector(cfg["vehicle_threshold"], cfg["device"])
        vehicle_result = detector.predict(blank)
        plate = PlateDetector("rtdetr-v2-r50", cfg["plate_threshold"], cfg["device"], local_snapshot)
        plate_result = plate.predict(blank)
        ocr = FastPlateOCR("cct-s-v2-global-model", device=cfg["device"])
        ocr_result = ocr.predict(blank[:160, :420])
        manifest["smoke_inference"] = {
            "vehicle_model": {"inference_succeeded": True, "detections_on_blank": len(vehicle_result), "device": cfg["device"]},
            "plate_rtdetr": {"inference_succeeded": True, "detections_on_blank": len(plate_result), "device": cfg["device"]},
            "ocr_fast_plate": {"inference_succeeded": True, "returned_text": bool(ocr_result.get("raw")), "providers": ocr.recognizer.model.get_providers()},
            "results_are_accuracy_evidence": False,
        }
        if include_yolo:
            yolo = PlateDetector("yolo-v9-s-608", cfg["plate_threshold"], cfg["device"])
            manifest["smoke_inference"]["plate_yolov9"] = {"inference_succeeded": True, "detections_on_blank": len(yolo.predict(blank)), "providers": yolo.model.model.get_providers()}
        if research_ocr:
            from src.recognition.indian_trocr_optional import IndianTrOCR
            trocr = IndianTrOCR(device=cfg["device"], local_dir=str(research_dir / "trocr_plate"))
            manifest["smoke_inference"]["indian_trocr"] = {"inference_succeeded": True, "returned_text": bool(trocr.predict(blank[:160, :420]).get("raw")), "device": cfg["device"]}
        if awiros_ocr or paddle_ocr:
            from contextlib import ExitStack
            from src.recognition.awiros_optional import AwirosOCR
            from src.recognition.paddleocr_fallback import PaddleOCRFallback

            with ExitStack() as workers:
                for enabled, name, reader_type in ((awiros_ocr, "awiros", AwirosOCR),
                                                   (paddle_ocr, "paddleocr", PaddleOCRFallback)):
                    if enabled:
                        reader = reader_type()
                        workers.callback(reader.close)
                        output = reader.predict(blank[:160, :420])
                        manifest["smoke_inference"][name] = {"inference_succeeded": True, "returned_text": bool(output.get("raw")), "device": "cpu"}
    # Record every downloaded model weight in the project cache with SHA-256.
    from src.utils.checksums import sha256_file
    manifest["weight_files"] = []
    for path in sorted(models_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".onnx", ".safetensors", ".pdiparams"}:
            manifest["weight_files"].append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    import importlib.metadata as metadata
    manifest["package_versions"] = {name: metadata.version(name) for name in ["rfdetr", "transformers", "supervision", "fast-plate-ocr", "open-image-models", "fast-alpr", "onnxruntime-gpu"]}
    manifest_path = models_dir / "manifest.yaml"
    manifest_draft = manifest_path.with_suffix(".yaml.partial")
    manifest_draft.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    manifest_draft.replace(manifest_path)
    print(manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local pretrained-only Indian ANPR POC")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Show GPU, CUDA and ONNX Runtime availability")
    dl = sub.add_parser("download", help="Download pretrained models and create a provenance manifest")
    dl.add_argument("--no-yolo", action="store_true", help="Skip the evaluation-only YOLOv9 challenger")
    dl.add_argument("--smoke", action="store_true", help="Run real inference on a neutral blank image after download")
    dl.add_argument("--research-ocr", action="store_true", help="Also download and smoke-test the non-commercial Indian TrOCR reader (does not download AGPL plate detector)")
    dl.add_argument("--paddle-ocr", action="store_true", help="Install pinned PaddleOCR model artifacts into the local cache")
    dl.add_argument("--awiros", action="store_true", help="Install the pinned experimental Awiros OCR assets")
    for name in ("image", "images"):
        p = sub.add_parser(name, help="Process one image or all images in a folder")
        p.add_argument("--input", required=True, type=Path)
        p.add_argument("--output", default=Path(os.getenv("ANPR_OUTPUT_DIR", ROOT / "output")), type=Path)
        p.add_argument("--plate-model", choices=PLATE_MODELS)
        p.add_argument("--config", type=Path)
        p.add_argument("--profile", choices=("standard", "recovery"))
    baseline = sub.add_parser("baseline", help="Run FastALPR's still-image wrapper baseline")
    baseline.add_argument("--input", required=True, type=Path)
    baseline.add_argument("--output", default=Path(os.getenv("ANPR_OUTPUT_DIR", ROOT / "output")), type=Path)
    baseline.add_argument("--device", default=os.getenv("ANPR_DEVICE", "cuda:0"))
    for name in ("video", "stream"):
        p = sub.add_parser(name, help="Process a recorded video, webcam index or RTSP URL")
        p.add_argument("--input" if name == "video" else "--source", required=True, dest="input", type=str)
        p.add_argument("--output", default=Path(os.getenv("ANPR_OUTPUT_DIR", ROOT / "output")), type=Path)
        p.add_argument("--plate-model", choices=PLATE_MODELS)
        p.add_argument("--stride", type=int)
        p.add_argument("--max-frames", type=int)
        p.add_argument("--max-duration", type=float, help="Maximum wall-clock processing time in seconds")
        p.add_argument("--config", type=Path)
        p.add_argument("--profile", choices=("standard", "recovery"))
    sub.add_parser("ui", help="Start the local Gradio application")
    args = parser.parse_args(argv)
    setup_logging()
    if args.command == "doctor":
        report = device_report()
        print(json.dumps(report, indent=2))
        return 0 if report.get("torch_cuda_probe_ok") else 1
    if args.command == "download":
        download_models(not args.no_yolo, args.smoke, args.research_ocr, args.paddle_ocr, args.awiros)
        return 0
    if args.command in {"image", "images", "video", "stream"}:
        cfg = load_config(args.config)
        if args.profile:
            cfg = apply_profile(cfg, args.profile)
        plate_model = args.plate_model or cfg["plate_model"]
    if args.command in {"image", "images"}:
        if not args.input.exists():
            parser.error(f"Input does not exist: {args.input}")
        results = run_images(args.input, args.output, cfg, plate_model)
        return int(any("error" in item for item in results))
    if args.command == "baseline":
        if not args.input.is_file():
            parser.error(f"Input image does not exist: {args.input}")
        result = run_fastalpr_baseline(args.input, args.output, args.device)
        print(json.dumps(result, indent=2))
        return 0
    if args.command in {"video", "stream"}:
        import signal
        from threading import Event

        stop = Event()
        original = signal.signal(signal.SIGINT, lambda *_: stop.set())
        try:
            result = run_video_cli(args.input, args.output, cfg, plate_model, args.stride, args.max_frames,
                                   max_duration_s=args.max_duration, cancelled=stop.is_set)
        finally:
            signal.signal(signal.SIGINT, original)
        status = result["metadata"].get("status")
        return 130 if status == "CANCELLED" else int(status in {"PARTIAL_DECODE", "RESOURCE_LIMIT"})
    if args.command == "ui":
        from src.app import launch
        launch()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        sys.exit(1)
