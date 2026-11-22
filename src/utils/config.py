from __future__ import annotations

import math
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
PLATE_MODELS = ("rtdetr-v2-r50", "yolo-v9-s-608")
VIDEO_OVERLAYS = ("full", "plates")
OCR_MODELS = {"fast-plate-ocr": "cct-s-v2-global-model", "indian-trocr": "subash1652007/indian-plate-ocr",
              "awiros": "Awiros/anpr-ocr"}


def configure_caches() -> None:
    """Set project-local caches before model libraries are imported."""
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    load_dotenv(ROOT / ".env", override=False)
    defaults = {
        "HF_HOME": ROOT / "models" / "huggingface",
        "TORCH_HOME": ROOT / "models" / "torch",
        "XDG_CACHE_HOME": ROOT / "models" / "cache",
    }
    for key, path in defaults.items():
        os.environ.setdefault(key, str(path.resolve()))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    if os.getenv("ANPR_OFFLINE") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def apply_profile(cfg: dict, profile: str) -> dict:
    if profile not in ("standard", "recovery"):
        raise ValueError(f"Unknown processing profile: {profile}")
    return dict(cfg, recovery_views=profile == "recovery", plate_tiles=profile == "recovery",
                vehicle_tiles=profile == "recovery")


def validate_config(cfg: dict) -> dict:
    cfg.setdefault("video_overlay", "full")
    cfg.setdefault("paddle_temporal_recovery", False)
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", str(cfg["device"])):
        raise ValueError("device must be cpu, cuda, or cuda:<index>.")
    if cfg["device"] == "cuda":
        cfg["device"] = "cuda:0"
    if cfg["plate_model"] not in PLATE_MODELS or cfg["vehicle_model"] != "rfdetr-large":
        raise ValueError("Unknown vehicle or plate detector.")
    if cfg["ocr_engine"] not in OCR_MODELS:
        raise ValueError("Unknown OCR engine.")
    if cfg["ocr_engine"] == "fast-plate-ocr" and cfg["ocr_model"] != OCR_MODELS["fast-plate-ocr"]:
        raise ValueError("Unknown Fast-Plate-OCR model.")
    if cfg["ocr_engine"] != "fast-plate-ocr":
        cfg["ocr_model"] = OCR_MODELS[cfg["ocr_engine"]]
    for key, value in cfg.items():
        if key.endswith(("_threshold", "_margin", "_containment", "_fraction")):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be between 0 and 1.")
    for key in ("video_frame_stride", "track_history_size", "tracker_lost_track_buffer", "video_decode_limit",
                "min_plate_width_px", "min_plate_height_px", "max_frame_pixels", "max_batch_images", "ui_preview_rows",
                "track_min_evidence_gap_ms", "minimum_consensus_observations", "plate_inference_batch_size", "cpu_inference_threads", "video_decoder_threads"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("blur_laplacian_min", "min_plate_contrast"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and non-negative.")
    if type(cfg["min_output_free_mib"]) is not int or cfg["min_output_free_mib"] < 0:
        raise ValueError("min_output_free_mib must be a non-negative integer.")
    for key in ("save_plate_crops", "vehicle_tiles", "plate_tiles", "recovery_views", "paddle_ocr_fallback",
                "paddle_temporal_recovery", "allow_cpu_fallback", "awiros_review"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be true or false.")
    if cfg["plate_inference_batch_size"] > 8:
        raise ValueError("plate_inference_batch_size must be between 1 and 8.")
    if cfg["cpu_inference_threads"] > 32 or cfg["video_decoder_threads"] > 32:
        raise ValueError("CPU inference and decoder threads must be 1 to 32.")
    if cfg["paddle_ocr_fallback"] and cfg["ocr_engine"] != "fast-plate-ocr":
        raise ValueError("PaddleOCR second opinion requires the CCT-S primary reader.")
    if cfg["paddle_temporal_recovery"] and not cfg["paddle_ocr_fallback"]:
        raise ValueError("PaddleOCR temporal recovery requires paddle_ocr_fallback.")
    if cfg["awiros_review"] and cfg["ocr_engine"] != "fast-plate-ocr":
        raise ValueError("Awiros review requires the CCT-S primary reader.")
    if cfg["write_video_fps"] != "source":
        raise ValueError("Only source FPS is supported for annotated output.")
    if cfg["video_overlay"] not in VIDEO_OVERLAYS:
        raise ValueError("video_overlay must be full or plates.")
    if cfg["track_history_size"] < cfg["minimum_consensus_observations"]:
        raise ValueError("track_history_size must cover minimum_consensus_observations.")
    return cfg


def load_config(path: str | Path | None = None) -> dict:
    configure_caches()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    if path:
        overrides = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("Configuration must be a YAML mapping.")
        unknown = overrides.keys() - cfg.keys()
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        cfg.update(overrides)
    cfg["device"] = os.getenv("ANPR_DEVICE", cfg["device"])
    return validate_config(cfg)
