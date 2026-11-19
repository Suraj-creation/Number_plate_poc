from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

from src.utils.config import ROOT
from src.utils.device import CudaUnavailable
from src.utils.onnx_runtime import make_session_options, prepare_onnx_runtime

log = logging.getLogger(__name__)
HF_MODEL = "justjuu/rtdetr-v2-license-plate-detection"
HF_REVISION = "0ca7598f58649fe4a27db22ad72dca7be70314b1"


class PlateDetector:
    def __init__(self, model_name: str = "rtdetr-v2-r50", threshold: float = 0.25, device: str = "cuda:0", model_dir: str | Path | None = None):
        self.model_name, self.threshold, self.device = model_name, threshold, device
        if model_name == "rtdetr-v2-r50":
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            from huggingface_hub import snapshot_download

            source = str(model_dir) if model_dir else str(snapshot_download(
                HF_MODEL,
                revision=HF_REVISION,
                local_files_only=os.getenv("ANPR_OFFLINE") == "1",
            ))
            from src.utils.model_manifest import verify_model_artifact
            weight_path = Path(source) / "model.safetensors"
            if weight_path.is_file():
                verify_model_artifact(weight_path)
            for name in ("config.json", "preprocessor_config.json"):
                verify_model_artifact(Path(source) / name)
            self.processor = AutoImageProcessor.from_pretrained(source)
            self.model = AutoModelForObjectDetection.from_pretrained(source, use_safetensors=True).to(device).eval()
        elif model_name == "yolo-v9-s-608":
            from open_image_models import create_detector
            from open_image_models.detection.core import hub

            self.processor = None
            prepare_onnx_runtime(device)
            hub.MODEL_CACHE_DIR = ROOT / "models" / "open-image-models"
            from src.utils.model_manifest import release_artifact
            release_artifact("yolo_v9_s")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
            if device.startswith("cuda:"):
                providers[0] = ("CUDAExecutionProvider", {"device_id": int(device.partition(":")[2] or 0)})
            self.model = create_detector("yolo-v9-s-608-license-plate-end2end", conf_thresh=threshold, providers=providers, sess_options=make_session_options())
            if device.startswith("cuda") and "CUDAExecutionProvider" not in self.model.model.get_providers():
                raise CudaUnavailable("YOLOv9 plate model did not activate ONNX CUDAExecutionProvider.")
        else:
            raise ValueError(f"Unknown plate model: {model_name}")
        log.info("Loaded plate detector %s on %s", model_name, device)

    def predict(self, bgr: np.ndarray) -> list[dict]:
        return self.predict_many([bgr])[0]

    @torch.inference_mode()
    def predict_many(self, images: list[np.ndarray]) -> list[list[dict]]:
        if self.model_name != "rtdetr-v2-r50":
            return [self._predict_yolo(bgr) for bgr in images]
        sizes = [bgr.shape[:2] for bgr in images]
        rgb = [bgr[:, :, ::-1].copy() for bgr in images]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        output = self.model(**inputs)
        results = self.processor.post_process_object_detection(output, target_sizes=sizes, threshold=self.threshold)
        return [
            [{"bbox": box, "score": score, "model": self.model_name}
             for box, score in zip(result["boxes"].tolist(), result["scores"].tolist(), strict=True)]
            for result in results
        ]

    def _predict_yolo(self, bgr: np.ndarray) -> list[dict]:
        """Normalize open-image-models output; its prediction schema varies by release."""
        result = self.model.predict(bgr)
        if hasattr(result, "xyxy"):
            boxes = np.asarray(result.xyxy)
            scores = np.asarray(getattr(result, "confidence", np.ones(len(boxes))))
            return [{"bbox": [float(v) for v in box], "score": float(score), "model": "yolo-v9-s-608"} for box, score in zip(boxes, scores) if float(score) >= self.threshold]
        if isinstance(result, (list, tuple)) and len(result) == 1 and isinstance(result[0], (list, tuple)):
            result = result[0]
        rows = getattr(result, "detections", result)
        out = []
        for row in rows:
            if isinstance(row, dict):
                box = row.get("bbox", row.get("xyxy", row.get("box")))
                score = row.get("confidence", row.get("score", 1.0))
            else:
                box = getattr(row, "bbox", getattr(row, "xyxy", getattr(row, "bounding_box", None)))
                score = getattr(row, "confidence", getattr(row, "score", 1.0))
            if box is not None and float(score) >= self.threshold:
                if all(hasattr(box, attr) for attr in ("x1", "y1", "x2", "y2")):
                    arr = np.asarray([box.x1, box.y1, box.x2, box.y2], dtype=float)
                else:
                    arr = np.asarray(box, dtype=float).reshape(-1)
                if len(arr) >= 4:
                    out.append({"bbox": [float(v) for v in arr[:4]], "score": float(score), "model": "yolo-v9-s-608"})
        return out
