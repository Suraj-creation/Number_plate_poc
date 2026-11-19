from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np


def paddle_assets(download=False):
    import yaml

    from src.utils.checksums import sha256_file

    root = Path(__file__).resolve().parents[2]
    catalog = yaml.safe_load((root / "configs" / "models.yaml").read_text(encoding="utf-8"))["paddle_optional"]
    assets = {}
    for key, entry in catalog.items():
        directory = root / "models" / "paddle" / entry["repository"].split("/")[-1]
        if download and os.getenv("ANPR_OFFLINE") != "1":
            from huggingface_hub import snapshot_download

            snapshot_download(entry["repository"], revision=entry["revision"],
                              allow_patterns=list(entry["files"]), local_dir=directory)
        for filename, sha in entry["files"].items():
            file = directory / filename
            if not file.is_file():
                raise FileNotFoundError("Pinned PaddleOCR models are missing. Run download --paddle-ocr online.")
            if sha256_file(file) != sha:
                raise RuntimeError(f"PaddleOCR artifact checksum mismatch: {directory.name}/{filename}")
        assets[key] = {**entry, "directory": str(directory)}
    return assets


class PaddleOCRFallback:
    """Keep Paddle's CPU and OpenCV packages isolated from PyTorch and ORT."""

    def __init__(self):
        from src.recognition.ocr_worker import OCRWorker

        python = os.getenv("ANPR_PADDLE_PYTHON")
        if not python:
            raise RuntimeError("Set ANPR_PADDLE_PYTHON to the Python executable of the optional PaddleOCR environment.")
        self.assets = paddle_assets()
        self.worker = OCRWorker(
            python, "src.recognition.paddleocr_fallback", "ANPR_PADDLE",
            root=Path(__file__).resolve().parents[2],
            extra_env={"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
        )
        self.close = self.worker.close

    def predict(self, crop) -> dict:
        return self.worker.predict(crop)


def _worker():
    from paddleocr import PaddleOCR

    assets = paddle_assets()
    engine = PaddleOCR(device="cpu", cpu_threads=4,
                       text_detection_model_name=Path(assets["detector"]["directory"]).name,
                       text_recognition_model_name=Path(assets["recognizer"]["directory"]).name,
                       text_detection_model_dir=assets["detector"]["directory"],
                       text_recognition_model_dir=assets["recognizer"]["directory"],
                       use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    print("ANPR_PADDLE {}")
    for encoded in sys.stdin:
        crop = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
        texts, scores, lines = [], [], []
        for page in engine.predict(input=crop):
            data = getattr(page, "json", None)
            if callable(data):
                data = data()
            if not isinstance(data, dict):
                continue
            data = data.get("res", data)
            for text, score, box in zip(data.get("rec_texts", []), data.get("rec_scores", []),
                                        data.get("rec_boxes", [])):
                texts.append(str(text))
                scores.append(float(score))
                lines.append({"text": str(text), "score": float(score), "bbox": [int(c) for c in box]})
        result = {"raw": "".join(texts), "raw_score": float(np.mean(scores)) if scores else None,
                  "scores": scores, "lines": lines, "model": "PP-OCRv5 server detection + English mobile recognition"}
        print("ANPR_PADDLE " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    _worker()
