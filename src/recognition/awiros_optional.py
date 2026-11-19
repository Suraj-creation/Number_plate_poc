"""Pinned, explicitly opt-in Awiros inference inside the Paddle worker runtime."""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def awiros_assets(download=False) -> dict:
    import yaml

    from src.utils.checksums import sha256_file
    catalog = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["awiros_optional"]
    directory = ROOT / "models" / "awiros"
    if download and os.getenv("ANPR_OFFLINE") != "1":
        from huggingface_hub import snapshot_download

        snapshot_download(catalog["repository"], revision=catalog["revision"],
                          allow_patterns=[*catalog["files"], "LICENSE"], local_dir=directory)
    for filename, checksum in catalog["files"].items():
        file = directory / filename
        if not file.is_file() or sha256_file(file) != checksum:
            raise RuntimeError(f"Awiros asset unavailable or failed verification: {filename}. Run download --awiros online.")
    return {**catalog, "directory": str(directory)}


class AwirosOCR:
    def __init__(self):
        from src.recognition.ocr_worker import OCRWorker
        python = os.getenv("ANPR_PADDLE_PYTHON")
        if not python:
            raise RuntimeError("Set ANPR_PADDLE_PYTHON to the optional Paddle environment.")
        self.worker = OCRWorker(
            python, "src.recognition.awiros_optional", "ANPR_AWIROS", root=ROOT,
            extra_env={"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
        )
        self.close = self.worker.close

    def predict(self, crop):
        return self.worker.predict(crop)


def load_awiros():
    import cv2
    import numpy as np
    import paddle

    assets = awiros_assets()
    directory = Path(assets["directory"])
    raw_source = os.getenv("ANPR_AWIROS_PADDLEOCR_DIR", "")
    if not raw_source:
        raise RuntimeError("Set ANPR_AWIROS_PADDLEOCR_DIR to the pinned PaddleOCR source checkout.")
    source = Path(raw_source).resolve()
    current = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    changes = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True, timeout=10)
    if current != assets["paddle_source_revision"] or changes:
        raise RuntimeError("Awiros requires a clean PaddleOCR source checkout at the pinned revision.")
    sys.path.insert(0, str(source))
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    spec = importlib.util.spec_from_file_location("awiros_pinned_reference", directory / "test.py")
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    paddle.set_device("cpu")
    cv2.setNumThreads(2)
    model = build_model(copy.deepcopy(vendor.MODEL_CONFIG["Architecture"]))
    missing, unexpected = model.set_state_dict(vendor.load_safetensors_to_paddle(paddle, directory / "model.safetensors"))
    if missing or unexpected:
        raise RuntimeError("Awiros weights did not match the pinned architecture.")
    model.eval()
    decoder = build_post_process({"name": "CTCLabelDecode", "character_dict_path": str(directory / "en_dict.txt"),
                                  "use_space_char": True})
    def predict_many(crops):
        batch = np.stack([vendor.preprocess(crop, vendor.IMAGE_SHAPE) for crop in crops])
        with paddle.no_grad():
            result = model(paddle.to_tensor(batch))
        output = result["ctc"] if isinstance(result, dict) else result
        return [{"raw": "" if text.strip().upper() == "<UNK>" else text, "model_raw": text,
                 "model": assets["repository"], "raw_score": float(score) if np.isfinite(score) else None,
                 "minimum_char_score": None, "abstained": text.strip().upper() == "<UNK>"}
                for text, score in decoder(output.numpy())]

    return predict_many


def _worker():
    import cv2
    import numpy as np

    predict_many = load_awiros()
    print("ANPR_AWIROS {}")
    for encoded in sys.stdin:
        crop = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
        response = predict_many([crop])[0]
        print("ANPR_AWIROS " + json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    _worker()
