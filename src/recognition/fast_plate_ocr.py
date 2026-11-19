from __future__ import annotations

import numpy as np

from src.utils.device import CudaUnavailable
from src.utils.onnx_runtime import make_session_options, prepare_onnx_runtime


class FastPlateOCR:
    def __init__(self, model_name: str = "cct-s-v2-global-model", device: str = "cuda"):
        from fast_plate_ocr import LicensePlateRecognizer

        self.model_name = model_name
        providers = None
        if device.startswith("cuda:"):
            providers = [("CUDAExecutionProvider", {"device_id": int(device.partition(":")[2] or 0)}), "CPUExecutionProvider"]
            device = "cuda"
        if device == "cuda":
            prepare_onnx_runtime(device)
        from src.utils.config import OCR_MODELS
        from src.utils.model_manifest import release_artifact

        if model_name != OCR_MODELS["fast-plate-ocr"]:
            raise ValueError(f"Unknown Fast-Plate-OCR model: {model_name}")
        model_path = release_artifact("cct_s_v2")
        config_path = release_artifact("cct_s_v2_config")
        self.recognizer = LicensePlateRecognizer(onnx_model_path=model_path, plate_config_path=config_path, device=device, providers=providers, sess_options=make_session_options())
        if device == "cuda" and "CUDAExecutionProvider" not in self.recognizer.model.get_providers():
            raise CudaUnavailable("Fast-Plate-OCR did not activate ONNX CUDAExecutionProvider.")

    def predict(self, bgr: np.ndarray) -> dict:
        return self.predict_many([bgr])[0]

    def predict_many(self, crops: list[np.ndarray]) -> list[dict]:
        predictions = self.recognizer.run([crop[:, :, ::-1].copy() for crop in crops], return_confidence=True)
        results = []
        for pred in predictions:
            raw = pred.plate or ""
            scores = np.asarray(pred.char_probs, dtype=float).reshape(-1)
            # The recognizer strips trailing pad symbols but returns every slot score.
            chars = scores[:len(raw)].tolist() if raw and np.isfinite(scores).all() else []
            results.append({
                "raw": raw, "char_probs": chars or None, "slot_probs": scores.tolist() if np.isfinite(scores).all() else None,
                "raw_score": float(np.mean(chars)) if chars else None,
                "minimum_char_score": min(chars) if chars else None, "model": self.model_name,
            })
        return results
