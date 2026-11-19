from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.config import ROOT, configure_caches


MODEL_ID = "subash1652007/indian-plate-ocr"
MODEL_REVISION = "82232c6220919c8c60d148ad74eb13dbea37e260"


class IndianTrOCR:
    """Non-commercial research challenger; model-card terms prohibit commercial use."""

    def __init__(self, device: str = "cuda:0", local_dir: str | None = None):
        from transformers import AutoImageProcessor, TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizerFast

        if local_dir:
            source = Path(local_dir)
        else:
            configure_caches()
            hf_home = Path(os.environ.get("HF_HOME", ROOT / "models" / "huggingface"))
            cached = hf_home / "hub" / "models--subash1652007--indian-plate-ocr" / "snapshots" / MODEL_REVISION / "trocr_plate"
            if cached.is_dir():
                source = cached
            else:
                from huggingface_hub import snapshot_download
                snapshot = snapshot_download(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                    allow_patterns=["trocr_plate/**", "README.md"],
                    local_files_only=os.getenv("ANPR_OFFLINE") == "1",
                )
                source = Path(snapshot) / "trocr_plate"
        from src.utils.model_manifest import verify_model_artifact
        model_weights = Path(source) / "model.safetensors"
        if model_weights.is_file():
            verify_model_artifact(model_weights)
        self.device = device
        # This checkpoint's tokenizer_config stores extra_special_tokens as a list;
        # Transformers 4.57 expects a mapping. Override only that tokenizer metadata in memory.
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(source, extra_special_tokens={})
        image_processor = AutoImageProcessor.from_pretrained(source)
        self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self.model = VisionEncoderDecoderModel.from_pretrained(source).to(device).eval()

    @torch.inference_mode()
    def predict(self, bgr: np.ndarray) -> dict:
        rgb = bgr[:, :, ::-1].copy()
        pixels = self.processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values.to(self.device)
        ids = self.model.generate(pixels, max_length=20, num_beams=3)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        # This model's generate API does not expose a calibrated read probability.
        return {"raw": text, "char_probs": None, "raw_score": None, "model": MODEL_ID}
