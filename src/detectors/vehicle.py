from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

COCO_VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}
COCO_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def clip_bbox_xyxy(box, width: int, height: int) -> list[float]:
    """Keep detector boxes inside the source image while preserving subpixels."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


class VehicleDetector:
    """RF-DETR COCO detector filtered to the required motor-vehicle classes."""

    def __init__(self, threshold: float = 0.35, device: str = "cuda:0"):
        from rfdetr import RFDETRLarge
        import torch

        self.threshold, self.device = threshold, device
        rfdetr_device = "cuda" if device.startswith("cuda") else device
        if rfdetr_device == "cuda" and ":" in device:
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        from src.utils.model_manifest import release_artifact
        weights = release_artifact("rf_detr_large")
        self.model = RFDETRLarge(device=rfdetr_device, pretrain_weights=str(weights))
        self.class_names = getattr(self.model, "class_names", None)
        log.info("Loaded RF-DETR Large on %s", device)

    def predict(self, bgr: np.ndarray) -> list[dict]:
        height, width = bgr.shape[:2]
        rgb = bgr[:, :, ::-1].copy()
        detections = self.model.predict(rgb, threshold=self.threshold)
        boxes = np.asarray(detections.xyxy)
        scores = np.asarray(detections.confidence)
        ids = np.asarray(detections.class_id)
        out = []
        for box, score, class_id in zip(boxes, scores, ids):
            name = self._name(int(class_id))
            if name not in COCO_VEHICLE_NAMES or not np.isfinite(box).all() or not np.isfinite(score):
                continue
            bbox = clip_bbox_xyxy(box, width, height)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            out.append({"bbox": bbox, "score": float(score), "class_id": int(class_id), "class_name": name})
        return out

    def _name(self, class_id: int) -> str:
        names = self.class_names
        if isinstance(names, dict):
            value = names.get(class_id, names.get(str(class_id)))
            if value:
                return str(value).lower()
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id]).lower()
        # RF-DETR's COCO category IDs are the canonical sparse COCO IDs.
        return COCO_IDS.get(class_id, "unknown")
