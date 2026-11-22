from __future__ import annotations

from collections import Counter
import hashlib
import time
from pathlib import Path

import cv2

from src.detectors.plate import PlateDetector
from src.detectors.vehicle import VehicleDetector
from src.output.overlays import annotate_plate_reads
from src.output.writers import safe_source
from src.pipeline.association import plate_association
from src.pipeline.geometry import box_iou, clip_box, deduplicate_plates, expand_box, map_detections, mapped_detections, recovery_tiles
from src.recognition.crop_quality import assess_crop
from src.recognition.evidence import recognition_evidence, recovery_views
from src.recognition.fast_plate_ocr import FastPlateOCR
from src.recognition.india_format import validate_indian_format
from src.recognition.normalize import normalize_text
from src.tracking.bytetrack import ByteTracker
from src.tracking.plate_reads import PlateReadTracker
from src.tracking.track_state import TrackState
from src.utils.device import require_cuda


class FramePipeline:
    def __init__(self, cfg: dict, plate_model: str | None = None, enable_tracking: bool = False):
        self.cfg = cfg
        import torch

        torch.set_num_threads(cfg["cpu_inference_threads"])
        cv2.setNumThreads(cfg["cpu_inference_threads"])
        require_cuda(cfg["device"])
        if cfg["device"].startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(cfg["device"] if ":" in cfg["device"] else "cuda:0")
        vehicle_threshold = min(cfg["vehicle_threshold"], cfg["vehicle_tracking_proposal_threshold"]) if enable_tracking else cfg["vehicle_threshold"]
        self.vehicle = VehicleDetector(vehicle_threshold, cfg["device"])
        self.plate_model_name = plate_model or cfg["plate_model"]
        self.plate = PlateDetector(self.plate_model_name, cfg["plate_threshold"], cfg["device"])
        if cfg["ocr_engine"] == "indian-trocr":
            from src.recognition.indian_trocr_optional import IndianTrOCR
            self.ocr = IndianTrOCR(device=cfg["device"])
        elif cfg["ocr_engine"] == "awiros":
            from src.recognition.awiros_optional import AwirosOCR
            self.ocr = AwirosOCR()
        else:
            self.ocr = FastPlateOCR(cfg["ocr_model"], device=cfg["device"])
        self.paddle = self.awiros_review = None
        if cfg["paddle_ocr_fallback"]:
            from src.recognition.paddleocr_fallback import PaddleOCRFallback
            self.paddle = PaddleOCRFallback()
        if cfg["awiros_review"]:
            from src.recognition.awiros_optional import AwirosOCR
            self.awiros_review = AwirosOCR()
        self.tracker = ByteTracker(cfg["tracker_track_activation_threshold"], cfg["tracker_lost_track_buffer"],
                                   cfg["tracker_match_threshold"]) if enable_tracking else None
        self.plate_read_tracker = PlateReadTracker(cfg) if enable_tracking and cfg.get("plate_track_recovery") else None
        self.track_state = TrackState(cfg["track_history_size"], cfg)
        self.frame_index = 0
        self.timings: list[dict] = []
        self.finished_tracks: list[dict] = []
        self.crop_dir: Path | None = None

    def close(self):
        try:
            for reader in (self.paddle, self.awiros_review, self.ocr):
                close_reader = getattr(reader, "close", None)
                if close_reader is not None:
                    close_reader()
        finally:
            self.paddle = self.awiros_review = self.ocr = self.plate = self.vehicle = None
            if self.cfg["device"].startswith("cuda"):
                import torch

                # Return unused allocations to other local GPU applications between runs.
                torch.cuda.empty_cache()

    def detect_vehicles(self, frame) -> list[dict]:
        vehicles = self.vehicle.predict(frame)
        if self.cfg["vehicle_tiles"]:
            height, width = frame.shape[:2]
            tiled = [result for roi in recovery_tiles(width, height)
                     for result in mapped_detections(self.vehicle, frame, roi, "tile")]
            for result in sorted(tiled, key=lambda item: item["score"], reverse=True):
                if not any(box_iou(result["bbox"], v["bbox"]) >= 0.45 for v in vehicles):
                    vehicles.append(result)
        return vehicles

    def detect_plates(self, frame, vehicles: list[dict]) -> list[dict]:
        height, width = frame.shape[:2]
        passes = [((0, 0, width, height), "full_frame")]
        if self.cfg["plate_tiles"]:
            passes.extend((roi, "tile") for roi in recovery_tiles(width, height))
        for vehicle in vehicles:
            roi = expand_box(vehicle["bbox"], self.cfg["vehicle_crop_margin"], width, height)
            passes.append((roi, "vehicle_crop"))
        detections = []
        visited = set()
        distinct_regions = []
        for roi, source_pass in passes:
            roi = tuple(roi)
            if roi in visited or roi[2] - roi[0] < 4 or roi[3] - roi[1] < 4:
                continue
            visited.add(roi)
            distinct_regions.append((roi, source_pass))
        batch_size = self.cfg["plate_inference_batch_size"]
        for index in range(0, len(distinct_regions), batch_size):
            regions = distinct_regions[index:index + batch_size]
            cropped = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2), _ in regions]
            results = self.plate.predict_many(cropped)
            for boxes, (roi, source_pass) in zip(results, regions, strict=True):
                detections.extend(map_detections(boxes, (height, width), roi, source_pass))
        return deduplicate_plates(detections)

    def process(self, frame, source: str, timestamp_ms: int = 0) -> tuple:
        h, w = frame.shape[:2]
        started = checkpoint = time.perf_counter()
        stages = {}

        def stage(name):
            nonlocal checkpoint
            now = time.perf_counter()
            stages[name] = round((now - checkpoint) * 1000, 3)
            checkpoint = now

        vehicles = self.detect_vehicles(frame)
        stage("vehicle_detection")
        if self.tracker:
            vehicles = [vehicle for vehicle in self.tracker.update(vehicles)
                        if vehicle["track_id"] is not None or vehicle["score"] >= self.cfg["vehicle_threshold"]]
            expiry = self.cfg["tracker_lost_track_buffer"] * self.cfg["video_frame_stride"]
            self.finished_tracks.extend(self.track_state.seen(vehicles, self.frame_index, expiry, timestamp_ms))
        else:
            vehicles = [dict(vehicle, track_id=None) for vehicle in vehicles]
        stage("tracking")
        plates = self.detect_plates(frame, vehicles)
        stage("plate_detection")

        associations = [plate_association(d["bbox"], vehicles, self.cfg["association_min_containment"],
                                          self.cfg["association_min_margin"]) for d in plates]
        credible_counts = Counter(a["vehicle_index"] for d, a in zip(plates, associations)
                                  if d["score"] >= self.cfg["plate_read_threshold"])
        associations = [dict(a, state="MULTIPLE_PLATES", vehicle_index=None)
                        if a["vehicle_index"] is not None and credible_counts[a["vehicle_index"]] > 1 else a
                        for a in associations]
        crops = []
        for d in plates:
            x1, y1, x2, y2 = expand_box(d["bbox"], self.cfg["plate_crop_margin"], w, h)
            crops.append(frame[y1:y2, x1:x2])
        primaries = []
        if isinstance(self.ocr, FastPlateOCR):
            for index in range(0, len(crops), 16):
                primaries.extend(self.ocr.predict_many(crops[index:index + 16]))
        else:
            primaries = [self.ocr.predict(crop) for crop in crops]

        rows = []
        associated_indices = set()
        source = safe_source(source)
        for d, association, crop, primary in zip(plates, associations, crops, primaries, strict=True):
            box = d["bbox"]
            x1, y1, x2, y2 = box
            clipped = x1 == 0 or y1 == 0 or x2 == w or y2 == h
            quality = assess_crop(frame[y1:y2, x1:x2], self.cfg["min_plate_width_px"],
                                  self.cfg["blur_laplacian_min"], self.cfg["min_plate_height_px"])
            recovery_crop = (not clipped and not quality["too_small"] and not quality["blurry"]
                             and quality["contrast"] >= self.cfg["min_plate_contrast"]
                             and d["score"] >= self.cfg["plate_read_threshold"])
            alternatives = self._alternatives(crop, primary, quality, recovery_crop)
            text, fmt, status, evidence = recognition_evidence(primary, quality, d, association,
                                                                clipped, alternatives, self.cfg)
            vehicle_index = association["vehicle_index"]
            vehicle = vehicles[vehicle_index] if vehicle_index is not None else None
            if vehicle_index is not None:
                associated_indices.add(vehicle_index)
            row = self._vehicle_row(source, timestamp_ms, vehicle, vehicle_index)
            track_id = row["vehicle_track_id"]
            vote = {}
            if track_id is not None:
                temporal_observation = {
                    "ocr_normalized": text if not self.cfg.get("paddle_temporal_recovery") or fmt["plausible"] else "",
                    "quality_score": quality["quality_score"],
                    "crop_readable": not (quality["too_small"] or quality["blurry"] or clipped),
                    "supports_track_read": evidence["supports_track_read"],
                    "fingerprint": hashlib.blake2b(crop.tobytes(), digest_size=12).hexdigest(),
                    "frame_index": self.frame_index, "timestamp_ms": int(timestamp_ms),
                }
                vote = self.track_state.add(track_id, temporal_observation)
                review_candidate = next((a["normalized"] for a in alternatives
                                         if a.get("confidence_role") == "review_only" and a["normalized"]), None)
                if review_candidate:
                    self.track_state.add_review(track_id, {**temporal_observation, "ocr_normalized": review_candidate,
                                                           "supports_track_read": False})
            row.update({
                "plate_bbox_xyxy": box, "plate_crop_bbox_xyxy": expand_box(box, self.cfg["plate_crop_margin"], w, h),
                "plate_source_pass": d["source_pass"], "plate_detection_passes": d["detection_passes"],
                "plate_detection_confidence": d["score"], "ocr_raw": primary.get("raw"), "ocr_normalized": text or None,
                "ocr_model_raw": primary.get("model_raw", primary.get("raw")),
                "ocr_raw_score": primary.get("raw_score"), "ocr_char_probs": primary.get("char_probs"),
                "ocr_slot_probs": primary.get("slot_probs"), "ocr_minimum_char_score": primary.get("minimum_char_score"),
                "ocr_alternatives": alternatives, "ocr_secondary": next((a["raw"] for a in alternatives if a["independent"]), None),
                "ocr_secondary_model": next((a["model"] for a in alternatives if a["independent"]), None),
                "ocr_secondary_agrees": evidence["independent_agreement"],
                "format_plausible": fmt["plausible"], "format_type": fmt["format"], "format_checked_text": text or None,
                "recognition_status": status, "recognition_evidence": evidence,
                "association_status": "ASSOCIATED" if vehicle else "UNASSOCIATED", "association_evidence": association,
                "crop_width_px": crop.shape[1], "crop_height_px": crop.shape[0], "crop_clipped": clipped,
                "plate_width_px": quality["width_px"], "plate_height_px": quality["height_px"],
                "laplacian_variance": quality["laplacian_variance"], "quality_score": round(quality["quality_score"], 4),
                "plate_contrast": quality["contrast"], "plate_median_intensity": quality["median_intensity"],
                "plate_shadow_fraction": round(quality["shadow_fraction"], 4),
                "plate_highlight_fraction": round(quality["highlight_fraction"], 4),
                "track_consensus": vote.get("text"), "track_support_count": vote.get("support", 0),
                "track_recognition_status": vote.get("status"), "track_evidence_frames": vote.get("evidence_frames", []),
            })
            rows.append(row)
            if self.crop_dir is not None:
                self.crop_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                filename = f"frame_{self.frame_index:08d}_plate_{len(rows):03d}.png"
                if not cv2.imwrite(str(self.crop_dir / filename), crop):
                    raise OSError("Could not save original plate crop.")
                row["crop_file"] = f"{self.crop_dir.name}/{filename}"
        for index, vehicle in enumerate(vehicles):
            if index not in associated_indices:
                rows.append(self._vehicle_row(source, timestamp_ms, vehicle, index))
        for track_id in {r["vehicle_track_id"] for r in rows
                         if r["recognition_status"] == "READ" and r["vehicle_track_id"] is not None}:
            self.track_state.active[track_id]["reads"] += 1
        if self.plate_read_tracker is not None:
            self.plate_read_tracker.update(frame, rows, self.frame_index, int(timestamp_ms))
        stage("ocr")
        if self.tracker is not None and self.cfg.get("video_overlay") == "plates":
            annotated = annotate_plate_reads(frame, rows)
        else:
            annotated = self._annotate(frame, vehicles, rows)
        stages["total"] = round((time.perf_counter() - started) * 1000, 3)
        self.timings.append(stages)
        for row in rows:
            row["stage_latency_ms"] = stages
        self.frame_index += 1
        return annotated, rows, vehicles

    def _alternatives(self, crop, primary: dict, quality: dict, recovery_crop: bool = True) -> list[dict]:
        score = primary.get("raw_score") or 0
        plausible = validate_indian_format(normalize_text(primary.get("raw")))["plausible"]
        alternatives = []
        if not plausible or score < self.cfg["ocr_read_threshold"] or (
                primary.get("minimum_char_score") or 0) < self.cfg["ocr_min_char_threshold"]:
            if self.cfg["recovery_views"]:
                for label, pixels in recovery_views(crop, quality):
                    alternatives.append((label, self.ocr.predict(pixels), False))
            if self.paddle and (recovery_crop or not self.cfg.get("paddle_temporal_recovery")) and (
                                not plausible or score < self.cfg["paddle_ocr_trigger_score"]
                                or (primary.get("minimum_char_score") or 0) < self.cfg["ocr_min_char_threshold"]):
                alternatives.append(("original", self.paddle.predict(crop), True))
            if self.awiros_review:
                prediction = self.awiros_review.predict(crop)
                alternatives.append(("original", {**prediction, "confidence_role": "review_only"}, True))
        return [{**prediction, "view": label, "independent": independent,
                 "normalized": normalize_text(prediction.get("raw")) or None}
                for label, prediction, independent in alternatives]

    def _vehicle_row(self, source, timestamp_ms, vehicle, index):
        return {
            "source": source, "frame_index": self.frame_index, "timestamp_ms": int(timestamp_ms),
            "vehicle_index": index, "vehicle_track_id": vehicle["track_id"] if vehicle else None,
            "vehicle_class": vehicle["class_name"] if vehicle else None,
            "vehicle_tracking_status": ("TRACKED" if vehicle["track_id"] is not None else "DETECTED") if vehicle else None,
            "vehicle_confidence": vehicle["score"] if vehicle else None,
            "vehicle_bbox_xyxy": [round(float(x), 2) for x in vehicle["bbox"]] if vehicle else None,
            "plate_bbox_xyxy": None, "plate_detection_model": self.plate_model_name,
            "plate_detection_confidence": None, "ocr_model": self.cfg["ocr_model"],
            "ocr_raw": None, "ocr_normalized": None, "ocr_raw_score": None,
            "ocr_char_probs": None, "track_consensus": None, "track_support_count": 0,
            "format_plausible": False, "format_type": "none",
            "recognition_status": "NO_PLATE_DETECTED", "association_status": "NO_PLATE_DETECTED",
        }

    @staticmethod
    def _annotate(frame, vehicles, rows):
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        for vehicle in vehicles:
            x1, y1, x2, y2 = clip_box(vehicle["bbox"], width, height)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 180, 70), 2)
            label = f"{vehicle['class_name']} {vehicle['score']:.2f}"
            if vehicle["track_id"] is not None:
                label += f" ID:{vehicle['track_id']}"
            cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 180, 70), 2, cv2.LINE_AA)
        colors = {"READ": (0, 210, 255), "UNASSOCIATED": (240, 160, 70), "UNREADABLE": (180, 180, 180)}
        for row in rows:
            if row["plate_bbox_xyxy"] is None:
                continue
            x1, y1, x2, y2 = row["plate_bbox_xyxy"]
            status = row["recognition_status"]
            color = colors.get(status, (0, 150, 255))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            text = row["ocr_normalized"] or "PLATE"
            label = f"{text}  {status}"
            if row["ocr_raw_score"] is not None:
                label += f"  OCR {row['ocr_raw_score']:.2f}"
            text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
            tx = max(0, min(x1, width - text_width - 2))
            cv2.putText(canvas, label, (tx, min(height - 4, y2 + 17)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        return canvas
