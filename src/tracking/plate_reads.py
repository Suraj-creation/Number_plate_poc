from __future__ import annotations

import hashlib

from src.tracking.bytetrack import ByteTracker
from src.tracking.track_state import TrackState


class PlateReadTracker:
    """Use plate motion and repeated crop evidence when vehicle association is uncertain."""

    def __init__(self, cfg: dict):
        self.read_threshold = cfg["plate_read_threshold"]
        # Small plate boxes can move quickly. Keep the session short after an occlusion.
        lost_buffer = min(4, cfg["tracker_lost_track_buffer"])
        self.expiry = lost_buffer * cfg["video_frame_stride"]
        self.tracker = ByteTracker(self.read_threshold, lost_buffer, cfg["tracker_match_threshold"])
        self.state = TrackState(cfg["track_history_size"], cfg)

    def update(self, frame, rows: list[dict], frame_index: int, timestamp_ms: int) -> None:
        plate_rows = [row for row in rows if row["plate_bbox_xyxy"] is not None
                      and row["plate_detection_confidence"] >= self.read_threshold]
        detections = [{"bbox": row["plate_bbox_xyxy"], "score": row["plate_detection_confidence"],
                       "class_id": 0, "class_name": "plate"} for row in plate_rows]
        tracks = self.tracker.update(detections)
        self.state.seen(tracks, frame_index, self.expiry, timestamp_ms)
        for row, track in zip(plate_rows, tracks, strict=True):
            track_id = track["track_id"]
            if track_id is None:
                continue
            text = row["ocr_normalized"]
            supported = row["recognition_evidence"]["supports_plate_consensus"]
            observation = {"ocr_normalized": text if supported else None, "supports_track_read": supported,
                           "quality_score": row["quality_score"], "frame_index": frame_index,
                           "timestamp_ms": timestamp_ms, "crop_readable": supported}
            if supported:
                x1, y1, x2, y2 = row["plate_crop_bbox_xyxy"]
                observation["fingerprint"] = hashlib.blake2b(frame[y1:y2, x1:x2].tobytes(), digest_size=12).hexdigest()
            vote = self.state.add(track_id, observation)
            row.update(plate_track_id=track_id, plate_track_consensus=vote["text"],
                       plate_track_support_count=vote["support"], plate_track_status=vote["status"],
                       plate_track_evidence_frames=vote["evidence_frames"])
