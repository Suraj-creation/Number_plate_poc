from __future__ import annotations

import numpy as np


class ByteTracker:
    """Supervision's ByteTrack wrapper, assigned IDs on sampled frames."""

    def __init__(self, activation_threshold: float = 0.25, lost_track_buffer: int = 45, match_threshold: float = 0.8):
        import supervision as sv

        self.sv = sv
        self.tracker = sv.ByteTrack(track_activation_threshold=activation_threshold, lost_track_buffer=lost_track_buffer, minimum_matching_threshold=match_threshold)

    def update(self, vehicles: list[dict]) -> list[dict]:
        sv = self.sv
        if not vehicles:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []
        detections = sv.Detections(
            xyxy=np.asarray([v["bbox"] for v in vehicles], dtype=np.float32),
            confidence=np.asarray([v["score"] for v in vehicles], dtype=np.float32),
            class_id=np.asarray([v["class_id"] for v in vehicles], dtype=int),
            data={"input_index": np.arange(len(vehicles))},
        )
        tracked = self.tracker.update_with_detections(detections)
        ids = dict(zip(tracked.data.get("input_index", []), tracked.tracker_id))
        return [dict(vehicle, track_id=int(ids[index]) if index in ids else None)
                for index, vehicle in enumerate(vehicles)]
