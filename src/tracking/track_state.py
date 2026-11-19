from __future__ import annotations

from collections import Counter, defaultdict


class TrackState:
    def __init__(self, max_observations: int = 12, cfg: dict | None = None):
        self.max_observations = max_observations
        self.cfg = cfg or {}
        self.by_track: dict[int, list[dict]] = defaultdict(list)
        self.review_by_track: dict[int, list[dict]] = defaultdict(list)
        self.active: dict[int, dict] = {}

    def seen(self, vehicles: list[dict], frame_index: int, expiry_frames: int, timestamp_ms: int | None = None) -> list[dict]:
        for vehicle in vehicles:
            track_id = vehicle["track_id"]
            if track_id is not None:
                track = self.active.setdefault(track_id, {"classes": Counter(), "frames": 0, "reads": 0,
                                                          "first_frame": frame_index, "first_seen_ms": timestamp_ms})
                track["classes"][vehicle["class_name"]] += 1
                track["frames"] += 1
                track["last_frame"] = frame_index
                track["last_seen_ms"] = timestamp_ms
                if track.get("stable_text") and not self._consistent_class(track):
                    track["stable_text"] = track["stable_read_ms"] = None
        expired = [key for key, track in self.active.items() if frame_index - track["last_frame"] > expiry_frames]
        return [self.finish(key, "TRACK_LOST") for key in expired]

    def finish(self, track_id: int, end_reason: str = "END_OF_INPUT") -> dict:
        track = self.active.pop(track_id)
        history = self.by_track.pop(track_id, [])
        result = self.vote(history)
        review_history = self.review_by_track.pop(track_id, [])
        review = self.vote(review_history)
        vehicle_class, class_count = track["classes"].most_common(1)[0]
        class_fraction = class_count / track["frames"]
        status = result["status"]
        if status == "READ" and not self._consistent_class(track):
            status = "LOW_CONFIDENCE"
        return {
            "vehicle_track_id": track_id, "vehicle_class": vehicle_class,
            "dominant_class_fraction": round(class_fraction, 4),
            "observations": track["frames"], "read_observations": track["reads"],
            "first_frame": track["first_frame"], "last_frame": track["last_frame"],
            "first_seen_ms": track["first_seen_ms"], "last_seen_ms": track.get("last_seen_ms"),
            "track_end_reason": end_reason,
            "stable_read_ms": (track.get("stable_read_ms") if status == "READ" else None),
            "consensus": result["text"], "consensus_support": result["support"],
            "consensus_margin": result["margin"], "consensus_fraction": result["fraction"],
            "recognition_status": status, "evidence_frame_indices": result["evidence_frames"],
            "candidates": dict(Counter(o["ocr_normalized"] for o in history if o.get("ocr_normalized"))),
            "best_frame_index": next((o.get("frame_index") for o in history
                                      if result["text"] and o.get("ocr_normalized") == result["text"]), None),
            "review_ocr_consensus": review["text"], "review_ocr_support": review["support"],
            "review_ocr_frames": sorted(row["frame_index"] for row in review_history
                                       if row["ocr_normalized"] == review["text"]),
        }

    def add(self, track_id: int, observation: dict) -> dict:
        self._remember(self.by_track[track_id], observation)
        return self._record_vote(track_id, observation)

    def add_review(self, track_id: int, observation: dict) -> None:
        self._remember(self.review_by_track[track_id], observation)

    def _remember(self, history, observation):
        frame_index = observation.get("frame_index")
        previous = next((i for i, row in enumerate(history) if row.get("frame_index") == frame_index), None) if frame_index is not None else None
        if previous is not None:
            if history[previous].get("quality_score", 0) >= observation.get("quality_score", 0):
                return
            history.pop(previous)
        if observation.get("ocr_normalized"):
            fingerprint = observation.get("fingerprint")
            duplicate = next((i for i, row in enumerate(history) if row.get("fingerprint") == fingerprint
                              and row["ocr_normalized"] == observation["ocr_normalized"]), None) if fingerprint else None
            if duplicate is not None:
                return
            history.append({key: value for key, value in observation.items() if key != "crop"})
        history.sort(key=lambda row: row.get("quality_score", 0), reverse=True)
        del history[self.max_observations:]

    def _record_vote(self, track_id: int, observation: dict) -> dict:
        result = self.vote(self.by_track[track_id])
        track = self.active.get(track_id)
        if track is not None:
            if result["status"] == "READ" and not self._consistent_class(track):
                result["status"] = "LOW_CONFIDENCE"
            stable_text = result["text"] if result["status"] == "READ" else None
            if track.get("stable_text") != stable_text:
                track["stable_text"] = stable_text
                track["stable_read_ms"] = observation.get("timestamp_ms") if stable_text else None
        return result

    def _consistent_class(self, track):
        return max(track["classes"].values()) / track["frames"] >= self.cfg.get("track_min_class_fraction", 0.75)

    def vote(self, history: list[dict]) -> dict:
        weights = defaultdict(float)
        counts = Counter()
        good_counts = Counter()
        evidence = defaultdict(list)
        # Keep competing full strings, including strings of different lengths.
        # Exact duplicate crops cannot independently add weight to a result.
        seen = set()
        for row in history:
            text = row["ocr_normalized"]
            fingerprint = row.get("fingerprint")
            key = (text, fingerprint)
            if fingerprint is not None and key in seen:
                continue
            seen.add(key)
            weights[text] += max(0.01, row.get("quality_score", 0.01))
            counts[text] += 1
            if row.get("crop_readable"):
                good_counts[text] += 1
            if row.get("supports_track_read"):
                evidence[text].append((row["timestamp_ms"], row["frame_index"]))
        ranked = sorted(weights, key=lambda text: (-weights[text], text))
        text = ranked[0] if ranked else None
        top = weights[text] if text else 0.0
        runner_up = weights[ranked[1]] if len(ranked) > 1 else 0.0
        margin = (top - runner_up) / top if top else 0.0
        fraction = top / sum(weights.values()) if top else 0.0
        evidence_frames = []
        previous_time = float("-inf")
        for timestamp, frame in sorted(evidence[text]):
            if timestamp - previous_time >= self.cfg.get("track_min_evidence_gap_ms", 200):
                evidence_frames.append(frame)
                previous_time = timestamp
        read = (len(evidence_frames) >= self.cfg.get("minimum_consensus_observations", 2)
                and margin >= self.cfg.get("track_min_consensus_margin", 0.35)
                and fraction >= self.cfg.get("track_min_consensus_fraction", 0.70))
        return {"text": text, "support": counts[text], "good_support": good_counts[text],
                "margin": round(margin, 4), "fraction": round(fraction, 4), "weight": top,
                "evidence_frames": evidence_frames,
                "status": "READ" if read else "LOW_CONFIDENCE" if text else "UNREADABLE"}
