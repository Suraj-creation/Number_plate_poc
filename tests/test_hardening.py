import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

from eval.metrics import evaluate_rows, match_boxes
from src.ingest.images import read_image
from src.output.row_spool import RowSpool, write_csv
from src.pipeline.association import plate_association
from src.pipeline.geometry import deduplicate_plates, recovery_tiles
from src.recognition.crop_quality import assess_crop
from src.recognition.evidence import recognition_evidence, recovery_views
from src.recognition.india_format import validate_indian_format
from src.tracking.bytetrack import ByteTracker
from src.tracking.track_state import TrackState
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.association = {"state": "ASSOCIATED"}
        self.quality = {"too_small": False, "blurry": False, "contrast": 80}
        self.ocr = {"raw": "KA03MN1234", "raw_score": 0.92, "minimum_char_score": 0.82}
        self.detection = {"score": 0.75}

    def recognize(self, *, changes=None, clipped=False, alternatives=None):
        return recognition_evidence({**self.ocr, **(changes or {})}, self.quality, self.detection,
                                    self.association, clipped, alternatives or [], self.cfg)

    def test_weak_character_overrides_high_mean(self):
        self.assertEqual(self.recognize()[2], "READ")
        result = self.recognize(changes={"minimum_char_score": 0.15})
        self.assertEqual(result[2], "LOW_CONFIDENCE")
        self.assertFalse(result[3]["supports_track_read"])
        self.assertEqual(self.recognize(clipped=True)[2], "LOW_CONFIDENCE")
        self.assertEqual(self.recognize(changes={"raw": "KA03M_N1234"})[2], "LOW_CONFIDENCE")

    def test_independent_disagreement_and_transforms(self):
        alternative = {"normalized": "KA03MN1238", "raw_score": 0.94, "independent": True}
        self.assertEqual(self.recognize(alternatives=[alternative])[2], "LOW_CONFIDENCE")
        primary = {"raw": "", "raw_score": None, "minimum_char_score": None}
        alternative["independent"] = False
        text, _, status, _ = self.recognize(changes=primary, alternatives=[alternative])
        self.assertEqual((text, status), ("", "LOW_CONFIDENCE"))

    def test_indian_formats_keep_legacy_and_current_state_codes(self):
        for plate in ("DL1CAA1234", "DL04C1969", "TG13E0143", "TS13E0143", "22BH1234AA", "24BH2345L"):
            self.assertTrue(validate_indian_format(plate)["plausible"])
        for plate in ("DL0CAA1234", "24BH0000A", "24BH2345I", "24BH2345LO"):
            self.assertFalse(validate_indian_format(plate)["plausible"])

    def test_quality_measured_on_original_pixels(self):
        image = np.full((10, 47, 3), 25, dtype=np.uint8)
        result = assess_crop(image)
        self.assertTrue(result["too_small"] and result["blurry"])
        self.assertEqual(result["contrast"], 0)
        variants = recovery_views(image, result)
        self.assertEqual([name for name, _ in variants], ["local_contrast"])
        self.assertTrue(np.all(image == 25))


class TrackingAndGeometryTests(unittest.TestCase):
    def test_overlapping_vehicles_are_ambiguous_despite_confidence(self):
        box = [40, 40, 70, 57]
        vehicles = [{"bbox": [10, 10, 90, 80], "score": 0.95}, {"bbox": [30, 30, 80, 70], "score": 0.37}]
        result = plate_association(box, vehicles)
        self.assertEqual(result["state"], "AMBIGUOUS")
        self.assertIsNone(result["vehicle_index"])
        self.assertEqual(plate_association([95, 40, 120, 58], vehicles)["state"], "NO_CONTAINING_VEHICLE")

    def test_duplicate_detection_keeps_pass_evidence_and_nearby_plate(self):
        candidates = [{"bbox": box, "score": score, "source_pass": label, "detector_roi_xyxy": [0, 0, 200, 90]}
                      for box, score, label in (([10, 10, 70, 30], .9, "vehicle_crop"),
                                                ([11, 11, 70, 30], .8, "full_frame"),
                                                ([71, 10, 90, 30], .7, "full_frame"))]
        result = deduplicate_plates(candidates)
        self.assertEqual([len(row["detection_passes"]) for row in result], [2, 1])
        self.assertEqual(len(recovery_tiles(1920, 1080)), 4)
        self.assertEqual(recovery_tiles(640, 480), [])

    def test_tracker_returns_new_detection_without_an_id(self):
        tracker = ByteTracker()
        vehicle = {"bbox": [10, 10, 100, 100], "score": 0.95, "class_id": 2, "class_name": "car"}
        first = tracker.update([vehicle])[0]
        next_vehicle = {**vehicle, "bbox": [190, 190, 290, 290], "class_id": 3, "class_name": "motorcycle"}
        second = tracker.update([{**vehicle, "score": 0.20}, next_vehicle])
        self.assertEqual(second[0]["track_id"], first["track_id"])
        self.assertIsNone(second[1]["track_id"])
        self.assertEqual(second[1]["class_name"], "motorcycle")

    def test_track_needs_separated_unique_evidence_and_clear_winner(self):
        state = TrackState(6)
        vehicle = {"track_id": 1, "class_name": "car"}
        state.seen([vehicle], 0, 4)

        def add(frame, timestamp, fingerprint, text="KA03MN1234"):
            return state.add(1, dict(frame_index=frame, timestamp_ms=timestamp, fingerprint=fingerprint,
                                     ocr_normalized=text, quality_score=0.9, crop_readable=True,
                                     supports_track_read=True))

        self.assertEqual(add(0, 0, "a")["status"], "LOW_CONFIDENCE")
        self.assertEqual(add(1, 300, "a")["status"], "LOW_CONFIDENCE")
        self.assertEqual(add(2, 75, "b")["status"], "LOW_CONFIDENCE")
        self.assertEqual(add(3, 400, "c")["status"], "READ")
        self.assertEqual(add(4, 700, "d", "KA03MN123")["status"], "READ")
        self.assertEqual(add(5, 950, "e", "KA03MN123")["status"], "LOW_CONFIDENCE")
        self.assertEqual(state.seen([], 6, 4)[0]["recognition_status"], "LOW_CONFIDENCE")
        self.assertFalse(state.by_track)


class OperationsTests(unittest.TestCase):
    def test_partial_config_is_merged_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.yaml"
            path.write_text("video_frame_stride: 3\nplate_model: yolo-v9-s-608\n")
            cfg = load_config(path)
            self.assertEqual(cfg["video_frame_stride"], 3)
            self.assertEqual(cfg["plate_model"], "yolo-v9-s-608")
            self.assertIn("association_min_containment", cfg)
            path.write_text("video_frame_stride: 0\n")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_spool_preview_and_spreadsheet_export(self):
        with RowSpool(1) as spool, tempfile.TemporaryDirectory() as temporary:
            spool.add([{"source": "=1+1", "plate_bbox_xyxy": [1, 2, 3, 4]}, {"source": "image.jpg"}])
            self.assertEqual(len(spool.preview), 1)
            self.assertEqual(len(list(spool)), 2)
            target = Path(temporary) / "output.csv"
            write_csv(target, spool, spool.fields)
            with target.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["source"], "'=1+1")
            self.assertEqual(json.loads(rows[0]["plate_bbox_xyxy"]), [1, 2, 3, 4])

    def test_bad_and_oversized_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "photo.png"
            path.write_text("not an image")
            with self.assertRaisesRegex(ValueError, "Cannot decode"):
                read_image(path)
            cv2.imwrite(str(path), np.zeros((20, 20, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "pixel limit"):
                read_image(path, 100)

    def test_unannotated_frames_are_outside_metric_scope(self):
        obs = {"source": "clip", "frame_index": 0, "recognition_status": "READ", "ocr_normalized": "XX",
               "plate_bbox_xyxy": [0, 0, 10, 10], "vehicle_bbox_xyxy": [0, 0, 30, 30], "vehicle_track_id": 1}
        gt = {"source": "clip", "frame_index": 0, "ground_truth": "KA03MN1234",
              "plate_bbox_xyxy": "0,0,10,10", "vehicle_bbox_xyxy": "0,0,30,30", "vehicle_track_id": "ABC"}
        scores = evaluate_rows({"observations": [obs, dict(obs, frame_index=999)]}, [gt])
        self.assertEqual(scores["plate_predictions"], 1)
        self.assertEqual(scores["false_confident_reads"], 1)
        self.assertEqual(scores["association_accuracy"], 1.0)
        self.assertEqual(scores["vehicle_recall"], 1.0)
        self.assertEqual(scores["track_level_exact_match"], 0.0)
        self.assertEqual(len(match_boxes([[0, 0, 10, 10], [4, 0, 14, 10]],
                                         [[2, 0, 12, 10], [0, 0, 8, 10]], .5)), 2)


@unittest.skipUnless(os.getenv("RUN_ANPR_GPU_TESTS") == "1", "opt-in test with real local models and traffic media")
class GPUIntegrationTests(unittest.TestCase):
    def test_video_cancellation_export_and_repeated_image_batch(self):
        from src.cli import run_images, run_video_cli

        cfg = load_config()
        source = ROOT / "input/videos/india_traffic_acceptance_2m14s_3fps.mp4"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            state = {"progress": 0}

            def progress(value, desc):
                if desc.startswith("Frame"):
                    state["progress"] += 1

            result = run_video_cli(str(source), output, cfg, cfg["plate_model"], stride=2,
                                   max_frames=9, progress=progress, cancelled=lambda: state["progress"] >= 4)
            self.assertEqual(result["metadata"]["status"], "CANCELLED")
            self.assertEqual(result["metadata"]["decoded_frames"], 4)
            self.assertEqual(result["metadata"]["inferred_frames"], 2)
            rows = json.loads(Path(result["json"]).read_text())["observations"]
            self.assertTrue(rows)
            self.assertLessEqual({row["frame_index"] for row in rows}, {0, 2})
            with zipfile.ZipFile(result["zip"]) as archive:
                self.assertIsNone(archive.testzip())
            self.assertFalse(list(output.glob("*.partial.*")))
            capture = cv2.VideoCapture(str(source))
            try:
                ok, frame = capture.read()
            finally:
                capture.release()
            self.assertTrue(ok)
            still = output / "upload.png"
            cv2.imwrite(str(still), frame)
            corrupt = output / "broken.png"
            corrupt.write_bytes(b"invalid")
            results = run_images(output, output / "batch", dict(cfg, save_plate_crops=True), cfg["plate_model"],
                                 paths=[still, corrupt, still])
            self.assertEqual(len(results), 3)
            self.assertIn("error", results[1])
            self.assertNotEqual(results[0]["annotated"], results[2]["annotated"])
            plate = next(r for r in results[0]["observations"] if r["plate_bbox_xyxy"])
            with zipfile.ZipFile(results[0]["zip"]) as archive:
                self.assertIn(plate["crop_file"], archive.namelist())


if __name__ == "__main__":
    unittest.main()
