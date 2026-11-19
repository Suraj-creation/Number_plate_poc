import unittest
import tempfile
import zipfile
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.association import associate_plate
from src.recognition.india_format import validate_indian_format
from src.recognition.normalize import normalize_text
from src.tracking.track_state import TrackState
from src.pipeline.frame_pipeline import clip_box, expand_box
from src.detectors.vehicle import clip_bbox_xyxy
from src.tracking.bytetrack import ByteTracker
from src.output.writers import safe_source, safe_source_stem, write_results
from src.utils.checksums import sha256_file
from src.utils.model_manifest import verify_model_artifact
from eval.metrics import evaluate


class CoreTests(unittest.TestCase):
    def test_association_requires_geometric_containment(self):
        vehicle = {"bbox": [100, 100, 400, 400], "score": 0.9, "track_id": 4}
        self.assertEqual(associate_plate([180, 300, 260, 330], [vehicle]), vehicle)
        self.assertIsNone(associate_plate([410, 300, 480, 330], [vehicle]))

    def test_indian_format_patterns(self):
        self.assertTrue(validate_indian_format("KA03MN1234")["plausible"])
        self.assertTrue(validate_indian_format("22BH1234AA")["plausible"])
        self.assertTrue(validate_indian_format("OR02AB1234")["plausible"])
        self.assertFalse(validate_indian_format("20BH1234AA")["plausible"])
        self.assertFalse(validate_indian_format("ZZ03MN1234")["plausible"])
        self.assertFalse(validate_indian_format("KAO3MN1234")["plausible"])

    def test_normalization_preserves_alphanumeric_model_output(self):
        self.assertEqual(normalize_text("ka 03-mn 1234"), "KA03MN1234")

    def test_track_vote_uses_full_strings_and_quality(self):
        state = TrackState(5)
        state.add(1, {"ocr_normalized": "KA03MN1234", "quality_score": 0.8})
        result = state.add(1, {"ocr_normalized": "KA03MN1234", "quality_score": 0.7})
        self.assertEqual(result["text"], "KA03MN1234")
        self.assertEqual(result["support"], 2)
        state.add(1, {"ocr_normalized": "KA03MN12", "quality_score": 0.1})
        self.assertEqual(state.add(1, {"ocr_normalized": "KA03MN1234", "quality_score": 0.9})["text"], "KA03MN1234")

    def test_track_read_requires_at_least_one_usable_crop(self):
        state = TrackState(5)
        state.add(1, {"ocr_normalized": "KA03MN1234", "quality_score": 0.6, "crop_readable": False, "frame_index": 1})
        result = state.add(1, {"ocr_normalized": "KA03MN1234", "quality_score": 0.7, "crop_readable": False, "frame_index": 2})
        self.assertEqual(result["support"], 2)
        self.assertEqual(result["good_support"], 0)
        result = state.add(1, {"ocr_normalized": "KA03MN1234", "quality_score": 0.8, "crop_readable": True, "frame_index": 3})
        self.assertEqual(result["good_support"], 1)

    def test_track_replaces_same_frame_observation_with_numpy_crop(self):
        state = TrackState(5)
        state.add(1, {"ocr_normalized": "KA03MN1234", "quality_score": 0.9, "frame_index": 1, "crop": np.zeros((4, 4, 3))})
        state.add(1, {"ocr_normalized": "KA03MN1234", "quality_score": 0.4, "frame_index": 2, "crop": np.zeros((4, 4, 3))})
        result = state.add(1, {"ocr_normalized": "KA03MN1234", "quality_score": 0.5, "frame_index": 2, "crop": np.ones((4, 4, 3))})
        self.assertEqual(result["support"], 2)

    def test_coordinate_clipping_and_margin(self):
        self.assertEqual(clip_box([-4, 2, 108, 90], 100, 80), [0, 2, 100, 80])
        self.assertEqual(expand_box([10, 20, 30, 40], 0.25, 100, 100), [5, 15, 35, 45])
        self.assertEqual(clip_bbox_xyxy([-4.5, 2.25, 100.08, 80.12], 100, 80), [0.0, 2.25, 100.0, 80.0])

    def test_bytetrack_assigns_consistent_id(self):
        tracker = ByteTracker(activation_threshold=0.1, lost_track_buffer=30, match_threshold=0.8)
        detection = {"bbox": [10, 10, 100, 100], "score": 0.9, "class_id": 2, "class_name": "car"}
        first = tracker.update([detection])
        second = tracker.update([{**detection, "bbox": [12, 10, 102, 100]}])
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])
        self.assertEqual(tracker.update([]), [])

    def test_result_exports_and_source_redaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotated = root / "annotated.png"
            cv2.imwrite(str(annotated), np.zeros((20, 20, 3), dtype=np.uint8))
            result = write_results("rtsp://alice:secret@camera.local/stream?token=x", [], annotated, root, "video", "rtdetr-v2-r50", {"device": "cpu", "ocr_model": "cct-s-v2-global-model"}, output_stem="safe-stream")
            self.assertTrue(Path(result["json"]).is_file())
            self.assertTrue(Path(result["csv"]).is_file())
            self.assertTrue(Path(result["zip"]).is_file())
            self.assertEqual(Path(result["json"]).name, "safe-stream.json")
            with zipfile.ZipFile(result["zip"]) as archive:
                self.assertIn("model_manifest.yaml", archive.namelist())
            self.assertEqual(safe_source("rtsp://alice:secret@camera.local/stream?token=x"), "rtsp://***@camera.local/stream?[redacted]")
            self.assertEqual(safe_source_stem("rtsp://alice:secret@camera.local/stream?token=x"), "stream")

    def test_manifest_checksum_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
            weight = root / "model.onnx"
            weight.write_bytes(b"model weights")
            checksum = sha256_file(weight)
            (root / "models" / "manifest.yaml").write_text(f"weight_files:\n  - path: model.onnx\n    sha256: {checksum}\n")
            external_dir = root.parent / f"{root.name}-external"
            external_dir.mkdir()
            external = external_dir / "model.onnx"
            external.write_bytes(b"model weights")
            with patch("src.utils.model_manifest.ROOT", root):
                verify_model_artifact(weight)
                verify_model_artifact(external)
                weight.write_bytes(b"modified weights")
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    verify_model_artifact(weight)
            external.unlink()
            external_dir.rmdir()

    def test_evaluation_reports_detection_ocr_association_and_track_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.json"
            predictions.write_text(__import__("json").dumps({
                "run_metadata": {"performance_ms": {"total": {"mean": 100.0}}},
                "observations": [
                    {"source": "frame.jpg", "frame_index": 0, "plate_bbox_xyxy": [0, 0, 10, 10], "ocr_normalized": "KA03MN1234", "recognition_status": "READ", "vehicle_track_id": 2},
                    {"source": "frame.jpg", "frame_index": 0, "plate_bbox_xyxy": [20, 20, 30, 30], "ocr_normalized": "XX", "recognition_status": "READ", "vehicle_track_id": None},
                ],
                "tracks": [{"vehicle_track_id": 2, "consensus": "KA03MN1234"}],
            }))
            ground_truth = root / "ground_truth.csv"
            ground_truth.write_text('source,frame_index,plate_bbox_xyxy,ground_truth,vehicle_track_id,track_ids_are_prediction_ids\nframe.jpg,0,"0,0,10,10",KA03MN1234,2,true\n')
            result = evaluate(predictions, ground_truth)
            self.assertEqual(result["plate_recall"], 1.0)
            self.assertEqual(result["plate_precision"], 0.5)
            self.assertEqual(result["ocr_exact_match"], 1.0)
            self.assertEqual(result["association_accuracy"], 1.0)
            self.assertEqual(result["track_level_exact_match"], 1.0)
            self.assertEqual(result["estimated_pipeline_fps"], 10.0)


if __name__ == "__main__":
    unittest.main()
