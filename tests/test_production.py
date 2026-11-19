import tempfile
import unittest
import shutil
import subprocess
from pathlib import Path

from eval.diagnostics import diagnose, edit_alignment
from eval.metrics import evaluate_rows
from src.ingest.ffmpeg_video import Av1Capture
from src.output.run_guard import OutputSpaceError, output_run
from src.recognition.evidence import recognition_evidence
from src.tracking.track_state import TrackState
from src.utils.config import load_config, validate_config
from src.utils.device import CudaUnavailable, require_cuda


class ProductionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "optional system FFmpeg")
    def test_system_av1_reads_a_bounded_exact_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.mp4"
            encoders = subprocess.check_output(["ffmpeg", "-v", "error", "-encoders"], text=True, timeout=10)
            if "libaom-av1" not in encoders:
                self.skipTest("FFmpeg does not provide a test AV1 encoder")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=64x48:r=10",
                            "-frames:v", "3", "-c:v", "libaom-av1", "-cpu-used", "8", "-threads", "1", str(path)],
                           check=True, timeout=10)
            capture = Av1Capture(path, 1, 1024**2)
            try:
                self.assertEqual(capture.fps, 10)
                for _ in range(3):
                    ok, frame = capture.read()
                    self.assertTrue(ok)
                    self.assertEqual(frame.shape, (48, 64, 3))
                self.assertEqual(capture.read(), (False, None))
                self.assertFalse(capture.failed)
            finally:
                capture.release()

    def test_run_directory_lock_is_released_and_space_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with output_run(path, 0):
                with self.assertRaisesRegex(OSError, "Another ANPR job"), output_run(path, 0):
                    self.fail("Second run acquired the same lock")
            with self.assertRaises(OutputSpaceError), output_run(path, 10**15):
                self.fail("Impossible free-space requirement accepted")
            with output_run(path, 0):
                self.assertTrue(path.is_dir())

    def test_temporal_read_timestamp_resets_after_contradiction(self):
        state = TrackState(max_observations=3)
        vehicle = {"track_id": 7, "class_name": "car"}

        def add(frame, text, quality=1):
            timestamp = frame * 300
            state.seen([vehicle], frame, 2, timestamp)
            return state.add(7, {"ocr_normalized": text, "quality_score": quality,
                                 "fingerprint": str(frame), "crop_readable": True,
                                 "frame_index": frame, "timestamp_ms": timestamp,
                                 "supports_track_read": True})

        add(0, "MH12AA5555", .6)
        self.assertEqual(add(1, "MH12AA5555", .61)["status"], "READ")
        self.assertEqual(add(2, "MH12AA5556")["status"], "LOW_CONFIDENCE")
        self.assertEqual(add(3, "MH12AA5556", .95)["status"], "READ")
        track, = state.seen([], 9, 2, 2700)
        self.assertEqual(track["stable_read_ms"], 900)
        self.assertEqual(track["first_seen_ms"], 0)
        self.assertEqual(track["last_seen_ms"], 900)
        self.assertEqual(track["track_end_reason"], "TRACK_LOST")

    def test_review_model_never_confirms_or_refutes_a_frame_read(self):
        recognized = {"raw": "MH12AA5555", "raw_score": .94, "minimum_char_score": .9}
        quality = {"too_small": False, "blurry": False, "contrast": 150}
        alternative = {"independent": True, "normalized": "MH12AA5556",
                       "raw_score": .99, "confidence_role": "review_only"}
        args = (quality, {"score": .95}, {"state": "ASSOCIATED"}, False, [alternative], load_config())
        self.assertEqual(recognition_evidence(recognized, *args)[2], "READ")
        no_char_scores = dict(recognized, minimum_char_score=None)
        self.assertEqual(recognition_evidence(no_char_scores, *args)[2], "LOW_CONFIDENCE")
        with self.assertRaises(ValueError):
            validate_config(dict(load_config(), ocr_engine="awiros", awiros_review=True))

    def test_track_keeps_a_review_consensus_separate_from_primary_evidence(self):
        state = TrackState()
        for i in range(3):
            timestamp = i * 500
            state.seen([{"track_id": 1, "class_name": "car"}], i, 20, timestamp)
            row = {"quality_score": .8, "supports_track_read": False, "fingerprint": str(i % 2),
                   "frame_index": i, "timestamp_ms": timestamp}
            state.add(1, {**row, "ocr_normalized": "MH12AA5555"})
            state.add_review(1, {**row, "ocr_normalized": "MH12AA5556"})
        track = state.finish(1)
        self.assertEqual(track["consensus"], "MH12AA5555")
        self.assertEqual(track["review_ocr_consensus"], "MH12AA5556")
        self.assertEqual(track["review_ocr_frames"], [0, 1])
        self.assertEqual(track["recognition_status"], "LOW_CONFIDENCE")

    def test_bad_cuda_index_fails_before_model_loading(self):
        with self.assertRaises(CudaUnavailable):
            require_cuda("cuda:999")

    def test_error_analysis_attributes_duplicate_overlap_and_partial_miss(self):
        gt = [{"source": "test", "plate_bbox_xyxy": "0,0,10,10", "ground_truth": "AB01",
               "boxes_exhaustive": "false"},
              {"source": "test", "plate_bbox_xyxy": "20,20,30,30"}]
        boxes = [[0, 0, 10, 10], [0, 0, 10, 9], [29, 29, 35, 35]]
        result, _ = diagnose({"observations": [
            {"source": "test", "plate_bbox_xyxy": b, "plate_detection_confidence": 1-index/10,
             "ocr_normalized": "ABO1", "ocr_raw_score": .99, "recognition_status": "READ"}
            for index, b in enumerate(boxes)]}, gt)
        self.assertEqual(result["ocr"]["counts"]["duplicate_overlap"], 1)
        self.assertEqual(result["ocr"]["counts"]["overlapping_miss"], 1)
        self.assertEqual(result["ocr"]["incorrect_read_rate"], 1)
        self.assertIsNone(result["ocr"]["background_false_reads"])
        self.assertIn(("0", "O"), edit_alignment("AB01", "ABO1"))

    def test_track_cer_and_latency_include_a_missing_track(self):
        gt = [{"source": "test", "frame_index": i, "plate_bbox_xyxy": "0,0,10,10",
               "ground_truth": "AB01", "vehicle_track_id": f"gt{i}"} for i in (0, 1)]
        observation = {"source": "test", "frame_index": 0, "plate_bbox_xyxy": [0, 0, 10, 10],
                       "ocr_normalized": "AB01", "vehicle_track_id": 3}
        result = evaluate_rows({"observations": [observation], "tracks": [
            {"vehicle_track_id": 3, "consensus": "AB01", "recognition_status": "READ",
             "first_seen_ms": 100, "stable_read_ms": 420}]}, gt)
        self.assertEqual(result["track_level_character_error_rate"], .5)
        self.assertEqual(result["correct_stable_read_latency_ms"]["median"], 320)


if __name__ == "__main__":
    unittest.main()
