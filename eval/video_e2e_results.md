# Video end-to-end test results

Historical baseline. See [the current evaluation](hardening_results.md) for hardened video results and export validation.

Run date: 2026-11-19. No training, fine-tuning, or annotation was performed.

## Runtime

- Host GPUs: 2× NVIDIA GeForce GTX 1080 Ti (11 GiB each); the requested RTX 4000 Ada is not present on this machine.
- PyTorch 2.3.1+cu118 / CUDA runtime 11.8; ONNX Runtime GPU 1.20.1 reported `CUDAExecutionProvider` and `CPUExecutionProvider`.
- RF-DETR Large, RT-DETRv2 R50, YOLOv9-S 608, Fast-Plate-OCR CCT-S v2 Global and optional Indian TrOCR artifacts are pinned/recorded in `models/manifest.yaml`; primary runtime models loaded and inferred on CUDA. The manifest records artifact hashes and license caveats.
- `python -m unittest discover -s tests -v`: 11 tests passed. `python -m src.cli doctor`: CUDA probe passed. `pip check`: no broken requirements.
- Gradio callbacks were exercised with two temporary still images and a six-frame video: both workflows returned annotated media and JSON/CSV/ZIP downloads. Empty image/video submissions returned clear validation errors.

## Full default video run

Input: `input/videos/india_traffic_acceptance_2m14s_3fps.mp4`, a 134.002-second, 1920×1080, 3 fps compilation of five distinct Pexels traffic clips. Provenance and transformations are in [`input/videos/README_sources.md`](../input/videos/README_sources.md).

Command: `python -m src.cli video --input input/videos/india_traffic_acceptance_2m14s_3fps.mp4 --output output/video_e2e/rtdetr_full --plate-model rtdetr-v2-r50 --stride 1`

| Check | Result |
| --- | ---: |
| Source frames decoded | 402 |
| Pipeline frames inferred (stride 1) | 402 |
| Annotated MP4 frames decoded | 402 |
| Source / output duration | 134.002 s / 134.000 s |
| Output dimensions / frame rate | 1920×1080 / 3 fps |
| JSON observations / CSV rows | 7,338 / 7,338 |
| Vehicle track IDs | 309 |
| Plate boxes retained, including unassociated cases | 4,493 |
| Statuses | 28 READ, 1,597 LOW_CONFIDENCE, 2,435 UNREADABLE, 433 UNASSOCIATED, 2,845 NO_PLATE_DETECTED |

Vehicle tracks included car (218), motorcycle (38), bus (13), and truck (40). Every source frame index from 0 through 401 had pipeline timing recorded; JSON rows cover every frame in this clip. The decoded annotated video was sampled against the source and all 133 sampled vehicle-box overlay checks found the expected overlay color. The results ZIP contains the MP4, JSON, CSV and model manifest.

Mean / p95 frame-stage times recorded in the run JSON:

| Stage | Mean | p95 |
| --- | ---: | ---: |
| Vehicle detection | 96.6 ms | 111.0 ms |
| Tracking | 12.7 ms | 25.1 ms |
| Plate detection (full frame plus vehicle crops) | 1,029.5 ms | 1,733.4 ms |
| OCR | 36.6 ms | 67.3 ms |
| End-to-end pipeline | 1,191.8 ms | 1,943.0 ms |

The completed 402-frame render took about 8 minutes 20 seconds on this host. Plate detection over multiple vehicle crops is the current bottleneck. Torch-reported peak allocation was 427.7 MiB (ONNX Runtime allocation is separate).

Real-video bounds validation found RF-DETR vehicle boxes occasionally exceeded the frame edge by a fraction of a pixel. `VehicleDetector` now clips boxes before tracking/export, and a regression test was added. A 30-frame real-video rerun reported zero out-of-bounds boxes. The completed MP4 renderer had already clipped the visible rectangles; the full-run JSON/CSV/ZIP exports were corrected to those same image bounds after the run.

## Night-condition run

Input: `input/videos/india_night_traffic_validation_53s_3fps.mp4`, 53.001 seconds, 159 frames. The full 159-frame video was inferred at stride 1 and the annotated output decodes to all 159 frames. The pipeline retained 1,358 JSON/CSV observations across 105 tracks and 589 plate boxes; its conservative statuses included no READ outputs on this short sample. Mean end-to-end latency was 646.8 ms/frame, with 523.5 ms/frame mean plate-detection time.

This confirms night-footage execution and uncertainty surfacing. It does not establish that the system can read night plates reliably.

## Plate-detector challenger comparison

Both detectors processed the identical first 30 frames (10 seconds) of the main compilation, with RF-DETR Large, ByteTrack and CCT-S OCR held constant.

| Plate detector | Plate boxes | Track IDs | Mean pipeline time |
| --- | ---: | ---: | ---: |
| RT-DETRv2 R50 | 80 | 14 | 351.0 ms/frame |
| YOLOv9-S 608 | 54 | 14 | 314.1 ms/frame |

These are candidate output and latency counts only. There are no labeled plate boxes or verified plate strings for these clips, so precision, recall, CER, exact match, association accuracy and false confident-read rate are **unmeasured**. Detector output counts do not identify a winner. YOLOv9 weights remain evaluation-only because their separate weight license is unresolved.

## Interpretation and limitations

- The main and night inputs were converted from 25–60 fps footage to 3 fps before inference. Every frame in each prepared video was processed and appears in the output; the run did not infer every original source frame. The annotated result therefore plays at 3 fps and does not demonstrate 25/30/60 fps throughput.
- `READ` is a model status based on OCR confidence, Indian-format plausibility and, where available, track evidence. It is not a verified registration. The 28 main-video READ statuses have no ground-truth validation.
- Stock footage includes mixed distances and viewpoints and is not an owner-supplied camera acceptance set. Accuracy remains unmeasured.
- No webcam was available and no authorized RTSP endpoint was configured. Live-source stability was not tested in this run.
- The requested RTX 4000 Ada was unavailable; these timings are for dual GTX 1080 Ti hardware.
- Existing still-image detector smoke results and component/UI checks are recorded in [`smoke_results.md`](smoke_results.md).
