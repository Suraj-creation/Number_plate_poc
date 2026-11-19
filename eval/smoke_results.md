# Implementation and smoke-test record

Historical record. The later audit found nested publisher text labels in part of the sample. See [the current evaluation](hardening_results.md).

Date: 2026-11-19. These checks verify runtime and a small exploratory plate-box comparison; they do not establish production accuracy.

## Host and runtime

- Host GPU: 2× NVIDIA GeForce GTX 1080 Ti, 11 GiB each, compute capability 6.1; NVIDIA driver 580.178.04. The host reports CUDA 13.0, while the tested application uses PyTorch 2.3.1+cu118 and ONNX Runtime GPU 1.20.1.
- The requested RTX 4000 Ada is not present. Performance and memory results below are from this GTX 1080 Ti host and must not be treated as Ada measurements.
- `python -m src.cli doctor`: both PyTorch CUDA availability and a real CUDA convolution probe passed. ONNX Runtime reports CUDA and TensorRT providers; direct OCR and YOLO sessions activated `CUDAExecutionProvider`.
- The project venv passed `pip check`; the tested default dependency snapshot is in `requirements.lock`.

## Model and component execution

| Component | Verification |
| --- | --- |
| RF-DETR Large | Loaded the recorded pretrained checkpoint and ran CUDA inference on blank, bus, Indian car, motorcycle, truck, and traffic-scene images. COCO vehicle mapping emitted car, motorcycle, bus, and truck in the exercised images. |
| RT-DETRv2 R50 | Pinned HF revision `0ca7598f58649fe4a27db22ad72dca7be70314b1`; loaded and ran PyTorch CUDA inference. Returned boxes mapped to source pixels and were matched against sample XML boxes. |
| YOLOv9-S 608 | Loaded the recorded ONNX artifact; CUDA provider was active. Ran full image pipeline inference and offline inference. Weight terms remain unresolved, so use remains internal evaluation only. |
| Fast-Plate-OCR CCT-S v2 Global | Loaded the recorded ONNX model; CUDA provider was active. Ran OCR on actual detected crops and offline inference. Blank input returned no text as expected. |
| Indian TrOCR challenger | Pinned revision `82232c6220919c8c60d148ad74eb13dbea37e260`; loaded from the project cache and ran CUDA inference on a real sample plate crop in offline mode. Its license is non-commercial research/education only. The AGPL `yolo_best.pt` artifact was not downloaded. |
| ByteTrack | Assigned stable IDs across two repeated frames of an Indian traffic-scene clip, including 12 vehicles across bus, car, and motorcycle detections. The test clip repeats a still image and does not measure motion or occlusion handling. |
| Format, association, voting, coordinates, checksums | Unit tests cover standard/BH/legacy Indian patterns, geometric association, whole-string voting, unusable-crop gating, same-frame crop replacement, coordinate clipping, model SHA verification, and RTSP source redaction. |
| PaddleOCR PP-OCRv5 | Optional fallback is disabled and was not installed in the default environment; no fallback accuracy claim is made. |

The default model smoke command and the optional TrOCR smoke command completed with real model inference. The manifest records exact artifact checksums, the RT-DETRv2/TrOCR revisions, package versions, and license notes.

## Plate-detector comparison

Two candidates ran through the same end-to-end image pipeline on 10 images selected from the public Indian Number Plates Dataset sample, with one plate bounding-box annotation per image. We used one-to-one greedy matching at IoU ≥ 0.50:

| Candidate | Ground-truth boxes | Pipeline plate detections | Matched | Recall | Precision | Mean plate-stage latency | Mean total frame latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RT-DETRv2 R50 | 10 | 20 | 10 | 1.00 | 0.50 | 367 ms | 614 ms |
| YOLOv9-S 608 | 10 | 15 | 10 | 1.00 | 0.67 | 407 ms | 661 ms |

These are exploratory results from a small convenience sample and a single pass per candidate, not a statistically reliable ranking. The YOLO candidate had fewer unmatched detections in this sample, but its checkpoint license is unresolved; RT-DETRv2 remains the default. The sample annotations provided plate boxes but did not include the OCR text attribute in the downloaded XML files. OCR exact-match, character-error rate, false confident-read rate, and legal registration correctness therefore remain unmeasured. Vehicle recall and association accuracy also remain unmeasured because the sample has no vehicle-box ground truth.

The sample included close car plates, motorcycle/two-line plates, truck plates, and a distant traffic scene. It did not cover a representative night/IR set, rain, long occlusion, or real traffic-video motion. The sample is published for non-commercial evaluation under CC BY-NC-ND 4.0; downloaded samples and derived previews were used only for local evaluation, removed afterward, and are not included in this project. See the [dataset card](https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset).

## Image, video, UI, and export checks

- A generic bus image completed all pipeline stages. False plate-like detections on bus artwork produced `LOW_CONFIDENCE` or `UNREADABLE`; none was marked as a plausible Indian `READ`.
- FastALPR's standalone still-image wrapper ran with CUDA providers and wrote an annotated image and JSON; it returned no plates on that bus image.
- The 10-image Indian batch completed with both detector candidates. Full-frame and vehicle-crop results were merged; detections remained in original-frame coordinates.
- Two- and four-frame MP4s made from repeated Indian traffic stills completed end to end. Vehicle IDs persisted across repeated frames; with stride 2, JSON retained source frame indexes 0 and 2. The system wrote annotated MP4, JSON observations, CSV rows, per-track summary, and ZIP bundle.
- Gradio built successfully; its local server returned HTTP 200 on `127.0.0.1`. The image-batch callback returned two annotated previews, a table, JSON/CSV, and a ZIP; the video callback returned an annotated video and downloads. Empty image input returned a readable error.
- Offline inference passed for the default RT-DETRv2 path, YOLOv9 path, CCT-S OCR, and cached Indian TrOCR.
- On the two-frame traffic-still video, average per-processed-frame timings were vehicle 296 ms, plate detection 884 ms, OCR 123 ms, total 1,326 ms. PyTorch peak allocated/reserved memory was 558/702 MiB. Those counters exclude ONNX Runtime's CUDA allocator and are not total-process VRAM. They include only two repeated frames and are not steady-state benchmarks.
- No webcam device or RTSP endpoint was available. An unavailable camera index returned a clear open-source error; live-stream success and a long-duration stability run remain unverified.

## Remaining acceptance work

Provide authorized representative Indian images and real video from the target cameras, including night/IR and difficult angles, along with manually checked text/vehicle ground truth. Then rerun the evaluation for both plate detectors and OCR profiles. Until that sample is checked, treat `READ` as a model-supported candidate and do not describe system outputs as verified registrations.
