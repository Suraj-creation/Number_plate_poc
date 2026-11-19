# Engineering audit and evaluation

Audit date is 19 November 2026. All inference used existing pretrained models or their pinned publisher artifacts. Public images were evaluated using existing annotations. The labeled sample and plate strings are excluded from this repository.

## Changes and validation

| Previous behavior | Implemented change | Evidence |
| --- | --- | --- |
| ByteTrack filtered out detections without an ID. Association could favor the highest detector score when two vehicles contained a plate. | Preserve qualified trackless detections, feed lower-score proposals to existing tracks, and surface ambiguous associations. | Tracker and geometry regression tests, real videos and complete exports |
| An image could lose a vehicle-without-plate row because all image track IDs were null. | Assign a per-frame vehicle index and track association independently. | Real repeated image batch and frame pipeline tests |
| Temporal voting could label a frame READ while the overlay showed a different string. Final exports ran a second vote. | Frame text controls the frame overlay. Track summaries use the same bounded vote as the pipeline, with separate statuses, unique crop fingerprints and source-frame evidence. | Evidence and track regression tests, end-to-end video |
| CCT used every model output slot to compute an OCR mean. | Retain slot data and apply OCR gates to emitted characters, crop conditions, source plate dimensions and weakest-character evidence. | Real CCT CUDA smoke, annotation comparison and component tests |
| Processing retained every video observation and stored debug crops globally. UI jobs reused shared output names. | Spool observations to disk, bound previews, isolate jobs, and write original crops only when enabled. | Cancelled video with intact archive, corrupt batch and collision regression, Chromium flow |
| Cold model download without smoke omitted some weights. | Separate asset acquisition from inference, verify release checksums, pin HF inference snapshots and preserve acquisition metadata. | Offline acquisition and CUDA smoke of selected artifacts |

## Publisher-labeled sample

DataCluster Labs, [revision 2bb7cd4](https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset/tree/2bb7cd4e46e58af4f6fa413ce8e205ac69e605a7). The same 28 sample images include 33 box labels and 25 existing text labels. The earlier 10-image report checked the wrong XML level for nested strings. Two of its images contain nested text labels. This run reads the nested attributes and scores strings against the publisher labels. No independent relabeling was performed.

One-to-one box matching uses IoU ≥ 0.50. Box precision here means overlap with supplied annotations. An unmatched detection may depict an additional real plate because coverage is not known to be exhaustive. Evaluated READs are matched predictions with a publisher text label. OCR exact match is also reported across all 25 text labels in the machine-readable report, which treats missed boxes as misses.

| Profile | Matched boxes / 33 | All plate boxes | Exact strings / matched text labels | Incorrect / evaluated READs |
| --- | ---: | ---: | ---: | ---: |
| Original RT-DETRv2 | 31 | 55 | 8 / 23 | 1 / 5 |
| Hardened RT-DETRv2 Standard | 31 | 55 | 8 / 23 | 0 / 4 |
| Hardened RT-DETRv2 Recovery | 31 | 83 | 7 / 23 | 0 / 1 |
| Original YOLOv9-S 608 | 30 | 40 | 10 / 22 | 1 / 7 |
| Hardened YOLOv9-S 608 | 30 | 40 | 10 / 22 | 1 / 6 |
| Hardened RT-DETRv2 with optional Paddle | 31 | 55 | 8 / 23 | 0 / 4 |

RT-DETR Standard recovered 28 of the labeled plates in the full-frame pass, 27 in vehicle crops, and 31 in the combined output. YOLOv9 matched 29 in the full-frame pass, 26 in vehicle crops, and 30 combined. Recovery produced additional boxes without a labeled recall gain and remains an experimental option. YOLOv9 remains evaluation-only because its exact weight terms are unresolved. No alternative vehicle checkpoint was adopted because the available labels do not measure vehicle-recall improvement.

The independent reader comparison used publisher-box crops with the same 8% source margin on 25 plates. CCT matched 9, Indian TrOCR matched 7, and PaddleOCR matched 13. PaddleOCR emitted an incorrect string for 9 of 21 crops above its own 0.85 score gate. The opt-in Paddle end-to-end pass retained the Standard primary exact count while supplying alternative evidence. Regional TrOCR weights remain non-commercial research only. The CCT model has ten output slots, which limits longer registrations. Its published region list does not explicitly include India.

Contrast and row flattening on annotated crops yielded no additional exact matches in the tested subset. Cubic upscaling tied the original at 9 / 25. The tested unmodified 8% crop margin scored above the tested zero, 4%, 14%, and 20% margins. These probes used the same sample as the model comparison and do not establish general operating thresholds. No generative image enhancement or plate string correction was applied.

## Runtime and regression

Both GTX 1080 Ti GPUs passed a PyTorch convolution probe. RT-DETRv2, YOLOv9-S 608, CCT-S v2 and Indian TrOCR loaded and ran on CUDA. FastALPR also ran in CPU mode. Paddle's two pinned assets ran locally on CPU in its isolated environment. The dependency lock passed `pip check`.

On the same 30 day-clip frames, RT-DETR batch size four matched scalar processing on every exported plate box, vehicle box, track ID, OCR candidate and recognition status across 230 observations. Mean plate-stage time fell from 716 to 527 ms. Peak Torch allocation rose from 428 to 780 MiB. These are Torch allocator counters and omit ONNX CUDA allocation. A separate process-level NVIDIA sample observed 1,744 MiB for the batched run. Other jobs were active on the host, so timings are directional.

The regression suite includes deterministic tests and an opt-in real CUDA test for stop behavior, stride, annotated video, image batch, optional crops, archive integrity, config, confidence, geometry, and evaluation. Chromium covered empty validation, partial batch recovery, filtering, download, video review and active cancellation. Screenshots and evidence are in the [UI audit](../docs/ui/audit.md).

## Remaining limitations

- The day and night video ground truth is unavailable. Vehicle recall, association accuracy, exact track matches and ID switches on these clips are unmeasured. The public still labels support the stated plate and OCR comparisons only.
- There is no dedicated IR, rain, calibrated motion blur, or target-camera acceptance set. The two-line experiment had very few qualifying crops. Denoising, perspective rectification, and rotations remain unadopted without a demonstrated benefit.
- CCT, Paddle and TrOCR can read the same plate incorrectly. A syntax match and repeated observations do not guarantee accuracy. The small READ denominators provide limited evidence about false confident reads in operation.
- ByteTrack is for a single camera. Crowding, hard scene cuts and long occlusions can cause track switches. An unassociated plate stays in the results and does not vote on a vehicle.
- There is no live camera or authorized RTSP feed on this host. The video comparison uses prepared 1920×1080 footage at 3 FPS. It does not establish native 25–60 FPS throughput or performance on RTX 4000 Ada.
- Production distribution requires a review of the exact pretrained weights and training-data provenance. Full model and package details are in the [model manifest](../models/manifest.yaml).

## Dependency review

`pip-audit` reported advisories for the pinned Filelock, Pillow, and Transformers versions. The reported Filelock and Pillow fixes require Python 3.11+, and the current runtime is Python 3.10. CUDA vendor wheels were unavailable to the advisory resolver, so this audit gives no coverage for those wheels. The build remains local-only. Runtime model acquisition verifies recorded file hashes and Hugging Face inference uses pinned, local safetensors snapshots. The dependency findings remain open for a compatible Python and CUDA runtime upgrade. [Filelock release](https://pypi.org/project/filelock/4.2.1/), [Pillow release](https://pypi.org/project/pillow/13.0.0/), [Transformers security guidance](https://github.com/huggingface/transformers/blob/main/SECURITY.md).

## Full day and night integration

The final Standard profile processed each frame of the prepared videos. The integrity check decoded the written MP4s with FFmpeg, checked observation coordinates and source indices, compared CSV and JSON counts, and verified ZIP CRCs.

| Input | Decoded, inferred and encoded frames | Observations | Tracks | Frame READs | Frame pipeline mean / p95 | Decode-to-MP4 wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Day compilation](https://github.com/Suraj-creation/Number_plate_poc/releases/download/v0.2.0/india_traffic_acceptance_2m14s_3fps_results.zip) | 402 | 12,054 | 311 | 8 | 1,034 / 1,573 ms | 427 s |
| [Night compilation](https://github.com/Suraj-creation/Number_plate_poc/releases/download/v0.2.0/india_night_traffic_validation_53s_3fps_results.zip) | 159 | 2,685 | 105 | 0 | 1,027 / 1,791 ms | 168 s |

Both exports are 1920×1080 at 3 FPS. In the full day run, capping inference and decoder threads preserved the published boxes, track IDs, candidates and recognition statuses across all 12,054 observations. Decode-to-MP4 wall time fell from 632 to 427 seconds across runs on the two GTX cards. That comparison reflects different workstation load. Neither video supplied registration labels or produced a final track READ. Earlier software reported 28 day frame READs. The changed counts reflect evidence rules and retained detections. They cannot establish that more or fewer plates were correctly read. The browser cancellation retained two decoded frames and produced a valid archive. The unavailable webcam case returned a clear open error. All 24 tests passed with the real CUDA integration enabled.

Among the four publisher plates under 96 px wide, Standard detected three and read one exactly across all four labels. In the 25-crop OCR comparison, CCT made 52 character edits, Paddle made 54, and TrOCR made 114 across 231 reference characters. Paddle's advantage was on whole-string matches, while CCT had slightly fewer total edits. See [regression_results.json](regression_results.json) for denominators, timings, source-width breakdowns, and advisory IDs.
