# Indian ANPR accuracy and production readiness

**Benchmark dates** 19–20 November 2026

**Release decision** Retain the standard pipeline for operator review. Add Awiros as an optional review aid. The measured video accuracy and available ground truth support a supervised pilot only.

## 1. Frozen baseline and evidence

The baseline was the supplied *Indian ANPR Benchmark Report* and commit `a8ce57dbe0768319f2ed843b65649918fb5ffe87`. Before changing source, the frozen pipeline reproduced 47 of 52 labeled detections and 8 of 25 end-to-end exact registrations on the 47 published DataCluster stills. It emitted 76 proposals. On the detector publisher's separate 882-image test split, fresh inference reproduced 72.03% AP50–95, 98.81% AP50 and 58.27% small-object AP. COCO val2017 vehicle AP was reproduced by rescoring the archived predictions for all 5,000 images, yielding 56.37% all-class AP and 60.94% AP on car, motorcycle, bus and truck. The earlier KITTI car experiment supports retaining the tracker proposal cutoff of 0.35. Its second sequence group gained 11.29 HOTA points and had 54 ID switches versus 116 at 0.15.

The OCR study fixed the identity-grouped split before evaluating candidate transforms. Validation had 24 images with 12 labeled strings. The held-out portion had 23 images with 13 labeled strings. Both portions come from the same 47-image historical sample. The held-out portion was outside this round's tuning, and training overlap with pretrained checkpoints is unknown. FANVID adds a separate publisher test split with nine clips from two Indian sources. After the changes, the default still-image pipeline reproduced the entire frozen aggregate, including box counts, exact strings, character error rates and READ outcomes.

## 2. Models and configuration

| Function | Pretrained component | Role and result |
|---|---|---|
| Vehicles | RF-DETR Large | Default. Separate COCO benchmark documents car, motorcycle, bus and truck. Indian stills lack vehicle boxes. |
| Plates | RT-DETRv2 R50, justjuu | Default. Full-frame and vehicle-crop passes recover complementary plates. |
| Plates | YOLOv9-S 608, Open Image Models | Evaluation only. Weight and training data provenance require review. |
| OCR | Fast-Plate-OCR CCT-S v2 Global | Default. Raw slot and character scores support abstention. |
| OCR | Awiros/anpr-ocr | New opt-in CPU reader. Pinned checkpoint, code and decoder. Apache-2.0 declared by the publisher. |
| OCR | PaddleOCR PP-OCRv5 server detection and English mobile recognition | Existing independent second opinion. Separate CPU process. |
| OCR | Indian TrOCR | Existing research-only candidate. Review the non-commercial terms. |
| Tracking | ByteTrack | Association algorithm, no pretrained checkpoint. One tracker per input video. |

Local smoke tests ran actual inference for every listed checkpoint. The two GTX 1080 Ti cards use CUDA 11.8 and an ONNX Runtime build for Pascal. Awiros and Paddle run on CPU in a separate environment. Exact revisions and hashes are in the model catalog and manifest. No fine-tuning was performed. The sample sizes, dataset terms and uncertain training overlap would make a fine-tuning comparison unreliable.

## 3. Where the still-image pipeline loses accuracy

At IoU 0.5, Standard missed five of 52 annotations. Four misses had no overlapping proposal and one had an overlap below the threshold. The 47 localized boxes averaged 0.854 IoU. There were 29 unmatched proposals and zero unmatched duplicate overlaps at IoU 0.5 after deduplication. The annotations are partial. Annotation-match precision, F1, AP50 and AP50–95 were 61.8%, 73.4%, 82.9% and 58.1%. The ranked AP uses the submitted proposals. These figures measure agreement with the annotated subset. Background false detections and background false READs remain unknown.

Of 25 labeled strings, 23 had matched boxes, 8 were exact and 15 were incorrect. The other two were localization misses. Matched-crop CER was 21.6%. Including misses, CER was 27.7%. Even a raw score of 0.95 or higher yielded 6 exact candidates out of 9. All four official READs with verified labels were correct. With four cases, the error rate is imprecise. The alignment includes two B-to-8 substitutions, two A-to-0 substitutions, two 4-to-2 substitutions, and single 0-to-D and D-to-0 substitutions. The evaluator records all substitutions, insertions and deletions. It preserves the OCR output without format correction.

| Source-pixel or geometry stratum | Localized boxes | Exact text |
|---|---:|---:|
| Plate width under 96 px | 3/4 | 1/4 |
| Plate width 96–159 px | 2/2 | 0/2 |
| Plate width 160 px or larger | 42/46 | 7/19 |
| Dark crop, median gray under 70 | 4/5 | 1/4 |
| Soft crop, Laplacian variance under 40 | 2/2 | 0/1 |
| Compact box, aspect ratio below 2.4 | 26/29 | 6/13 |

Luminance, sharpness and aspect ratio are measured proxies. The published still labels do not establish motorcycle layout, ambient light, IR, glare, occlusion, true angle, font or vehicle matching. Those categories cannot be scored from these stills. The local review gallery shows ground-truth boxes, original crops, reader outputs and confidence. It contains nine cases Awiros corrected and five where CCT was correct and Awiros was wrong.

## 4. Paired OCR experiment and detection ablations

| Same crop protocol | CCT exact | Awiros exact | CCT CER | Awiros CER |
|---|---:|---:|---:|---:|
| Validation, all text labels including misses | 4/12 | 6/12 | 28.2% | 24.8% |
| Held-out, all text labels including misses | 4/13 | 6/13 | 27.2% | 18.4% |
| Held-out, only matched detections | 4/12 | 6/12 | 21.7% | 12.3% |
| Held-out, publisher-box crops with equal padding | 6/13 | 6/13 | 20.2% | 18.4% |

On the held-out split Awiros fixed five CCT errors and lost three CCT successes. The exact paired two-sided probability is 0.727 under equal discordant outcomes. The gain is measured on this sample. It is insufficient to promote Awiros as the default or to assume a better auto-READ rate. The new combined review profile preserves both candidates. Awiros has no weakest-character score and cannot independently meet the automatic READ rule.

On validation predicted crops, original CCT was 4/11 exact. Tight and wide margins, denoising and deskew tied it. Bicubic and unsharp returned 3/11, CLAHE 2/11 and adaptive thresholding 1/11. Awiros was 6/11 exact on the original crop. Several transforms tied and none increased exact match. On publisher-box crops the original Awiros result was 7/12 and CCT was 3/12. Greater padding or a crop transform can help individual cases, but no transform was adopted as a default. Generative restoration was excluded because it can change character strokes. Use independently verified registration text to evaluate any future restoration model.

The frozen detector ablation matched 45 reference boxes using full-frame evidence and 39 using vehicle crops. Their combined result matched 47. Prior Recovery runs with tiling left the total at 47 on this still sample. Lowering the RT-DETR threshold matched 49 and still read 8/25, with 96 proposals versus 76. The YOLO challenger matched 48 with 60 proposals and read 10/25. On the detector publisher's 882 images, the licensed default had the better reported overall localization AP. Different country mix, split provenance and uncertain YOLO weight rights support keeping the challenger out of production.

## 5. Labeled Indian video and real-traffic integration

FANVID supplied nine test clips from Mumbai and Bangalore, with one target plate per clip. Both source videos were recreated from the publisher's frame numbers. Sample every tenth source frame on the 59.94 FPS asset and every fifth frame on the 30 FPS asset, yielding 215 matched-time test opportunities at each resolution. A visual audit found that the publisher's four Bangalore clips interchange the top and right coordinate columns. The reader performs that deterministic repair. Five Mumbai clips provide a sensitivity check with unchanged coordinates. Source videos, case images and registration strings stay local.

| Same FANVID timestamps, CCT default | 1280×720 | 320×180 |
|---|---:|---:|
| Target-plate recall at IoU 0.5 | 116/215 (54.0%) | 110/215 (51.2%) |
| Annotation AP50 / AP50–95 | 39.2% / 14.9% | 37.0% / 12.3% |
| Mean IoU of matches | 0.708 | 0.699 |
| End-to-end exact frame strings | 28/215 (13.0%) | 0/215 |
| End-to-end frame CER | 55.6% | 93.0% |
| Incorrect READs among verified READs | 0/20 | No verified READs |
| Track exact candidates | 4/9 | 0/9 |
| Track CER including misses | 6.7% | 81.1% |
| Matched-plate track-ID switches | 0 | 1 |
| Additional matched IDs per target, total | 0 | 1 |
| Correct stable READ time after first tracked sighting | 2.84 s in one clip | Unavailable |
| Inferred frames per wall second on this host | 2.42 | 2.97 |

Mumbai alone matched 45/143 at 720p and 41/143 at 320×180. Exact frame strings were 8/143 and 0/143. At 720p, the 198 car frames had 100 localized boxes and 25 exact strings. The 17 van frames had 16 localized boxes and 3 exact strings. The clip tracks contain repeated observations. Nine clips and two source videos capture the volume of independent evidence more accurately than the raw frame total. FANVID's original scoring weights localization and character accuracy differently, so these measurements do not reproduce the paper's headline metric. Its target-only labels leave vehicle association precision, background false reads and duplicate physical gate events unmeasured.

| Same 720p video crops, paired OCR | CCT default | Awiros candidate |
|---|---:|---:|
| Exact strings on localized plates | 28/116 (24.1%) | 104/116 (89.7%) |
| CER on the same crops | 17.8% | 1.4% |
| End-to-end candidate exact match including detection misses | 28/215 (13.0%) | 104/215 (48.4%) |
| End-to-end candidate CER including detection misses | 55.6% | 46.8% |

Awiros corrected 77 CCT candidates and lost one CCT success. Seven of nine clips improved in exact-candidate count and two tied. These are offline candidate comparisons using identical detector boxes. The comparison does not evaluate Awiros track accuracy or automatic READs. The 99 misses cap the result. The review profile aggregates Awiros suggestions separately across associated frames where CCT is uncertain.

On 13 held-out DataCluster publisher crops, controlled corruptions yielded 1/13 exact at 42 px, 2/13 with added glare, 3/13 with 11-pixel horizontal blur and 5/13 with perspective distortion. Temporal replay produced 6/13 exact with 23 edits out of 114 characters. Simple majority also produced 6/13 exact with 25 edits. The best-quality frame alone produced 3/13 exact with 43 edits. Replay freezes detection and association and reuses crop content across variants. It supports preserving uncertainty when aggregating, with no claim of independent video accuracy from the replay.

The existing real traffic acceptance clips processed all 402 day frames and 159 night frames. The current build retained 309 day and 106 night vehicle tracks. It emitted nine frame READs by day, none by night and no final track READs for either. Neither clip provides reference text. The wall times were 379.2 s and 99.1 s and the PyTorch process peaked at 780 MiB allocated on each GPU. That excludes ORT memory and other processes. FFprobe decoded the exported media and CSV, JSON, ZIP and observation counts were checked. These figures are deployment demonstrations.

## 6. Engineering changes and acceptance

- Awiros loads pinned artifacts and code by SHA-256 and checks out an exact, clean PaddleOCR source version. Both optional OCR integrations now enforce subprocess read and write deadlines and bounded payloads. The application keeps uncertain OCR candidates visible.
- Track summaries record first and last media timestamps, the track end reason, evidence frames and a stable READ timestamp. Contradictory evidence and inconsistent vehicle classes retract a stable READ. Tracks represent observed visibility, with no physical gate-crossing or cross-camera identity claim.
- An output-directory lock prevents concurrent writers from clobbering each other's outputs. Long videos use a disk-spooled observation stream and monitor available space. Configurable startup CPU fallback exports both requested and effective devices. The failure-path test selected invalid CUDA index 99 and processed the image on CPU.
- OpenCV could open the FANVID AV1 source and return no decodable frames on this host. System FFmpeg with libdav1d now handles recognized local AV1 inputs. A real eight-frame GPU run exported all eight frames. The fallback uses nominal timestamps, including for variable-rate sources.
- Browser verification covered mobile layout, mixed-validity batches, filtering, file download, video preview, vehicle tracks and active cancellation. An external output-directory failure was reproduced and corrected. The application permits downloads from the generated run directory. That directory belongs to the trusted workspace and is shared by users of a deployed instance.
- All 32 tests passed with real GPU integration enabled. A 12-frame FANVID integration check exercised the combined review mode and exported 34 frame-level Awiros suggestions and four separate track-level suggestions. One track had repeated distinct observations. Both Compose variants passed configuration validation. The Docker daemon socket denied access, so image build, container process startup and NVIDIA passthrough still need validation on a deployment host. Structured job completion logging passed on a real GPU video run.

## 7. Gates for a production release

1. **Capture and evaluation.** Qualify each intended camera lane, with separate day, night and IR cases and adequate native plate detail. Collect permission-cleared, exhaustively labeled Indian videos with vehicle boxes, track IDs, plate text and lane events. Keep sites and plate identities disjoint across splits. This enables per-site recall, incorrect READ rate, association accuracy and duplicate event rates.
2. **Recognition quality.** Investigate source-faithful orientation and two-line handling on the labeled validation set. Compare the CCT and Awiros pair, a licensed specialist detector and validated source-preserving rectification. Set the acceptance risk and coverage targets with an explicit minimum number of verified reads and measure on the locked test set.
3. **Service reliability.** Run a long soak test, VRAM contention and disk-pressure tests and actual camera reconnect tests on the deployment host. Add per-operator isolation, idempotent jobs and a durable queue for shared use. Define export retention, upload limits and authenticated transport. Validate the container deployment before relying on it.
4. **Deployment decision.** Review-only operation is supported locally. Automatic decisions should wait for a site-specific false-READ and association benchmark. The current labeled video sample has one correct stable read timestamp across nine HD clips and none at low resolution.

## Sources and reproducibility

- [DataCluster Labs pinned still-image sample](https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset/tree/2bb7cd4e46e58af4f6fa413ce8e205ac69e605a7). The publisher declares CC BY-NC-ND. Raw benchmark media and crop galleries are kept local.
- [Awiros model card](https://huggingface.co/Awiros/anpr-ocr) and [RT-DETRv2 plate checkpoint](https://huggingface.co/justjuu/rtdetr-v2-license-plate-detection).
- [FANVID paper](https://arxiv.org/abs/2506.07304) and [publisher annotation index](https://huggingface.co/datasets/kv1388/FANVID-Face_and_License_Plate_Recognition_in_Low-Resolution_Videos/tree/678a078/data). Source footage is fetched separately.
- [Indian Licence Plate Dataset in the Wild](https://arxiv.org/abs/2111.06054). Published baselines are 83.10% VOC AP50 for FCOS, 83.13% for HRNet and character accuracy of 75.6% versus 66.3% for quadrilateral versus rectangular crops. The [authors' repository](https://github.com/sanchit2843/Indian_LPR) states that the split is unavailable publicly. These numbers use different references and scoring.
- The source checkout includes the frozen protocol, benchmark commands, aggregate JSON and optional GPU test instructions. See `eval/README.md`, `eval/production_stills_results.json` and `eval/production_video_results.json`.
