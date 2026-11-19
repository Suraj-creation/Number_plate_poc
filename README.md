# India ANPR

Local, pretrained automatic number plate recognition for Indian traffic. Process single images, image batches, recorded video, or a bounded webcam or RTSP capture. Review annotated media, frame observations, independent OCR candidates, vehicle tracks, and downloadable evidence.

The default stack is RF-DETR Large for vehicles, RT-DETRv2 R50 for plates, Fast-Plate-OCR CCT-S v2 Global for text, and ByteTrack for single-camera vehicle IDs. All model inference runs on this workstation. See the [architecture](docs/architecture.md), [engineering audit and evaluation](eval/hardening_results.md), and [model catalog](configs/models.yaml).

## Install on Ubuntu

Use Python 3.10, FFmpeg, and a CUDA-capable NVIDIA card. The tested workstation has two 11 GiB GTX 1080 Ti cards (compute capability 6.1), driver 580.178.04, PyTorch 2.3.1 with CUDA 11.8, and ONNX Runtime 1.20.1 with CUDA 11.8 and cuDNN 8. The driver can report CUDA 13 while the application uses CUDA 11.8. These runtime versions keep Pascal support.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.3.1 torchvision==0.18.1
python -m pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/ onnxruntime-gpu==1.20.1
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
python -m pip check
python -m src.cli doctor
python -m src.cli download --smoke --no-yolo
```

Use `download --smoke` to add the evaluation-only YOLOv9 artifact. Add `--research-ocr` to download Indian TrOCR for non-commercial research. The download command records exact revisions and checksums in [the model manifest](models/manifest.yaml). Inference can run offline after the selected models have been downloaded. Set `ANPR_DEVICE=cuda:1` to select the second GPU, or `ANPR_DEVICE=cpu` for CPU inference. Copy `.env.example` to `.env` for persistent local settings.

## Start the review workspace

```bash
ANPR_OFFLINE=1 python -m src.cli ui
```

Open [the local app](http://127.0.0.1:7860). Choose a media tab, upload input, and process it. Open **Processing settings** for the detector, device, OCR reader, video stride, and crop retention. Standard is the default. Recovery is experimental and adds overlapping tiles and OCR views. It produced no extra matched plates in the public sample and cost more inference time.

Review the annotated preview and select an observation or vehicle track for its evidence. Search or filter the observation preview. The complete run is available in JSON, CSV, and ZIP under **Download results**. Use **Stop** to end at the next processing boundary. Completed frames or images remain downloadable. Newly submitted jobs can use the same interface after a failure or stop.

The UI binds to loopback. Runs are stored under `output/ui_runs/` in separate directories. `ANPR_OUTPUT_DIR` changes the base directory. Delete run folders according to your retention policy. The UI upload limit is 2 GB per file, with a default limit of 100 images per job and a 2,000 observation preview. Model diagnostics report CUDA probes and cache presence. Original crop saving is opt-in.

## CLI

```bash
python -m src.cli image --input input/images/traffic.jpg --output output/still
python -m src.cli images --input input/images --output output/batch
python -m src.cli video --input input/videos/india_traffic_acceptance_2m14s_3fps.mp4 --output output/day
python -m src.cli video --input input/videos/india_night_traffic_validation_53s_3fps.mp4 --stride 2 --max-frames 45 --output output/night
python -m src.cli stream --source 0 --max-duration 30 --output output/capture
python -m src.cli baseline --input input/images/traffic.jpg --output output/baseline
```

Use `--config path.yaml` to apply partial overrides of `configs/default.yaml`. Explicit `--plate-model`, `--profile`, or `--stride` options override the corresponding settings. Press Ctrl+C during CLI video to finalize the processed prefix and return exit code 130. An early decode failure after valid frames returns a partial export and exit code 1. Batches keep successful files and return exit code 1 when any image fails decoding. Use a separate output directory for each CLI run to preserve earlier downloads. The tracker proposal cutoff defaults to 0.35 after a two-group KITTI evaluation. Set `vehicle_tracking_proposal_threshold: 0.15` in an override to reproduce the v0.2.0 tracking settings.

## Optional PaddleOCR second opinion

Install Paddle in a separate environment because its OpenCV distribution and PyYAML pin differ from the primary runtime. The installed model uses PP-OCRv5 server text detection and English mobile recognition on CPU. The primary model continues to use the selected CUDA device. Paddle supplies alternative text, line boxes, and separate scores. It may corroborate a candidate or expose disagreement.

```bash
python3.10 -m venv .venv-paddle
.venv-paddle/bin/python -m pip install -r requirements-paddle.lock
.venv/bin/python -m src.cli download --paddle-ocr --research-ocr
export ANPR_PADDLE_PYTHON="$PWD/.venv-paddle/bin/python"
```

Enable **Use optional PaddleOCR second opinion** in the UI or set `paddle_ocr_fallback: true` in a CLI override. The assets must be present and match the configured hashes. See the [PaddleOCR project documentation](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html).

## Test and evaluate

```bash
python -m unittest discover -s tests -v
ANPR_OFFLINE=1 RUN_ANPR_GPU_TESTS=1 python -m unittest discover -s tests -v
python -m eval.run_eval --predictions output/day/traffic.json --ground-truth eval/acceptance.csv
```

The opt-in GPU test uses actual pretrained models and a locally prepared traffic video. A GitHub checkout includes the preparation notes and links to the source clips. See [video fixture sources](input/videos/README_sources.md). For the optional browser test, install Playwright in a separate test environment, start the local app, and run `python tests/browser_smoke.py`. The [evaluation guide](eval/README.md) documents ground-truth fields, metrics, and the public-sample comparison.

## Published results

The [v0.2.0 release](https://github.com/Suraj-creation/Number_plate_poc/releases/tag/v0.2.0) contains the full day and night result ZIPs. Each contains an annotated MP4, JSON evidence, CSV observations, and the run's model manifest. See the [aggregate regression data](eval/regression_results.json) and [archive checksums](eval/published_artifacts.sha256).

## Troubleshooting

- **CUDA failed or unavailable.** Run `nvidia-smi` and `python -m src.cli doctor`. Match the PyTorch and ONNX CUDA/cuDNN wheels. Select the GPU with available memory. Set `plate_inference_batch_size: 1` on memory-constrained hardware.
- **Model cache error.** Run the download command online. A checksum failure requires restoring the exact released artifact before retrying. For HF downloads, use the configured revision and check free space and HTTPS access.
- **Input did not open.** Check permissions and codec support with `ffprobe`. Remux or convert unsupported files to a supported MP4. The annotated video contains processed images at the reported source FPS, without audio. Odd frame dimensions are padded by up to one pixel per axis.
- **Camera failed.** Check the device index, RTSP transport, and authentication. Enter credentials only in the local connection field. Successful live capture has not been validated on this workstation.
- **Weak reads.** Check the original plate resolution, angle, sharpness, and lighting. Inspect reasons, alternative predictions, and supporting frames before relying on a candidate. Scores and status gates are uncalibrated.

See [known limitations and regression evidence](eval/hardening_results.md) before using this system for decisions that depend on plate accuracy.
