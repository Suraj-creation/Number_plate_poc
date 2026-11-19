# Model cache and provenance

The [catalog](../configs/models.yaml) pins source URLs, SHA-256 checksums, known license terms and evaluation restrictions. The [manifest](manifest.yaml) records the files present, exact revisions, installed model package versions, and smoke checks. Runtime uses local artifacts with `ANPR_OFFLINE=1`.

`python -m src.cli download --no-yolo --smoke` acquires and exercises the default stack. Omit `--no-yolo` for YOLOv9 internal evaluation. Add `--research-ocr` for Indian TrOCR non-commercial research. Add `--paddle-ocr` for the pinned PP-OCRv5 detector and English mobile reader, then install the isolated CPU environment from `requirements-paddle.lock`.

The Open Image Models and Fast-Plate-OCR code licenses do not establish the provenance of their weights or data. The RT-DETRv2 and Paddle model cards declare Apache-2.0. The Indian TrOCR detector artifact is outside this stack. See the [evaluation](../eval/hardening_results.md) for the tested model decisions.
