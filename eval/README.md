# Evaluation

Run existing annotations against a result export.

```bash
python -m eval.run_eval --predictions output/run.json --ground-truth eval/acceptance.csv
```

CSV columns include `source`, `frame_index` (blank for images), `plate_bbox_xyxy` (quoted `x1,y1,x2,y2`), and optional `ground_truth`, `vehicle_bbox_xyxy`, `vehicle_track_id`, `condition`, and `boxes_exhaustive`. Source must match the result export. A row without a plate or vehicle box can represent an annotated negative frame. Set `boxes_exhaustive=false` when boxes represent only selected plates. Unmatched READs are reported separately and counted as known false reads only on frames declared exhaustive. Supply vehicle boxes to score association against independent geometry. Legacy comparisons may opt into ID equality with `track_ids_are_prediction_ids=true` if the CSV IDs were deliberately defined in the prediction namespace.

The evaluator restricts metrics to listed frames and uses maximum one-to-one IoU matching within each frame. It reports OCR exact match and character error rate on matched detections and across all supplied text labels, including missed boxes. Track IDs are matched by one-to-one observation overlap within each source. Track switch counts use matched plate observations. Per-condition metrics use the optional `condition` field, which should be consistent for each frame in the CSV.

## Public sample

The [DataCluster Labs sample](https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset/tree/2bb7cd4e46e58af4f6fa413ce8e205ac69e605a7) includes Pascal VOC boxes and some nested `number_plate_text` attributes. Its license permits non-commercial evaluation. Keep media and labels in a local scratch directory. To reproduce the comparison, use the 28 image IDs listed in [sample_manifest.csv](sample_manifest.csv) and the corresponding `Annotations/` entries.

```bash
python -m src.cli images --input /path/to/sample/images --output /tmp/standard
python -m src.cli images --input /path/to/sample/images --output /tmp/recovery --profile recovery
python -m eval.public_sample --sample /path/to/sample --predictions /tmp/standard /tmp/recovery --score-readers
```

Add `--research-ocr` only for a non-commercial TrOCR comparison. The current audit includes no new annotations. See [results](hardening_results.md) for the measured scope and limitations.

For a finished video run, validate the exported media and row integrity with `python -m eval.media_integrity output/day/traffic.json`. The check decodes the complete MP4, validates source coordinates and sampled frame indices, and checks CSV counts and ZIP member integrity. Add `--paddle-ocr` to the public reader comparison after installing the optional worker.
