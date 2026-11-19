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

## Full COCO and KITTI reruns

The consolidated DOCX is in [docs](../docs/Indian_ANPR_Benchmark_Report.docx). The [aggregate benchmark JSON](model_benchmark_results.json) records the checkpoints, cohort sizes, exact counts, and settings. Input media, per-plate strings, and COCO prediction files stay in a local scratch directory.

For COCO, unpack the official val2017 images and instances annotations into a directory containing `val2017/` and `annotations/instances_val2017.json`. Run inference on the full split and score with pycocotools. An optional shard pair can split the 5,000 images across two GPUs.

```bash
ANPR_OFFLINE=1 python -m eval.vehicle_tracking_benchmark coco-infer --root /data/coco --output /tmp/coco_predictions.json
python -m eval.vehicle_tracking_benchmark coco-score --root /data/coco --predictions /tmp/coco_predictions.json --output /tmp/coco_metrics.json
```

For KITTI, keep the official `training/image_02/<sequence>/` and `training/label_02/<sequence>.txt` layout. Run groups `0000 0003 0017` and `0002 0005 0010` separately. The benchmark script stages the existing labels and a sequence map under the output directory’s ground_truth folder. It writes tracker predictions for proposal cutoffs 0.15 and 0.35. Use TrackEval's KITTI 2D script for the car class, with that folder as GT_FOLDER and tracker names `standard` and `high_scores_only`. The pinned TrackEval revision in the JSON still uses legacy NumPy aliases. On NumPy 1.26, map its removed `np.float`, `np.int`, and `np.bool` aliases to the corresponding built-ins before running the evaluation.

```bash
ANPR_OFFLINE=1 python -m eval.vehicle_tracking_benchmark kitti --root /data/kitti --output /tmp/kitti-tracks --sequences 0000 0003 0017
```

To compare OCR readers on the identical crops, prepare the private crop directory from the complete DataCluster sample and Standard RT-DETR JSON outputs with `python -m eval.ocr_challenger_benchmark prepare --dataset /data/datacluster --runs /tmp/standard --crops /tmp/ocr-study`. Run the same module with actions `cct`, `paddle`, or `trocr` to collect predictions. Awiros uses the publisher's inference script on each crop folder with the exact model and PaddleOCR source revisions in the aggregate JSON. Save its prediction JSON as `awiros_<crop-folder-name>.json` under the crop directory. Run the module's `score` action against that directory for aggregate results.

The RT-DETRv2 plate checkpoint's publisher test split is available as a Parquet file at revision `b76dbba86154c33fa370bc3087fbc7c766845a66` of `justjuu/license-plate-detection`. With pyarrow 20.0 installed in the benchmark environment, run `python -m eval.plate_detector_benchmark --parquet /data/justjuu-test.parquet --output /tmp/plate_metrics.json`. Use `--model yolo-v9-s-608` for the evaluation-only challenger. This check uses full-frame inference and COCO box AP, with no tracker or OCR.
