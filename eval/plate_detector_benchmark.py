"""Reproduce the published RT-DETR plate checkpoint test split with COCO metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.utils.config import configure_caches


def benchmark(parquet, model_name, device, batch_size):
    import pyarrow.parquet as pq
    import torch

    from src.detectors.plate import PlateDetector

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    detector = PlateDetector(model_name, .001, device)
    categories = [{"id": 1, "name": "license_plate"}]
    images, annotations, predictions = [], [], []
    started = perf_counter()
    for group in pq.ParquetFile(parquet).iter_batches(batch_size=batch_size):
        batch = []
        for row in group.to_pylist():
            frame = cv2.imdecode(np.frombuffer(row["image"]["bytes"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Cannot decode a publisher test image.")
            height, width = frame.shape[:2]
            images.append({"id": len(images) + 1, "height": height, "width": width})
            batch.append(frame)
            for box, category in zip(row["objects"]["bbox"], row["objects"]["category"], strict=True):
                if category != 0 or box[2] <= 0 or box[3] <= 0:
                    raise ValueError("Invalid annotation in publisher test split.")
                annotations.append({"id": len(annotations) + 1, "image_id": images[-1]["id"], "category_id": 1,
                                    "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
        for frame_info, detections in zip(images[-len(batch):], detector.predict_many(batch), strict=True):
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                if x2 > x1 and y2 > y1:
                    predictions.append({"image_id": frame_info["id"], "category_id": 1,
                                        "bbox": [x1, y1, x2-x1, y2-y1], "score": detection["score"]})
    if len(images) != 882:
        raise ValueError("Publisher test split should contain 882 images.")
    infer_s = perf_counter() - started
    truth = COCO()
    truth.dataset = {"info": {}, "images": images, "annotations": annotations, "categories": categories}
    truth.createIndex()
    evaler = COCOeval(truth, truth.loadRes(predictions), "bbox")
    evaler.evaluate()
    evaler.accumulate()
    evaler.summarize()
    matched, count = 0, 0
    for row in evaler.evalImgs[:len(images)]:
        if row is not None:
            keep = (np.asarray(row["dtScores"]) >= .25) & ~row["dtIgnore"][0]
            count += int(keep.sum())
            matched += int((keep & (row["dtMatches"][0] > 0)).sum())
    return {"model": model_name, "device": device, "test_images": len(images), "ground_truth_boxes": len(annotations),
            "ap": float(evaler.stats[0]), "ap50": float(evaler.stats[1]), "ap75": float(evaler.stats[2]),
            "ap_small": float(evaler.stats[3]), "score_floor": .001, "wall_ms_per_image": infer_s * 1000 / len(images),
            "at_0_25": {"matched": matched, "predictions": count, "precision": matched/max(count, 1),
                        "recall": matched/len(annotations)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--model", default="rtdetr-v2-r50", choices=("rtdetr-v2-r50", "yolo-v9-s-608"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configure_caches()
    args.output.write_text(json.dumps(benchmark(args.parquet, args.model, args.device, args.batch_size), indent=2) + "\n")


if __name__ == "__main__":
    main()
