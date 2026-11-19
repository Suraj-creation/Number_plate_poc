"""Reproduce detector and tracker measurements on COCO and KITTI ground truth."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import cv2
import numpy as np

from src.utils.config import configure_caches, load_config

VEHICLES = ("car", "motorcycle", "bus", "truck")
TRACK_PROPOSALS = {"standard": 0.15, "high_scores_only": 0.35}


def images(detector, paths, batch_size):
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start:start + batch_size]:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode benchmark image {path.name}")
            batch.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        predictions = detector.model.predict(batch, threshold=0.001)
        yield from predictions if isinstance(predictions, list) else [predictions]


def model(device):
    configure_caches()
    import torch

    from src.detectors.vehicle import VehicleDetector

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    return VehicleDetector(device=device)


def coco_inference(root, path, device, shard, shards, batch_size):
    data = json.loads((root / "annotations/instances_val2017.json").read_text())
    selected = sorted(data["images"], key=lambda r: r["id"])[shard::shards]
    category_ids = {item["name"]: item["id"] for item in data["categories"]}
    detector = model(device)
    rows = []
    started = perf_counter()
    for item, predictions in zip(selected, images(detector, [root / "val2017" / r["file_name"] for r in selected],
                                                  batch_size), strict=True):
        for box, score, cls in zip(predictions.xyxy, predictions.confidence, predictions.class_id):
            name = detector._name(int(cls))
            if name not in category_ids:
                continue
            x1, y1, x2, y2 = map(float, box)
            rows.append({"image_id": item["id"], "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "score": float(score), "category_id": category_ids[name]})
    path.write_text(json.dumps({"images": [i["id"] for i in selected], "device": device,
                                "wall_s": perf_counter() - started, "predictions": rows}))


def coco_score(root, paths):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    device_measurements = []
    predictions, ids = [], []
    for file in paths:
        result = json.loads(file.read_text())
        predictions.extend(result["predictions"])
        ids.extend(result["images"])
        device_measurements.append({k: result[k] for k in ("device", "wall_s")})
    truth = COCO(str(root / "annotations/instances_val2017.json"))
    if sorted(ids) != sorted(truth.imgs):
        raise ValueError("Expected all 5,000 official COCO validation images exactly once.")
    result = truth.loadRes(predictions)
    scorer = COCOeval(truth, result, "bbox")
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    vehicle_ids = truth.getCatIds(catNms=list(VEHICLES))
    precision = scorer.eval["precision"]
    recall = scorer.eval["recall"]

    def scalar(a):
        return float(a[a > -1].mean()) if (a > -1).any() else None

    metrics = {}
    for name in VEHICLES:
        cid = truth.getCatIds(catNms=[name])[0]
        index = scorer.params.catIds.index(cid)
        totals = Counter()
        sweep = {key: Counter() for key in (0.15, 0.25, 0.35, 0.50)}
        for row in scorer.evalImgs[index * 4 * len(ids):(index * 4 + 1) * len(ids)]:
            if row is None:
                continue
            totals["reference"] += int(np.count_nonzero(~np.asarray(row["gtIgnore"], dtype=bool)))
            for cutoff, result in sweep.items():
                keep = (np.asarray(row["dtScores"]) >= cutoff) & ~row["dtIgnore"][0]
                matched = row["dtMatches"][0] > 0
                result["predictions"] += int(keep.sum())
                result["matched"] += int((keep & matched).sum())
        metrics[name] = {
            "ap": scalar(precision[:, :, index, 0, -1]),
            "ap50": scalar(precision[0, :, index, 0, -1]),
            "ap_small": scalar(precision[:, :, index, 1, -1]),
            "ar100": scalar(recall[:, index, 0, -1]),
            "reference": totals["reference"],
            "score_cutoffs_iou50": {
                str(cutoff): {**row, "precision": row["matched"] / max(row["predictions"], 1),
                              "recall": row["matched"] / max(totals["reference"], 1)}
                for cutoff, row in sweep.items()},
        }
    return {"images": len(ids), "all_class_ap": float(scorer.stats[0]), "all_class_ap50": float(scorer.stats[1]),
            "four_class_ap": float(np.mean([r["ap"] for r in metrics.values()])),
            "category_ids": vehicle_ids, "classes": metrics, "workers": device_measurements}


def kitti_inference(root, output, device, sequences, batch_size):
    from src.detectors.vehicle import clip_bbox_xyxy
    from src.tracking.bytetrack import ByteTracker

    cfg = load_config()
    detector = model(device)
    folders = {profile: output / profile / "data" for profile in TRACK_PROPOSALS}
    for folder in folders.values():
        folder.mkdir(exist_ok=True, parents=True)
    truth_dir = output / "ground_truth"
    (truth_dir / "label_02").mkdir(parents=True, exist_ok=True)
    seqmap = []
    for seq in sequences:
        copyfile(root / "training/label_02" / f"{seq}.txt", truth_dir / "label_02" / f"{seq}.txt")
        files = sorted((root / "training/image_02" / seq).glob("*.png"))
        if not files or [int(f.stem) for f in files] != list(range(len(files))):
            raise ValueError(f"Incomplete KITTI sequence {seq}")
        seqmap.append(f"{seq} empty 000000 {len(files)}")
        trackers = {profile: ByteTracker(cfg["tracker_track_activation_threshold"], cfg["tracker_lost_track_buffer"],
                                         cfg["tracker_match_threshold"]) for profile in folders}
        rows = {profile: [] for profile in folders}
        for file, dets in zip(files, images(detector, files, batch_size), strict=True):
            height, width = cv2.imread(str(file)).shape[:2]
            low = []
            for box, score, cls in zip(dets.xyxy, dets.confidence, dets.class_id):
                name = detector._name(int(cls))
                if name in VEHICLES and score >= min(TRACK_PROPOSALS.values()):
                    low.append({"bbox": clip_bbox_xyxy(box, width, height), "score": float(score),
                                "class_id": int(cls), "class_name": name})
            for profile, tracker in trackers.items():
                candidates = [d for d in low if d["score"] >= TRACK_PROPOSALS[profile]]
                for d in tracker.update(candidates):
                    if d["track_id"] is not None and d["class_name"] == "car":
                        coords = " ".join(f"{v:.5f}" for v in d["bbox"])
                        rows[profile].append(f"{int(file.stem)} {d['track_id']} car -1 -1 -10 {coords} "
                                             f"-1 -1 -1 -1000 -1000 -1000 -10 {d['score']:.6f}")
        for profile, folder in folders.items():
            lines = rows[profile]
            (folder / f"{seq}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    (truth_dir / "evaluate_tracking.seqmap.training").write_text("\n".join(seqmap) + "\n")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("coco-infer", "kitti"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--root", required=True, type=Path)
        cmd.add_argument("--output", required=True, type=Path)
        cmd.add_argument("--device", default="cuda:0")
        cmd.add_argument("--batch-size", type=int, default=4)
        if name == "coco-infer":
            cmd.add_argument("--shard", type=int, default=0)
            cmd.add_argument("--shards", type=int, default=1)
        else:
            cmd.add_argument("--sequences", nargs="+", required=True)
    cmd = commands.add_parser("coco-score")
    cmd.add_argument("--root", required=True, type=Path)
    cmd.add_argument("--predictions", type=Path, nargs="+", required=True)
    cmd.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "coco-infer":
        coco_inference(args.root, args.output, args.device, args.shard, args.shards, args.batch_size)
    elif args.command == "kitti":
        kitti_inference(args.root, args.output, args.device, args.sequences, args.batch_size)
    else:
        args.output.write_text(json.dumps(coco_score(args.root, args.predictions), indent=2) + "\n")


if __name__ == "__main__":
    main()
