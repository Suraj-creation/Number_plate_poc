from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--iou", default=0.5, type=float)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.predictions, args.ground_truth, args.iou), indent=2))


if __name__ == "__main__":
    main()
