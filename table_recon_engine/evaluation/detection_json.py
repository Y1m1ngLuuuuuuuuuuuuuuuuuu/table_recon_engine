from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from table_recon_engine.structure_json import STRUCTURE_CLASSES, bbox_iou, image_key, load_structure_records, normalize_class_name


def _objects_by_class(record: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in record.get("objects", []):
        class_name = normalize_class_name(str(obj.get("class", "")), obj.get("class_id"))
        if class_name in STRUCTURE_CLASSES:
            grouped[class_name].append(obj)
    return grouped


def _match_class(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[int, int, int, float]:
    target_used = [False] * len(targets)
    tp = 0
    fp = 0
    matched_iou = 0.0
    sorted_predictions = sorted(predictions, key=lambda obj: float(obj.get("confidence", obj.get("score", 1.0))), reverse=True)
    for pred in sorted_predictions:
        best_iou = 0.0
        best_idx = -1
        for idx, target in enumerate(targets):
            if target_used[idx]:
                continue
            iou = bbox_iou(pred["bbox"], target["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            target_used[best_idx] = True
            tp += 1
            matched_iou += best_iou
        else:
            fp += 1
    fn = len(targets) - sum(target_used)
    return tp, fp, fn, matched_iou


def evaluate_records(
    gt_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    gt_by_image = {image_key(record): record for record in gt_records}
    pred_by_image = {image_key(record): record for record in pred_records}
    keys = sorted(gt_by_image.keys())

    per_class = {
        class_name: {"tp": 0, "fp": 0, "fn": 0, "matched_iou_sum": 0.0}
        for class_name in STRUCTURE_CLASSES
    }
    exact_counts = Counter()
    image_records = []

    for key in keys:
        gt_grouped = _objects_by_class(gt_by_image[key])
        pred_grouped = _objects_by_class(pred_by_image.get(key, {"objects": []}))
        image_ok = True
        counts = {}
        for class_name in STRUCTURE_CLASSES:
            gt_count = len(gt_grouped.get(class_name, []))
            pred_count = len(pred_grouped.get(class_name, []))
            if gt_count == pred_count:
                exact_counts[class_name] += 1
            else:
                image_ok = False
            counts[class_name] = {"gt": gt_count, "pred": pred_count, "delta": pred_count - gt_count}

            tp, fp, fn, matched_iou = _match_class(
                pred_grouped.get(class_name, []),
                gt_grouped.get(class_name, []),
                iou_threshold,
            )
            per_class[class_name]["tp"] += tp
            per_class[class_name]["fp"] += fp
            per_class[class_name]["fn"] += fn
            per_class[class_name]["matched_iou_sum"] += matched_iou

        if image_ok:
            exact_counts["all_classes"] += 1
        image_records.append({"image": key, "count_exact": image_ok, "counts": counts})

    class_metrics = {}
    for class_name, totals in per_class.items():
        tp = totals["tp"]
        fp = totals["fp"]
        fn = totals["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        class_metrics[class_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": totals["matched_iou_sum"] / max(tp, 1),
            "count_exact_rate": exact_counts[class_name] / max(len(keys), 1),
        }

    return {
        "iou_threshold": iou_threshold,
        "images": len(keys),
        "missing_predictions": sorted(set(gt_by_image) - set(pred_by_image)),
        "extra_predictions": sorted(set(pred_by_image) - set(gt_by_image)),
        "all_class_count_exact_rate": exact_counts["all_classes"] / max(len(keys), 1),
        "per_class": class_metrics,
        "records": image_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted structure JSON against ground-truth structure JSON.")
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--pred-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_records(
        gt_records=load_structure_records(args.gt_json),
        pred_records=load_structure_records(args.pred_json),
        iou_threshold=args.iou_threshold,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "images={images} all_count_exact={all_class_count_exact_rate:.3f}".format(**summary)
    )
    for class_name in STRUCTURE_CLASSES:
        metric = summary["per_class"][class_name]
        print(
            f"{class_name}: P={metric['precision']:.3f} R={metric['recall']:.3f} "
            f"F1={metric['f1']:.3f} count_exact={metric['count_exact_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
