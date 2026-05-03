from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from table_recon_engine.structure_json import load_structure_records, normalize_class_name, write_structure_records


DEFAULT_THRESHOLDS = {
    "table": 0.40,
    "table row": 0.40,
    "table column": 0.65,
    "table spanning cell": 0.40,
    "table column header": 0.50,
    "table projected row header": 0.40,
}

DEFAULT_AXIS_NMS = {
    "table row": ("y", 0.75),
    "table column": ("x", 0.75),
    "table column header": ("y", 0.75),
    "table projected row header": ("y", 0.75),
}

DEFAULT_TOP_ONE_CLASSES = {"table"}


def confidence(obj: dict[str, Any]) -> float:
    return float(obj.get("confidence", obj.get("score", 1.0)))


def bbox(obj: dict[str, Any]) -> list[float]:
    return [float(value) for value in obj["bbox"]]


def axis_overlap(a: dict[str, Any], b: dict[str, Any], axis: str) -> float:
    ax0, ay0, ax1, ay1 = bbox(a)
    bx0, by0, bx1, by1 = bbox(b)
    if axis == "x":
        intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        denominator = max(1e-6, min(ax1 - ax0, bx1 - bx0))
        return intersection / denominator
    if axis == "y":
        intersection = max(0.0, min(ay1, by1) - max(ay0, by0))
        denominator = max(1e-6, min(ay1 - ay0, by1 - by0))
        return intersection / denominator
    raise ValueError(f"Unsupported axis: {axis}")


def axis_nms(objects: list[dict[str, Any]], class_name: str, axis: str, threshold: float) -> list[dict[str, Any]]:
    target = [obj for obj in objects if normalize_class_name(str(obj.get("class", "")), obj.get("class_id")) == class_name]
    rest = [obj for obj in objects if normalize_class_name(str(obj.get("class", "")), obj.get("class_id")) != class_name]
    kept: list[dict[str, Any]] = []
    for obj in sorted(target, key=confidence, reverse=True):
        if all(axis_overlap(obj, old, axis) < threshold for old in kept):
            kept.append(obj)
    if axis == "x":
        kept.sort(key=lambda item: (bbox(item)[0] + bbox(item)[2]) * 0.5)
    else:
        kept.sort(key=lambda item: (bbox(item)[1] + bbox(item)[3]) * 0.5)
    return rest + kept


def postprocess_record(
    record: dict[str, Any],
    thresholds: dict[str, float],
    axis_nms_rules: dict[str, tuple[str, float]],
    top_one_classes: set[str] | None = None,
) -> dict[str, Any]:
    objects = []
    for obj in record.get("objects", []):
        class_name = normalize_class_name(str(obj.get("class", "")), obj.get("class_id"))
        if confidence(obj) < thresholds.get(class_name, 0.0):
            continue
        clean_obj = dict(obj)
        clean_obj["class"] = class_name
        objects.append(clean_obj)

    for class_name, (axis, threshold) in axis_nms_rules.items():
        objects = axis_nms(objects, class_name=class_name, axis=axis, threshold=threshold)

    for class_name in top_one_classes or set():
        objects = keep_top_one(objects, class_name)

    output = dict(record)
    output["objects"] = objects
    return output


def keep_top_one(objects: list[dict[str, Any]], class_name: str) -> list[dict[str, Any]]:
    target = [
        obj
        for obj in objects
        if normalize_class_name(str(obj.get("class", "")), obj.get("class_id")) == class_name
    ]
    rest = [
        obj
        for obj in objects
        if normalize_class_name(str(obj.get("class", "")), obj.get("class_id")) != class_name
    ]
    if not target:
        return rest
    return rest + [max(target, key=confidence)]


def parse_thresholds(items: list[str] | None) -> dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid threshold override {item!r}, expected class=value")
        name, value = item.split("=", 1)
        thresholds[normalize_class_name(name.strip())] = float(value)
    return thresholds


def parse_axis_nms(items: list[str] | None, disable_defaults: bool) -> dict[str, tuple[str, float]]:
    rules = {} if disable_defaults else dict(DEFAULT_AXIS_NMS)
    for item in items or []:
        parts = item.split("=")
        if len(parts) != 3:
            raise ValueError(f"Invalid NMS rule {item!r}, expected class=axis=threshold")
        name, axis, threshold = parts
        rules[normalize_class_name(name.strip())] = (axis.strip(), float(threshold))
    return rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess predicted structure JSON boxes.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        action="append",
        default=None,
        help="Override a confidence threshold, e.g. --threshold 'table row=0.50'.",
    )
    parser.add_argument(
        "--axis-nms",
        action="append",
        default=None,
        help="Override/add an axis NMS rule, e.g. --axis-nms 'table row=y=0.75'.",
    )
    parser.add_argument("--no-default-axis-nms", action="store_true")
    parser.add_argument("--pretty-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.threshold)
    axis_nms_rules = parse_axis_nms(args.axis_nms, disable_defaults=args.no_default_axis_nms)
    records = [
        postprocess_record(
            record,
            thresholds=thresholds,
            axis_nms_rules=axis_nms_rules,
            top_one_classes=DEFAULT_TOP_ONE_CLASSES,
        )
        for record in load_structure_records(args.input_json)
    ]
    write_jsonl = args.output_json.suffix.lower() == ".jsonl" and not args.pretty_json
    write_structure_records(args.output_json, records, jsonl=write_jsonl)
    manifest = {
        "input_json": str(args.input_json),
        "output_json": str(args.output_json),
        "thresholds": thresholds,
        "axis_nms": axis_nms_rules,
        "top_one_classes": sorted(DEFAULT_TOP_ONE_CLASSES),
        "records": len(records),
        "objects": sum(len(record.get("objects", [])) for record in records),
    }
    manifest_path = args.output_json.with_suffix(args.output_json.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} postprocessed record(s) to {args.output_json}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
