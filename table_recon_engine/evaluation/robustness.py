from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance

from table_recon_engine.evaluation.detection_json import evaluate_records
from table_recon_engine.postprocess_structure_json import (
    DEFAULT_AXIS_NMS,
    DEFAULT_THRESHOLDS,
    DEFAULT_TOP_ONE_CLASSES,
    postprocess_record,
)
from table_recon_engine.structure_json import (
    STRUCTURE_CLASSES,
    image_key,
    load_structure_records,
    normalize_class_name,
    write_structure_records,
)


CONDITIONS = ["clean", "gaussian_noise", "gaussian_blur", "brightness_contrast", "jpeg_compression"]


def select_device(requested: str | None) -> str | int:
    if requested:
        return requested
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_image(image_root: Path, image_name: str) -> Path | None:
    candidate = image_root / Path(image_name).name
    if candidate.exists():
        return candidate
    stem = Path(image_name).stem
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        match = image_root / f"{stem}{suffix}"
        if match.exists():
            return match
    return None


def select_records(records: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:samples]


def build_gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if radius is None:
        radius = int(np.ceil(3.0 * sigma))

    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            dy = y - radius
            dx = x - radius
            kernel[y, x] = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    kernel /= float(kernel.sum())
    return kernel


def hand_written_gaussian_blur(image: Image.Image, sigma: float = 1.35) -> Image.Image:
    kernel = build_gaussian_kernel(sigma=sigma)
    radius = kernel.shape[0] // 2
    src = np.asarray(image.convert("RGB"), dtype=np.float32)
    padded = np.pad(src, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    dst = np.zeros_like(src, dtype=np.float32)

    height, width = src.shape[:2]
    for ky in range(kernel.shape[0]):
        for kx in range(kernel.shape[1]):
            weight = kernel[ky, kx]
            patch = padded[ky : ky + height, kx : kx + width]
            dst += patch * weight

    dst = np.clip(dst, 0, 255).astype(np.uint8)
    return Image.fromarray(dst, mode="RGB")


def apply_condition(image: Image.Image, condition: str, rng: np.random.Generator) -> Image.Image:
    image = image.convert("RGB")
    if condition == "clean":
        return image
    if condition == "gaussian_noise":
        arr = np.asarray(image).astype(np.float32)
        noise = rng.normal(loc=0.0, scale=18.0, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    if condition == "gaussian_blur":
        return hand_written_gaussian_blur(image, sigma=1.35)
    if condition == "brightness_contrast":
        image = ImageEnhance.Brightness(image).enhance(0.72)
        return ImageEnhance.Contrast(image).enhance(1.38)
    if condition == "jpeg_compression":
        return image
    raise ValueError(f"Unsupported robustness condition: {condition}")


def prepare_condition_images(
    records: list[dict[str, Any]],
    image_root: Path,
    output_dir: Path,
    condition: str,
    seed: int,
) -> Path:
    condition_dir = output_dir / "images" / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for record in records:
        source = find_image(image_root, str(record.get("image", "")))
        if source is None:
            continue
        image = Image.open(source).convert("RGB")
        transformed = apply_condition(image, condition, rng)
        target = condition_dir / Path(str(record.get("image", source.name))).name
        if condition == "jpeg_compression":
            transformed.save(target, quality=35, optimize=True)
        else:
            transformed.save(target)
    return condition_dir


def infer_yolo_records(
    weights: Path,
    source: Path,
    imgsz: int,
    conf: float,
    device: str | int,
) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict(source=str(source), imgsz=imgsz, conf=conf, device=device, verbose=False)
    payload: list[dict[str, Any]] = []
    for result in results:
        objects = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            for coords, score, class_id in zip(xyxy, confs, classes):
                class_idx = int(class_id)
                raw_name = str(model.names.get(class_idx, class_idx))
                objects.append(
                    {
                        "class": normalize_class_name(raw_name, class_idx),
                        "class_id": class_idx,
                        "bbox": [float(value) for value in coords],
                        "confidence": float(score),
                    }
                )
        payload.append(
            {
                "image": Path(str(result.path)).name,
                "image_path": str(result.path),
                "width": int(result.orig_shape[1]),
                "height": int(result.orig_shape[0]),
                "objects": objects,
            }
        )
    return payload


def postprocess_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        postprocess_record(
            record,
            thresholds=DEFAULT_THRESHOLDS,
            axis_nms_rules=DEFAULT_AXIS_NMS,
            top_one_classes=DEFAULT_TOP_ONE_CLASSES,
            project_spans=True,
            span_overlap_threshold=0.60,
            snap_span_bbox=False,
        )
        for record in records
    ]


def flatten_summary(condition: str, summary: dict[str, Any]) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "condition": condition,
        "images": int(summary["images"]),
        "all_count_exact": float(summary["all_class_count_exact_rate"]),
    }
    for class_name in STRUCTURE_CLASSES:
        metric = summary["per_class"][class_name]
        prefix = class_name.replace("table ", "").replace(" ", "_")
        row[f"{prefix}_f1"] = float(metric["f1"])
        row[f"{prefix}_precision"] = float(metric["precision"])
        row[f"{prefix}_recall"] = float(metric["recall"])
        row[f"{prefix}_count_exact"] = float(metric["count_exact_rate"])
    return row


def write_summary_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [str(row["condition"]) for row in rows]
    x = np.arange(len(conditions))
    width = 0.22
    series = [
        ("all_count_exact", "All exact"),
        ("row_f1", "Row F1"),
        ("column_f1", "Column F1"),
        ("spanning_cell_f1", "Span F1"),
    ]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for idx, (key, label) in enumerate(series):
        values = [float(row[key]) for row in rows]
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=label)
    ax.set_title("Robustness under Image Degradation", fontsize=16, fontweight="bold", pad=14)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, [condition.replace("_", "\n") for condition in conditions], fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.savefig(path, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robustness under non-geometric image degradations.")
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = args.output_dir / "runtime"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_dir / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_dir / "matplotlib"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    gt_records = select_records(load_structure_records(args.gt_json), samples=args.samples, seed=args.seed)
    subset_gt = args.output_dir / "gt_subset.jsonl"
    write_structure_records(subset_gt, gt_records, jsonl=True)

    device = select_device(args.device)
    summary_rows: list[dict[str, float | str | int]] = []
    manifest = {
        "gt_json": str(args.gt_json),
        "image_root": str(args.image_root),
        "weights": str(args.weights),
        "output_dir": str(args.output_dir),
        "samples": len(gt_records),
        "seed": args.seed,
        "conditions": CONDITIONS,
        "device": str(device),
    }

    for idx, condition in enumerate(CONDITIONS):
        print(f"[{condition}] preparing images", flush=True)
        image_dir = prepare_condition_images(
            gt_records,
            image_root=args.image_root,
            output_dir=args.output_dir,
            condition=condition,
            seed=args.seed + idx * 997,
        )
        print(f"[{condition}] inference", flush=True)
        raw_pred = infer_yolo_records(
            weights=args.weights,
            source=image_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
        )
        raw_path = args.output_dir / f"pred_{condition}_raw.json"
        write_structure_records(raw_path, raw_pred, jsonl=False)

        print(f"[{condition}] postprocess + evaluate", flush=True)
        post_pred = postprocess_records(raw_pred)
        post_path = args.output_dir / f"pred_{condition}_post.json"
        write_structure_records(post_path, post_pred, jsonl=False)
        summary = evaluate_records(gt_records, post_pred, iou_threshold=0.5)
        eval_path = args.output_dir / f"eval_{condition}.json"
        eval_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary_rows.append(flatten_summary(condition, summary))
        print(
            f"[{condition}] all_exact={summary['all_class_count_exact_rate']:.3f} "
            f"row_f1={summary['per_class']['table row']['f1']:.3f} "
            f"col_f1={summary['per_class']['table column']['f1']:.3f} "
            f"span_f1={summary['per_class']['table spanning cell']['f1']:.3f}",
            flush=True,
        )

    write_summary_csv(args.output_dir / "robustness_summary.csv", summary_rows)
    (args.output_dir / "robustness_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    plot_summary(args.output_dir / "robustness_summary.png", summary_rows)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"summary={args.output_dir / 'robustness_summary.csv'}")


if __name__ == "__main__":
    main()
