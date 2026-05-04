from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from table_recon_engine.detection.infer_yolo import infer_yolo
from table_recon_engine.postprocess_structure_json import (
    DEFAULT_AXIS_NMS,
    DEFAULT_THRESHOLDS,
    DEFAULT_TOP_ONE_CLASSES,
    postprocess_record,
)
from table_recon_engine.render_structure_json import render_record
from table_recon_engine.structure_json import normalize_class_name, write_structure_records


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLORS = {
    "table": (20, 20, 20),
    "table row": (44, 160, 44),
    "table column": (255, 127, 14),
    "table spanning cell": (214, 39, 40),
    "table column header": (148, 103, 189),
    "table projected row header": (23, 190, 207),
}


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return [path for path in sorted(source.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_boxes(record: dict[str, Any], image_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(13)
    for obj in record.get("objects", []):
        class_name = normalize_class_name(str(obj.get("class", "")), obj.get("class_id"))
        box = obj.get("projected_bbox") if class_name == "table spanning cell" and obj.get("projected_bbox") else obj.get("bbox")
        if not box or len(box) != 4:
            continue
        x0, y0, x1, y1 = [float(value) for value in box]
        color = COLORS.get(class_name, (220, 20, 60))
        width = 4 if class_name == "table spanning cell" else (3 if class_name == "table" else 2)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
        score = obj.get("confidence", obj.get("score"))
        label = class_name.replace("table ", "")
        if score is not None:
            label = f"{label} {float(score):.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        tx = max(0, int(x0))
        ty = max(0, int(y0) - th - 4)
        draw.rectangle((tx, ty, tx + tw + 6, ty + th + 4), fill=(255, 255, 255))
        draw.text((tx + 3, ty + 2), label, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final local demo: YOLO -> post-v3 JSON -> boxes + LaTeX render.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--device", default=None, help="Use mps/cpu/0. Defaults to auto selection.")
    parser.add_argument("--placeholder-mode", choices=["blank", "coords"], default="blank")
    parser.add_argument("--style", choices=["booktabs", "grid"], default="booktabs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(args.source)
    if not images:
        raise FileNotFoundError(f"No demo image found in {args.source}")

    raw_json = args.output_dir / "detections_raw.json"
    raw_records = infer_yolo(
        weights=args.weights,
        source=args.source,
        output_json=raw_json,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save_visuals=False,
    )
    post_records = postprocess_records(raw_records)
    post_json = args.output_dir / "detections_post_v3.json"
    write_structure_records(post_json, post_records, jsonl=False)

    image_by_name = {path.name: path for path in images}
    summaries = []
    for record in post_records:
        image_path = image_by_name.get(Path(str(record.get("image", ""))).name)
        if image_path is None:
            continue
        stem = image_path.stem
        box_path = args.output_dir / "box_overlays" / f"{stem}_boxes.png"
        draw_boxes(record, image_path, box_path)
        render_summary = render_record(
            record=record,
            output_dir=args.output_dir / "latex_renders",
            image_root=args.source if args.source.is_dir() else args.source.parent,
            width=900,
            height=700,
            placeholder_mode=args.placeholder_mode,
            style=args.style,
        )
        summaries.append(
            {
                "image": str(image_path),
                "box_overlay": str(box_path),
                "latex_render": render_summary["render"],
                "latex_comparison": render_summary["comparison"],
                "latex": render_summary["latex"],
                "shape": render_summary["shape"],
                "spans": render_summary["spans"],
            }
        )

    summary = {
        "weights": str(args.weights),
        "source": str(args.source),
        "raw_json": str(raw_json),
        "post_json": str(post_json),
        "results": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summaries)} final demo result(s) to {args.output_dir}")
    print(f"summary={args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
