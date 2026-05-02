from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from table_recon_engine.converters.common import (
    find_image,
    first_existing,
    image_size,
    place_image,
    prepare_yolo_dirs,
    read_json_records,
    write_dataset_yaml,
    write_yolo_label,
)
from table_recon_engine.data_structures import DetectionBox


def _image_name(record: dict[str, Any]) -> str:
    value = first_existing(
        record,
        ("filename", "file_name", "image_path", "img_path", "image", "image_id", "document_id"),
    )
    if value is None:
        raise KeyError("Cannot infer image name from record.")
    return str(value)


def _cell_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("cells", "cell", "annotations"):
        value = record.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    html = record.get("html", {})
    if isinstance(html, dict) and isinstance(html.get("cells"), list):
        return list(html["cells"])
    return []


def _bbox_from_cell(cell: dict[str, Any]) -> DetectionBox | None:
    raw = first_existing(cell, ("bbox", "pdf_bbox", "image_bbox", "box", "rect"))
    if raw is None:
        return None
    if isinstance(raw, dict):
        x0 = first_existing(raw, ("x0", "xmin", "left", "l"))
        y0 = first_existing(raw, ("y0", "ymin", "top", "t"))
        x1 = first_existing(raw, ("x1", "xmax", "right", "r"))
        y1 = first_existing(raw, ("y1", "ymax", "bottom", "b"))
        if None in (x0, y0, x1, y1):
            width = first_existing(raw, ("width", "w"))
            height = first_existing(raw, ("height", "h"))
            if None in (x0, y0, width, height):
                return None
            x1 = float(x0) + float(width)
            y1 = float(y0) + float(height)
        return DetectionBox(float(x0), float(y0), float(x1), float(y1)).ordered()
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        return DetectionBox(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])).ordered()
    return None


def convert_pubtables(
    annotations: Path,
    image_root: Path,
    output_dir: Path,
    split: str,
    copy_images: bool,
) -> int:
    records = read_json_records(annotations)
    image_out, label_out = prepare_yolo_dirs(output_dir, split)

    converted = 0
    for record in records:
        src_image = find_image(image_root, _image_name(record))
        width, height = image_size(src_image)
        boxes = [box for cell in _cell_items(record) if (box := _bbox_from_cell(cell)) is not None]
        dst_image = place_image(src_image, image_out, copy_images=copy_images)
        write_yolo_label(label_out / f"{dst_image.stem}.txt", boxes, width, height)
        converted += 1

    write_dataset_yaml(output_dir, dataset_name="pubtables")
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PubTables/PubTabNet-like JSON to YOLO labels.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_pubtables(
        annotations=args.annotations,
        image_root=args.image_root,
        output_dir=args.output_dir,
        split=args.split,
        copy_images=args.copy_images,
    )
    print(f"Converted {count} images into {args.output_dir}")


if __name__ == "__main__":
    main()
