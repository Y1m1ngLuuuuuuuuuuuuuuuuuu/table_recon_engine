from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from table_recon_engine.converters.common import (
    find_image,
    first_existing,
    image_size,
    place_image,
    prepare_yolo_dirs,
    write_dataset_yaml,
    write_yolo_label,
)
from table_recon_engine.data_structures import DetectionBox


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _parse_points(points: str) -> DetectionBox | None:
    coords: list[tuple[float, float]] = []
    for item in points.replace(";", " ").split():
        if "," not in item:
            continue
        x_text, y_text = item.split(",", 1)
        coords.append((float(x_text), float(y_text)))
    if not coords:
        return None
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    return DetectionBox(min(xs), min(ys), max(xs), max(ys)).ordered()


def _box_from_attrs(attrs: dict[str, str]) -> DetectionBox | None:
    x0 = first_existing(attrs, ("x0", "xmin", "left", "l"))
    y0 = first_existing(attrs, ("y0", "ymin", "top", "t"))
    x1 = first_existing(attrs, ("x1", "xmax", "right", "r"))
    y1 = first_existing(attrs, ("y1", "ymax", "bottom", "b"))
    if None not in (x0, y0, x1, y1):
        return DetectionBox(float(x0), float(y0), float(x1), float(y1)).ordered()

    x = first_existing(attrs, ("x", "left"))
    y = first_existing(attrs, ("y", "top"))
    width = first_existing(attrs, ("width", "w"))
    height = first_existing(attrs, ("height", "h"))
    if None not in (x, y, width, height):
        return DetectionBox(float(x), float(y), float(x) + float(width), float(y) + float(height))

    points = first_existing(attrs, ("points", "Coords"))
    if points is not None:
        return _parse_points(str(points))
    return None


def _box_from_element(element: ET.Element) -> DetectionBox | None:
    direct = _box_from_attrs(dict(element.attrib))
    if direct is not None:
        return direct
    for child in element.iter():
        tag = _strip_ns(child.tag)
        if tag in {"coords", "polygon"} or "coord" in tag:
            box = _box_from_attrs(dict(child.attrib))
            if box is not None:
                return box
    return None


def parse_ctdar_xml(path: Path) -> list[DetectionBox]:
    root = ET.parse(path).getroot()
    boxes: list[DetectionBox] = []
    candidates = [el for el in root.iter() if "cell" in _strip_ns(el.tag)]
    if not candidates:
        candidates = [el for el in root.iter() if _strip_ns(el.tag) in {"coords", "polygon"}]
    for element in candidates:
        box = _box_from_element(element)
        if box is not None:
            boxes.append(box)
    return boxes


def convert_ctdar(
    annotation_root: Path,
    image_root: Path,
    output_dir: Path,
    split: str,
    copy_images: bool,
) -> int:
    image_out, label_out = prepare_yolo_dirs(output_dir, split)
    converted = 0
    for xml_path in sorted(annotation_root.rglob("*.xml")):
        src_image = find_image(image_root, xml_path.stem)
        width, height = image_size(src_image)
        boxes = parse_ctdar_xml(xml_path)
        dst_image = place_image(src_image, image_out, copy_images=copy_images)
        write_yolo_label(label_out / f"{dst_image.stem}.txt", boxes, width, height)
        converted += 1
    write_dataset_yaml(output_dir, dataset_name="ctdar")
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ICDAR cTDaR XML annotations to YOLO labels.")
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_ctdar(
        annotation_root=args.annotation_root,
        image_root=args.image_root,
        output_dir=args.output_dir,
        split=args.split,
        copy_images=args.copy_images,
    )
    print(f"Converted {count} XML files into {args.output_dir}")


if __name__ == "__main__":
    main()
