from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from table_recon_engine.converters.common import prepare_yolo_dirs


PUBTABLES_CLASSES = {
    "table": 0,
    "table row": 1,
    "table column": 2,
    "table spanning cell": 3,
    "table column header": 4,
    "table projected row header": 5,
}


def _text(parent: ET.Element, path: str, default: str = "") -> str:
    value = parent.findtext(path)
    return value.strip() if value else default


def _parse_box(obj: ET.Element) -> tuple[float, float, float, float] | None:
    box = obj.find("bndbox")
    if box is None:
        return None
    try:
        x0 = float(_text(box, "xmin"))
        y0 = float(_text(box, "ymin"))
        x1 = float(_text(box, "xmax"))
        y1 = float(_text(box, "ymax"))
    except ValueError:
        return None
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _to_yolo_row(class_id: int, box: tuple[float, float, float, float], width: int, height: int) -> str | None:
    x0, y0, x1, y1 = box
    x0 = min(max(x0, 0.0), float(width))
    y0 = min(max(y0, 0.0), float(height))
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w <= 1.0 or box_h <= 1.0:
        return None
    cx = (x0 + x1) * 0.5 / width
    cy = (y0 + y1) * 0.5 / height
    return f"{class_id} {cx:.8f} {cy:.8f} {box_w / width:.8f} {box_h / height:.8f}"


def parse_pubtables_xml(
    xml_path: Path,
    class_map: dict[str, int],
) -> tuple[str, int, int, list[str]]:
    root = ET.parse(xml_path).getroot()
    filename = _text(root, "filename", f"{xml_path.stem}.jpg")
    width = int(float(_text(root, "size/width", "0")))
    height = int(float(_text(root, "size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    rows: list[str] = []
    for obj in root.findall("object"):
        name = _text(obj, "name")
        if name not in class_map:
            continue
        box = _parse_box(obj)
        if box is None:
            continue
        row = _to_yolo_row(class_map[name], box, width, height)
        if row is not None:
            rows.append(row)
    return filename, width, height, rows


def _read_filelist(filelist: Path) -> list[str]:
    items = []
    for line in filelist.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(Path(line).name)
    return items


def _place_image(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def write_dataset_yaml(output_dir: Path, class_map: dict[str, int]) -> Path:
    names = {idx: name.replace("table ", "") for name, idx in class_map.items()}
    yaml_path = output_dir / "data.yaml"
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx in sorted(names):
        lines.append(f"  {idx}: {names[idx]}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def convert_split(
    extracted_dir: Path,
    output_dir: Path,
    split: str,
    max_samples: int | None,
    seed: int,
    copy_images: bool,
    class_map: dict[str, int],
) -> dict[str, int]:
    filelist = extracted_dir / f"{split}_filelist.txt"
    xml_names = _read_filelist(filelist)
    rng = random.Random(seed)
    rng.shuffle(xml_names)
    if max_samples is not None:
        xml_names = xml_names[:max_samples]

    image_out, label_out = prepare_yolo_dirs(output_dir, split)
    converted = 0
    skipped_missing_image = 0
    empty_labels = 0
    objects = 0

    for xml_name in xml_names:
        xml_path = extracted_dir / xml_name
        if not xml_path.exists():
            continue
        image_name, _width, _height, rows = parse_pubtables_xml(xml_path, class_map)
        image_path = extracted_dir / image_name
        if not image_path.exists():
            skipped_missing_image += 1
            continue
        if not rows:
            empty_labels += 1
            continue
        _place_image(image_path, image_out / image_name, copy_images=copy_images)
        (label_out / f"{Path(image_name).stem}.txt").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        converted += 1
        objects += len(rows)

    return {
        "converted": converted,
        "skipped_missing_image": skipped_missing_image,
        "empty_labels": empty_labels,
        "objects": objects,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PubTables-1M Structure VOC XML to YOLO labels.")
    parser.add_argument("--extracted-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(PUBTABLES_CLASSES.keys()),
        choices=list(PUBTABLES_CLASSES.keys()),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_map = {name: idx for idx, name in enumerate(args.classes)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_stats = convert_split(
        extracted_dir=args.extracted_dir,
        output_dir=args.output_dir,
        split="train",
        max_samples=args.train_samples,
        seed=args.seed,
        copy_images=args.copy_images,
        class_map=class_map,
    )
    val_stats = convert_split(
        extracted_dir=args.extracted_dir,
        output_dir=args.output_dir,
        split="val",
        max_samples=args.val_samples,
        seed=args.seed + 1,
        copy_images=args.copy_images,
        class_map=class_map,
    )
    yaml_path = write_dataset_yaml(args.output_dir, class_map)
    print(f"data_yaml={yaml_path}")
    print(f"train={train_stats}")
    print(f"val={val_stats}")


if __name__ == "__main__":
    main()
