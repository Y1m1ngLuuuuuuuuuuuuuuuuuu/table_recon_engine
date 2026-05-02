from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from table_recon_engine.structure_json import STRUCTURE_CLASSES, load_structure_records, object_to_yolo_row


def prepare_yolo_dirs(output_dir: Path, split: str) -> tuple[Path, Path]:
    image_out = output_dir / "images" / split
    label_out = output_dir / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    return image_out, label_out


def _place_image(src: Path, dst: Path, copy_images: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        return True
    if copy_images:
        shutil.copy2(src, dst)
        return True
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)
    return True


def write_dataset_yaml(output_dir: Path) -> Path:
    yaml_path = output_dir / "data.yaml"
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(STRUCTURE_CLASSES):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def convert_split(
    annotations: Path,
    image_root: Path,
    output_dir: Path,
    split: str,
    copy_images: bool,
) -> dict[str, int | str]:
    records = load_structure_records(annotations)
    image_out, label_out = prepare_yolo_dirs(output_dir, split)

    converted = 0
    missing_images = 0
    empty_labels = 0
    objects = 0
    for record in records:
        image_name = Path(str(record["image"])).name
        image_path = image_root / image_name
        if not _place_image(image_path, image_out / image_name, copy_images=copy_images):
            missing_images += 1
            continue

        width = int(record["width"])
        height = int(record["height"])
        rows = []
        for obj in record.get("objects", []):
            row = object_to_yolo_row(obj, width, height)
            if row is not None:
                rows.append(row)

        if not rows:
            empty_labels += 1
            continue
        (label_out / f"{Path(image_name).stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        converted += 1
        objects += len(rows)

    return {
        "split": split,
        "records": len(records),
        "converted": converted,
        "missing_images": missing_images,
        "empty_labels": empty_labels,
        "objects": objects,
        "annotations": str(annotations),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert standard structure JSON records to a YOLO dataset.")
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = convert_split(
        annotations=args.train_annotations,
        image_root=args.image_root,
        output_dir=args.output_dir,
        split="train",
        copy_images=args.copy_images,
    )
    val = convert_split(
        annotations=args.val_annotations,
        image_root=args.image_root,
        output_dir=args.output_dir,
        split="val",
        copy_images=args.copy_images,
    )
    data_yaml = write_dataset_yaml(args.output_dir)
    manifest = {"train": train, "val": val, "data_yaml": str(data_yaml)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
