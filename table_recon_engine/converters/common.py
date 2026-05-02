from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from table_recon_engine.data_structures import DetectionBox

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    for key in ("annotations", "data", "records", "images"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return list(data[key])
    raise ValueError(f"Unsupported JSON structure: {path}")


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def find_image(image_root: Path, name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    direct = image_root / candidate
    if direct.exists():
        return direct

    stem = candidate.stem or candidate.name
    for suffix in IMAGE_SUFFIXES:
        match = image_root / f"{stem}{suffix}"
        if match.exists():
            return match
    matches = list(image_root.rglob(f"{stem}.*"))
    for match in matches:
        if match.suffix.lower() in IMAGE_SUFFIXES:
            return match
    raise FileNotFoundError(f"Cannot find image for {name_or_path!r} under {image_root}")


def prepare_yolo_dirs(output_dir: Path, split: str) -> tuple[Path, Path]:
    image_out = output_dir / "images" / split
    label_out = output_dir / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    return image_out, label_out


def place_image(src: Path, dst_dir: Path, copy_images: bool) -> Path:
    dst = dst_dir / src.name
    if dst.exists():
        return dst
    if copy_images:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
    return dst


def write_yolo_label(path: Path, boxes: Iterable[DetectionBox], width: int, height: int) -> None:
    rows = []
    for box in boxes:
        clean = box.clamp(width, height)
        if clean.width > 1.0 and clean.height > 1.0:
            rows.append(clean.to_yolo(width, height))
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_dataset_yaml(output_dir: Path, dataset_name: str = "table-cells") -> Path:
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: cell",
                f"# dataset: {dataset_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def first_existing(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None
