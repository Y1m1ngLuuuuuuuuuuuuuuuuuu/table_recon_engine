from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


STRUCTURE_CLASSES = [
    "table",
    "table row",
    "table column",
    "table spanning cell",
    "table column header",
    "table projected row header",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(STRUCTURE_CLASSES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}

SHORT_TO_FULL_CLASS = {
    "table": "table",
    "row": "table row",
    "column": "table column",
    "spanning cell": "table spanning cell",
    "column header": "table column header",
    "projected row header": "table projected row header",
}


def normalize_class_name(name: str, class_id: int | None = None) -> str:
    clean = name.strip()
    if clean in CLASS_TO_ID:
        return clean
    if clean in SHORT_TO_FULL_CLASS:
        return SHORT_TO_FULL_CLASS[clean]
    if class_id is not None:
        try:
            idx = int(class_id)
        except (TypeError, ValueError):
            idx = -1
        if idx in ID_TO_CLASS:
            return ID_TO_CLASS[idx]
    return clean


def class_id_for(name: str) -> int:
    full_name = normalize_class_name(name)
    if full_name not in CLASS_TO_ID:
        raise KeyError(f"Unknown structure class: {name!r}")
    return CLASS_TO_ID[full_name]


def _xml_text(parent: ET.Element, path: str, default: str = "") -> str:
    value = parent.findtext(path)
    return value.strip() if value else default


def _parse_xml_box(obj: ET.Element) -> list[float] | None:
    box = obj.find("bndbox")
    if box is None:
        return None
    try:
        x0 = float(_xml_text(box, "xmin"))
        y0 = float(_xml_text(box, "ymin"))
        x1 = float(_xml_text(box, "xmax"))
        y1 = float(_xml_text(box, "ymax"))
    except ValueError:
        return None
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def clamp_bbox(bbox: Iterable[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return [
        min(max(x0, 0.0), float(width)),
        min(max(y0, 0.0), float(height)),
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
    ]


def bbox_area(bbox: Iterable[float]) -> float:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in a]
    bx0, by0, bx1, by1 = [float(value) for value in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = bbox_area((ax0, ay0, ax1, ay1)) + bbox_area((bx0, by0, bx1, by1)) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def pubtables_xml_to_record(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    filename = _xml_text(root, "filename", f"{xml_path.stem}.jpg")
    width = int(float(_xml_text(root, "size/width", "0")))
    height = int(float(_xml_text(root, "size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        name = normalize_class_name(_xml_text(obj, "name"))
        if name not in CLASS_TO_ID:
            continue
        bbox = _parse_xml_box(obj)
        if bbox is None:
            continue
        bbox = clamp_bbox(bbox, width, height)
        if bbox_area(bbox) <= 1.0:
            continue
        objects.append(
            {
                "class": name,
                "class_id": CLASS_TO_ID[name],
                "bbox": bbox,
            }
        )

    return {
        "image": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def object_to_yolo_row(obj: dict[str, Any], width: int, height: int) -> str | None:
    class_name = normalize_class_name(str(obj.get("class", "")), obj.get("class_id"))
    if class_name not in CLASS_TO_ID:
        return None
    x0, y0, x1, y1 = clamp_bbox(obj["bbox"], width, height)
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w <= 1.0 or box_h <= 1.0:
        return None
    cx = (x0 + x1) * 0.5 / width
    cy = (y0 + y1) * 0.5 / height
    return f"{CLASS_TO_ID[class_name]} {cx:.8f} {cy:.8f} {box_w / width:.8f} {box_h / height:.8f}"


def load_structure_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "annotations", "images", "data"):
            if isinstance(data.get(key), list):
                return list(data[key])
    raise ValueError(f"Unsupported structure JSON format: {path}")


def write_structure_records(path: Path, records: list[dict[str, Any]], jsonl: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        payload = "\n".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records)
        path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        return
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_key(record: dict[str, Any]) -> str:
    image = record.get("image") or record.get("image_path") or ""
    return Path(str(image)).stem
