from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


COLORS = {
    "table": (40, 40, 40),
    "table row": (40, 180, 40),
    "table column": (255, 130, 20),
    "table spanning cell": (30, 30, 255),
    "table column header": (180, 40, 180),
    "table projected row header": (40, 180, 220),
}


def read_filelist(extracted_dir: Path, split: str) -> list[str]:
    filelist = extracted_dir / f"{split}_filelist.txt"
    return [Path(line.strip()).name for line in filelist.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_xml_names_from_demo(demo_dir: Path) -> list[str]:
    names: list[str] = []
    for record_path in sorted((demo_dir / "json").glob("*.json")):
        names.append(f"{record_path.stem}.xml")
    return names


def parse_box(obj: ET.Element) -> tuple[int, int, int, int] | None:
    box = obj.find("bndbox")
    if box is None:
        return None
    try:
        x0 = int(round(float(box.findtext("xmin", "0"))))
        y0 = int(round(float(box.findtext("ymin", "0"))))
        x1 = int(round(float(box.findtext("xmax", "0"))))
        y1 = int(round(float(box.findtext("ymax", "0"))))
    except ValueError:
        return None
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def draw_xml(xml_path: Path, extracted_dir: Path, output_path: Path, classes: set[str]) -> bool:
    root = ET.parse(xml_path).getroot()
    image_name = root.findtext("filename", f"{xml_path.stem}.jpg")
    image_path = extracted_dir / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    for obj in root.findall("object"):
        name = obj.findtext("name", "")
        if classes and name not in classes:
            continue
        box = parse_box(obj)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        color = COLORS.get(name, (0, 0, 0))
        width = 2 if name in {"table", "table spanning cell"} else 1
        cv2.rectangle(image, (x0, y0), (x1, y1), color, width)
        cv2.putText(
            image,
            name.replace("table ", ""),
            (x0, max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PubTables XML ground-truth boxes on table images.")
    parser.add_argument("--extracted-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-dir", type=Path, default=None, help="Use demo JSON stems instead of random filelist.")
    parser.add_argument("--classes", nargs="*", default=list(COLORS.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = set(args.classes)
    if args.demo_dir is not None:
        xml_names = read_xml_names_from_demo(args.demo_dir)
    else:
        xml_names = read_filelist(args.extracted_dir, args.split)
        rng = random.Random(args.seed)
        rng.shuffle(xml_names)
    xml_names = xml_names[: args.limit]

    rendered = 0
    for xml_name in xml_names:
        xml_path = args.extracted_dir / Path(xml_name).name
        if not xml_path.exists():
            continue
        if draw_xml(xml_path, args.extracted_dir, args.output_dir / f"{xml_path.stem}_gt.jpg", classes):
            rendered += 1

    legend = {
        name: {"bgr": color, "meaning": name.replace("table ", "")}
        for name, color in COLORS.items()
        if name in classes
    }
    (args.output_dir / "legend.json").write_text(json.dumps(legend, indent=2), encoding="utf-8")
    print(f"Wrote {rendered} ground-truth visualization image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
