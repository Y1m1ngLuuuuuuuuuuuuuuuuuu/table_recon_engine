from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from table_recon_engine.demo_structure import parse_result, reconstruct


def read_filelist(extracted_dir: Path, split: str) -> list[str]:
    filelist = extracted_dir / f"{split}_filelist.txt"
    return [Path(line.strip()).name for line in filelist.read_text(encoding="utf-8").splitlines() if line.strip()]


def xml_counts(xml_path: Path) -> dict[str, int]:
    root = ET.parse(xml_path).getroot()
    counts = Counter(obj.findtext("name", "") for obj in root.findall("object"))
    return {
        "rows": counts["table row"],
        "cols": counts["table column"],
        "spans": counts["table spanning cell"],
    }


def image_name_from_xml(xml_path: Path) -> str:
    root = ET.parse(xml_path).getroot()
    return root.findtext("filename", f"{xml_path.stem}.jpg")


def evaluate(
    weights: Path,
    extracted_dir: Path,
    split: str,
    max_samples: int,
    seed: int,
    conf: float,
    imgsz: int,
    output_json: Path,
) -> dict:
    names = read_filelist(extracted_dir, split)
    rng = random.Random(seed)
    rng.shuffle(names)
    names = names[:max_samples]
    model = YOLO(str(weights))

    records = []
    exact_shape = 0
    exact_rows = 0
    exact_cols = 0
    exact_spans = 0
    processed = 0

    for xml_name in names:
        xml_path = extracted_dir / xml_name
        if not xml_path.exists():
            continue
        image_path = extracted_dir / image_name_from_xml(xml_path)
        if not image_path.exists():
            continue
        gt = xml_counts(xml_path)
        result = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        boxes = parse_result(result)
        _latex, pred_struct = reconstruct(boxes, placeholder_mode="blank")
        pred = {
            "rows": int(pred_struct["shape"][0]),
            "cols": int(pred_struct["shape"][1]),
            "spans": len(pred_struct["spans"]),
        }
        row_ok = pred["rows"] == gt["rows"]
        col_ok = pred["cols"] == gt["cols"]
        span_ok = pred["spans"] == gt["spans"]
        shape_ok = row_ok and col_ok
        exact_rows += int(row_ok)
        exact_cols += int(col_ok)
        exact_spans += int(span_ok)
        exact_shape += int(shape_ok)
        processed += 1
        records.append(
            {
                "xml": xml_name,
                "image": str(image_path),
                "gt": gt,
                "pred": pred,
                "shape_ok": shape_ok,
                "rows_delta": pred["rows"] - gt["rows"],
                "cols_delta": pred["cols"] - gt["cols"],
                "spans_delta": pred["spans"] - gt["spans"],
            }
        )

    summary = {
        "weights": str(weights),
        "split": split,
        "conf": conf,
        "imgsz": imgsz,
        "samples": processed,
        "row_accuracy": exact_rows / max(processed, 1),
        "col_accuracy": exact_cols / max(processed, 1),
        "shape_accuracy": exact_shape / max(processed, 1),
        "span_count_accuracy": exact_spans / max(processed, 1),
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate row/column count agreement against PubTables XML.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--extracted-dir", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(
        weights=args.weights,
        extracted_dir=args.extracted_dir,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        conf=args.conf,
        imgsz=args.imgsz,
        output_json=args.output_json,
    )
    print(
        "samples={samples} row_acc={row_accuracy:.3f} col_acc={col_accuracy:.3f} "
        "shape_acc={shape_accuracy:.3f} span_acc={span_count_accuracy:.3f}".format(**summary)
    )


if __name__ == "__main__":
    main()
