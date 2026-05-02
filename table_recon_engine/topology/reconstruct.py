from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from table_recon_engine.data_structures import DetectionBox
from table_recon_engine.topology import CellCluster, GridBuilder, LaTeXGenerator


def reconstruct_record(record: dict) -> str:
    detections = [
        DetectionBox(
            x0=float(box["bbox"][0]),
            y0=float(box["bbox"][1]),
            x1=float(box["bbox"][2]),
            y1=float(box["bbox"][3]),
            score=float(box.get("score", 1.0)),
            class_id=int(box.get("class_id", 0)),
            text=str(box.get("text", "")),
        )
        for box in record.get("boxes", [])
    ]
    cells, rows, cols = CellCluster().cluster(detections)
    grid = GridBuilder().build(cells, rows, cols)
    return LaTeXGenerator().generate(grid)


def reconstruct_json(detection_json: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(detection_json.read_text(encoding="utf-8"))
    for idx, record in enumerate(records):
        latex = reconstruct_record(record)
        stem = Path(record.get("image_path", f"table_{idx}")).stem
        (output_dir / f"{stem}.tex").write_text(latex + "\n", encoding="utf-8")
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct LaTeX tables from detection JSON.")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latex"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = reconstruct_json(args.detections, args.output_dir)
    print(f"Wrote {count} LaTeX table(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
