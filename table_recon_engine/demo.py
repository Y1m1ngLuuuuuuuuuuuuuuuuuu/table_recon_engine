from __future__ import annotations

import argparse
import json
from pathlib import Path

from table_recon_engine.detection.infer_yolo import infer_yolo
from table_recon_engine.topology.reconstruct import reconstruct_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO detection -> topology reconstruction -> LaTeX demo.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-visuals", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detection_json = args.output_dir / "detections.json"
    records = infer_yolo(
        weights=args.weights,
        source=args.source,
        output_json=detection_json,
        imgsz=args.imgsz,
        conf=args.conf,
        device=None,
        save_visuals=args.save_visuals,
    )
    latex_dir = args.output_dir / "latex"
    reconstruct_json(detection_json, latex_dir)
    summary = {
        "images": len(records),
        "detections_json": str(detection_json),
        "latex_dir": str(latex_dir),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Demo finished. Summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
