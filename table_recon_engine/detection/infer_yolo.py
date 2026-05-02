from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from table_recon_engine.structure_json import normalize_class_name


def select_yolo_device() -> str | int:
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_yolo(
    weights: Path,
    source: Path,
    output_json: Path,
    imgsz: int,
    conf: float,
    device: str | int | None,
    save_visuals: bool,
) -> list[dict[str, Any]]:
    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        device=select_yolo_device() if device is None else device,
        save=save_visuals,
        verbose=False,
    )

    payload: list[dict[str, Any]] = []
    for result in results:
        objects = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            for coords, score, class_id in zip(xyxy, confs, classes):
                class_idx = int(class_id)
                raw_name = str(model.names.get(class_idx, class_idx))
                objects.append(
                    {
                        "class": normalize_class_name(raw_name, class_idx),
                        "class_id": class_idx,
                        "bbox": [float(value) for value in coords],
                        "confidence": float(score),
                    }
                )
        payload.append(
            {
                "image": Path(str(result.path)).name,
                "image_path": str(result.path),
                "width": int(result.orig_shape[1]),
                "height": int(result.orig_shape[0]),
                "objects": objects,
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO structure detector and export standard structure JSON.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/detections.json"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-visuals", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = infer_yolo(
        weights=args.weights,
        source=args.source,
        output_json=args.output_json,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save_visuals=args.save_visuals,
    )
    print(f"Wrote detections for {len(payload)} image(s) to {args.output_json}")


if __name__ == "__main__":
    main()
