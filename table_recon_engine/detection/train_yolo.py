from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def select_yolo_device() -> str | int:
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_yolo(
    data_yaml: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: Path,
    name: str,
    device: str | int | None,
) -> None:
    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=name,
        device=select_yolo_device() if device is None else device,
        single_cls=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for table-cell detection.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/table_cells"))
    parser.add_argument("--name", default="yolo_cell_detector")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
