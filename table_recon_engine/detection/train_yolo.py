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
    single_cls: bool,
    amp: bool,
    workers: int,
    patience: int,
    seed: int,
    cache: bool | str,
    exist_ok: bool,
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
        single_cls=single_cls,
        amp=amp,
        workers=workers,
        patience=patience,
        seed=seed,
        cache=cache,
        exist_ok=exist_ok,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for table structure detection.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/table_cells"))
    parser.add_argument("--name", default="yolo_structure_detector")
    parser.add_argument("--device", default=None)
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP checks/training.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache",
        nargs="?",
        const=True,
        default=False,
        help="Enable Ultralytics dataset cache. Use --cache ram or --cache disk for explicit modes.",
    )
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
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
        single_cls=args.single_cls,
        amp=not args.no_amp,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
