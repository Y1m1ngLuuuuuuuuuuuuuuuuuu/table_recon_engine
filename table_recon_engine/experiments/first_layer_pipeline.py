from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def prepare_runtime_env(work_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    runtime_dir = work_dir / "runtime"
    yolo_config_dir = runtime_dir / "ultralytics"
    mpl_config_dir = runtime_dir / "matplotlib"
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
    env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    env.setdefault("PYTHONUNBUFFERED", "1")

    font_targets = [yolo_config_dir / "Arial.ttf", yolo_config_dir / "Ultralytics" / "Arial.ttf"]
    font_source_match = None
    if any(not target.exists() for target in font_targets):
        for font_source in (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/Library/Fonts/Arial.ttf"),
        ):
            if font_source.exists():
                font_source_match = font_source
                break
    if font_source_match is not None:
        for font_target in font_targets:
            if not font_target.exists():
                font_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(font_source_match, font_target)
    return env


def run_step(cmd: list[str], dry_run: bool = False, env: dict[str, str] | None = None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def infer_run_dir(project: Path, name: str) -> Path:
    return project / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first-layer table structure detection pipeline.")
    parser.add_argument("--extracted-dir", type=Path, required=True, help="PubTables-1M extracted directory.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Experiment root under the project workspace.")
    parser.add_argument("--name", default="structure_50k_yolov8s")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset directory suffix. Defaults to --name; set this to reuse prepared JSON/YOLO data.",
    )
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="yolov8s.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-infer-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name or args.name
    json_dir = args.work_dir / "datasets" / f"structure_json_{dataset_name}"
    yolo_dir = args.work_dir / "datasets" / f"yolo_{dataset_name}"
    runs_dir = args.work_dir / "runs"
    outputs_dir = args.work_dir / "outputs" / args.name
    run_dir = infer_run_dir(runs_dir, args.name)

    train_json = json_dir / "annotations" / "train.jsonl"
    val_json = json_dir / "annotations" / "val.jsonl"
    data_yaml = yolo_dir / "data.yaml"
    weights = run_dir / "weights" / "best.pt"
    pred_json = outputs_dir / "pred_structure_val.json"
    eval_json = outputs_dir / "eval_structure_val.json"

    manifest = {
        "name": args.name,
        "dataset_name": dataset_name,
        "extracted_dir": str(args.extracted_dir),
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "json_dir": str(json_dir),
        "yolo_dir": str(yolo_dir),
        "run_dir": str(run_dir),
        "outputs_dir": str(outputs_dir),
        "pred_json": str(pred_json),
        "eval_json": str(eval_json),
    }
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    py = sys.executable
    runtime_env = prepare_runtime_env(args.work_dir)
    if not args.skip_prepare:
        run_step(
            [
                py,
                "-m",
                "table_recon_engine.converters.pubtables_xml_to_json",
                "--extracted-dir",
                str(args.extracted_dir),
                "--output-dir",
                str(json_dir),
                "--train-samples",
                str(args.train_samples),
                "--val-samples",
                str(args.val_samples),
                "--seed",
                str(args.seed),
            ],
            dry_run=args.dry_run,
            env=runtime_env,
        )
        yolo_cmd = [
            py,
            "-m",
            "table_recon_engine.converters.structure_json_to_yolo",
            "--train-annotations",
            str(train_json),
            "--val-annotations",
            str(val_json),
            "--image-root",
            str(args.extracted_dir),
            "--output-dir",
            str(yolo_dir),
        ]
        if args.copy_images:
            yolo_cmd.append("--copy-images")
        run_step(yolo_cmd, dry_run=args.dry_run, env=runtime_env)

    if not args.skip_train:
        train_cmd = [
            py,
            "-m",
            "table_recon_engine.detection.train_yolo",
            "--data",
            str(data_yaml),
            "--model",
            args.model,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--workers",
            str(args.workers),
            "--patience",
            str(args.patience),
            "--seed",
            str(args.seed),
            "--project",
            str(runs_dir),
            "--name",
            args.name,
        ]
        if args.device is not None:
            train_cmd.extend(["--device", args.device])
        if args.no_amp:
            train_cmd.append("--no-amp")
        run_step(train_cmd, dry_run=args.dry_run, env=runtime_env)

    if not args.skip_infer_eval:
        run_step(
            [
                py,
                "-m",
                "table_recon_engine.detection.infer_yolo",
                "--weights",
                str(weights),
                "--source",
                str(yolo_dir / "images" / "val"),
                "--output-json",
                str(pred_json),
                "--imgsz",
                str(args.imgsz),
                "--conf",
                str(args.conf),
            ],
            dry_run=args.dry_run,
            env=runtime_env,
        )
        run_step(
            [
                py,
                "-m",
                "table_recon_engine.evaluation.detection_json",
                "--gt-json",
                str(val_json),
                "--pred-json",
                str(pred_json),
                "--output-json",
                str(eval_json),
            ],
            dry_run=args.dry_run,
            env=runtime_env,
        )

    print(f"\nmanifest={outputs_dir / 'pipeline_manifest.json'}")


if __name__ == "__main__":
    main()
