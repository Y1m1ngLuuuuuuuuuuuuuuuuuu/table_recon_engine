from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import albumentations as A
import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_transform() -> A.Compose:
    return A.Compose(
        [
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.25,
            ),
            A.GaussNoise(var_limit=(8.0, 45.0), p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.35),
            A.Perspective(scale=(0.02, 0.06), keep_size=True, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.08,
                rotate_limit=2,
                border_mode=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
                p=0.35,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.25,
        ),
    )


def _read_yolo_labels(path: Path) -> tuple[list[list[float]], list[int]]:
    if not path.exists():
        return [], []
    boxes: list[list[float]] = []
    labels: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        labels.append(int(float(parts[0])))
        boxes.append([float(value) for value in parts[1:5]])
    return boxes, labels


def _write_yolo_labels(path: Path, boxes: list[list[float]], labels: list[int]) -> None:
    rows = []
    for label, box in zip(labels, boxes):
        cx, cy, width, height = box
        if width > 0 and height > 0:
            rows.append(f"{label} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def augment_split(
    dataset_dir: Path,
    split: str,
    repeats: int,
    keep_original: bool,
) -> int:
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    transform = build_transform()

    images = [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    written = 0
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if keep_original:
            shutil.copy2(image_path, image_dir / f"{image_path.stem}_orig{image_path.suffix}")
            shutil.copy2(label_path, label_dir / f"{image_path.stem}_orig.txt")

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, labels = _read_yolo_labels(label_path)
        for idx in range(repeats):
            result = transform(image=image, bboxes=boxes, class_labels=labels)
            out_image = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            out_stem = f"{image_path.stem}_aug{idx:02d}"
            cv2.imwrite(str(image_dir / f"{out_stem}{image_path.suffix}"), out_image)
            _write_yolo_labels(
                label_dir / f"{out_stem}.txt",
                list(result["bboxes"]),
                [int(label) for label in result["class_labels"]],
            )
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Albumentations to a YOLO dataset split.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--keep-original", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = augment_split(
        dataset_dir=args.dataset_dir,
        split=args.split,
        repeats=args.repeats,
        keep_original=args.keep_original,
    )
    print(f"Wrote {count} augmented images for split={args.split}")


if __name__ == "__main__":
    main()
