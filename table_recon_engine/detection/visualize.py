from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def draw_detections(detection_json: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(detection_json.read_text(encoding="utf-8"))
    count = 0
    for record in records:
        image = cv2.imread(record["image_path"])
        if image is None:
            continue
        for box in record.get("boxes", []):
            x0, y0, x1, y1 = [int(round(value)) for value in box["bbox"]]
            cv2.rectangle(image, (x0, y0), (x1, y1), (38, 160, 255), 2)
            cv2.putText(
                image,
                f"{box.get('score', 0):.2f}",
                (x0, max(0, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (38, 160, 255),
                1,
                cv2.LINE_AA,
            )
        out_path = output_dir / Path(record["image_path"]).name
        cv2.imwrite(str(out_path), image)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw detection boxes from inference JSON.")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = draw_detections(args.detections, args.output_dir)
    print(f"Wrote {count} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
