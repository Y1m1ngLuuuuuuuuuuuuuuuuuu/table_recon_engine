from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


@dataclass(slots=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    name: str

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) * 0.5

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) * 0.5

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


@dataclass(slots=True)
class Span:
    row: int
    col: int
    rowspan: int
    colspan: int


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iou(a: Box, b: Box) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list[Box], threshold: float = 0.75) -> list[Box]:
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: item.score, reverse=True):
        if all(iou(box, old) < threshold for old in kept):
            kept.append(box)
    return kept


def parse_result(result) -> list[Box]:
    parsed: list[Box] = []
    if result.boxes is None:
        return parsed
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    confs = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    names = result.names
    for coords, score, cls_id in zip(xyxy, confs, classes):
        x0, y0, x1, y1 = coords
        parsed.append(Box(x0, y0, x1, y1, float(score), str(names[int(cls_id)])))
    return parsed


def covered_indexes(start: float, end: float, centers: list[float], margin_ratio: float = 0.04) -> list[int]:
    width = max(1.0, end - start)
    margin = width * margin_ratio
    return [idx for idx, center in enumerate(centers) if start - margin <= center <= end + margin]


def infer_spans(rows: list[Box], cols: list[Box], span_boxes: list[Box]) -> list[Span]:
    row_centers = [row.cy for row in rows]
    col_centers = [col.cx for col in cols]
    spans: list[Span] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in sorted(span_boxes, key=lambda item: item.score, reverse=True):
        covered_rows = covered_indexes(box.y0, box.y1, row_centers)
        covered_cols = covered_indexes(box.x0, box.x1, col_centers)
        if not covered_rows or not covered_cols:
            continue
        row = min(covered_rows)
        col = min(covered_cols)
        rowspan = max(1, len(covered_rows))
        colspan = max(1, len(covered_cols))
        if rowspan == 1 and colspan == 1:
            continue
        key = (row, col, rowspan, colspan)
        if key not in seen:
            seen.add(key)
            spans.append(Span(row=row, col=col, rowspan=rowspan, colspan=colspan))
    return spans


def placeholder(row: int, col: int, mode: str) -> str:
    if mode == "blank":
        return ""
    return f"r{row + 1}c{col + 1}"


def generate_latex(n_rows: int, n_cols: int, spans: list[Span], placeholder_mode: str) -> str:
    if n_rows <= 0 or n_cols <= 0:
        return "\\begin{tabular}{}\n\\end{tabular}"

    anchors = {(span.row, span.col): span for span in spans}
    covered: dict[tuple[int, int], Span] = {}
    for span in spans:
        for row in range(span.row, min(n_rows, span.row + span.rowspan)):
            for col in range(span.col, min(n_cols, span.col + span.colspan)):
                if (row, col) != (span.row, span.col):
                    covered[(row, col)] = span

    lines = [f"\\begin{{tabular}}{{{'c' * n_cols}}}"]
    for row in range(n_rows):
        parts: list[str] = []
        col = 0
        while col < n_cols:
            span = anchors.get((row, col))
            if span is not None:
                text = placeholder(row, col, placeholder_mode)
                if span.rowspan > 1:
                    text = f"\\multirow{{{span.rowspan}}}{{*}}{{{text}}}"
                if span.colspan > 1:
                    parts.append(f"\\multicolumn{{{span.colspan}}}{{c}}{{{text}}}")
                    col += span.colspan
                else:
                    parts.append(text)
                    col += 1
                continue
            if (row, col) in covered:
                parts.append("")
            else:
                parts.append(placeholder(row, col, placeholder_mode))
            col += 1
        lines.append(" & ".join(parts) + r" \\")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def draw_structure(image_path: Path, output_path: Path, rows: list[Box], cols: list[Box], spans: list[Box]) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for box in cols:
        cv2.rectangle(image, (int(box.x0), int(box.y0)), (int(box.x1), int(box.y1)), (255, 120, 20), 1)
    for box in rows:
        cv2.rectangle(image, (int(box.x0), int(box.y0)), (int(box.x1), int(box.y1)), (40, 180, 40), 1)
    for box in spans:
        cv2.rectangle(image, (int(box.x0), int(box.y0)), (int(box.x1), int(box.y1)), (30, 30, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def reconstruct(boxes: list[Box], placeholder_mode: str) -> tuple[str, dict]:
    rows = nms([box for box in boxes if box.name == "row"], threshold=0.8)
    cols = nms([box for box in boxes if box.name == "column"], threshold=0.8)
    span_boxes = nms([box for box in boxes if box.name == "spanning cell"], threshold=0.65)
    rows.sort(key=lambda box: box.cy)
    cols.sort(key=lambda box: box.cx)
    spans = infer_spans(rows, cols, span_boxes)
    latex = generate_latex(len(rows), len(cols), spans, placeholder_mode)
    payload = {
        "rows": [asdict(box) for box in rows],
        "cols": [asdict(box) for box in cols],
        "spanning_cells": [asdict(box) for box in span_boxes],
        "spans": [asdict(span) for span in spans],
        "shape": [len(rows), len(cols)],
    }
    return latex, payload


def collect_images(source: Path, limit: int | None) -> list[Path]:
    if source.is_file():
        return [source]
    images = [path for path in sorted(source.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    if limit is not None:
        images = images[:limit]
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO structure detection and export LaTeX demos.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--placeholder-mode", choices=["coords", "blank"], default="coords")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    images = collect_images(args.source, args.limit)
    summary = []
    for image_path in images:
        result = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        boxes = parse_result(result)
        latex, structure = reconstruct(boxes, args.placeholder_mode)
        stem = image_path.stem
        (args.output_dir / "latex").mkdir(exist_ok=True)
        (args.output_dir / "json").mkdir(exist_ok=True)
        (args.output_dir / "visuals").mkdir(exist_ok=True)
        (args.output_dir / "latex" / f"{stem}.tex").write_text(latex + "\n", encoding="utf-8")
        (args.output_dir / "json" / f"{stem}.json").write_text(
            json.dumps({"image": str(image_path), "detections": [asdict(box) for box in boxes], **structure}, indent=2),
            encoding="utf-8",
        )
        draw_structure(
            image_path=image_path,
            output_path=args.output_dir / "visuals" / f"{stem}_structure.jpg",
            rows=[Box(**item) for item in structure["rows"]],
            cols=[Box(**item) for item in structure["cols"]],
            spans=[Box(**item) for item in structure["spanning_cells"]],
        )
        summary.append({"image": str(image_path), "shape": structure["shape"], "spans": len(structure["spans"])})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(summary)} demo result(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
