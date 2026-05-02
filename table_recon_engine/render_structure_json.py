from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from table_recon_engine.structure_json import load_structure_records, normalize_class_name


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class JsonBox:
    x0: float
    y0: float
    x1: float
    y1: float
    name: str
    score: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) * 0.5

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) * 0.5

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


@dataclass(slots=True)
class SpanCell:
    row: int
    col: int
    rowspan: int
    colspan: int


@dataclass(slots=True)
class TableStructure:
    rows: list[JsonBox]
    cols: list[JsonBox]
    spans: list[SpanCell]
    table_box: JsonBox | None

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.cols)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def record_boxes(record: dict[str, Any]) -> list[JsonBox]:
    boxes: list[JsonBox] = []
    for obj in record.get("objects", []):
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(value) for value in bbox]
        name = normalize_class_name(str(obj.get("class", "")), obj.get("class_id"))
        score = float(obj.get("confidence", obj.get("score", 1.0)))
        boxes.append(JsonBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), name, score))
    return boxes


def nms_1d(boxes: list[JsonBox], axis: str, overlap_threshold: float = 0.72) -> list[JsonBox]:
    kept: list[JsonBox] = []
    for box in sorted(boxes, key=lambda item: item.score, reverse=True):
        if all(overlap_score(box, old, axis) < overlap_threshold for old in kept):
            kept.append(box)
    return kept


def overlap_score(a: JsonBox, b: JsonBox, axis: str) -> float:
    if axis == "xy":
        ix0 = max(a.x0, b.x0)
        iy0 = max(a.y0, b.y0)
        ix1 = min(a.x1, b.x1)
        iy1 = min(a.y1, b.y1)
        intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = a.width * a.height + b.width * b.height - intersection
        return intersection / union if union > 0 else 0.0
    return axis_overlap_ratio(a, b, axis)


def axis_overlap_ratio(a: JsonBox, b: JsonBox, axis: str) -> float:
    if axis == "y":
        start = max(a.y0, b.y0)
        end = min(a.y1, b.y1)
        denom = min(max(a.height, 1.0), max(b.height, 1.0))
    else:
        start = max(a.x0, b.x0)
        end = min(a.x1, b.x1)
        denom = min(max(a.width, 1.0), max(b.width, 1.0))
    return max(0.0, end - start) / denom


def covered_indexes(start: float, end: float, centers: list[float], margin_ratio: float = 0.035) -> list[int]:
    width = max(1.0, end - start)
    margin = width * margin_ratio
    return [idx for idx, center in enumerate(centers) if start - margin <= center <= end + margin]


def build_structure(record: dict[str, Any]) -> TableStructure:
    boxes = record_boxes(record)
    table_boxes = sorted([box for box in boxes if box.name == "table"], key=lambda item: item.score, reverse=True)
    rows = nms_1d([box for box in boxes if box.name == "table row"], axis="y")
    cols = nms_1d([box for box in boxes if box.name == "table column"], axis="x")
    span_boxes = nms_1d([box for box in boxes if box.name == "table spanning cell"], axis="xy")
    rows.sort(key=lambda item: item.cy)
    cols.sort(key=lambda item: item.cx)

    row_centers = [box.cy for box in rows]
    col_centers = [box.cx for box in cols]
    spans: list[SpanCell] = []
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
            spans.append(SpanCell(row, col, rowspan, colspan))

    return TableStructure(rows=rows, cols=cols, spans=spans, table_box=table_boxes[0] if table_boxes else None)


def latex_placeholder(row: int, col: int, mode: str) -> str:
    return "" if mode == "blank" else f"r{row + 1}c{col + 1}"


def generate_latex(structure: TableStructure, placeholder_mode: str, booktabs: bool = True) -> str:
    n_rows, n_cols = structure.shape
    if n_rows <= 0 or n_cols <= 0:
        return "\\begin{tabular}{}\n\\end{tabular}"

    anchors = {(span.row, span.col): span for span in structure.spans}
    covered: set[tuple[int, int]] = set()
    for span in structure.spans:
        for row in range(span.row, min(n_rows, span.row + span.rowspan)):
            for col in range(span.col, min(n_cols, span.col + span.colspan)):
                if (row, col) != (span.row, span.col):
                    covered.add((row, col))

    lines = [r"\begin{tabular}{" + ("c" * n_cols) + "}"]
    if booktabs:
        lines.append(r"\toprule")
    for row in range(n_rows):
        parts: list[str] = []
        col = 0
        while col < n_cols:
            span = anchors.get((row, col))
            if span is not None:
                text = latex_placeholder(row, col, placeholder_mode)
                if span.rowspan > 1:
                    text = f"\\multirow{{{span.rowspan}}}{{*}}{{{text}}}"
                if span.colspan > 1:
                    parts.append(f"\\multicolumn{{{span.colspan}}}{{c}}{{{text}}}")
                    col += span.colspan
                else:
                    parts.append(text)
                    col += 1
            elif (row, col) in covered:
                parts.append("")
                col += 1
            else:
                parts.append(latex_placeholder(row, col, placeholder_mode))
                col += 1
        lines.append(" & ".join(parts) + r" \\")
        if booktabs and row == 0 and n_rows > 1:
            lines.append(r"\midrule")
    if booktabs:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def grid_edges(boxes: list[JsonBox], count: int, low: float, high: float, axis: str) -> list[float]:
    if count <= 0:
        return []
    if len(boxes) != count:
        step = (high - low) / count
        return [low + idx * step for idx in range(count + 1)]
    if axis == "x":
        starts = [box.x0 for box in boxes]
        ends = [box.x1 for box in boxes]
    else:
        starts = [box.y0 for box in boxes]
        ends = [box.y1 for box in boxes]

    edges = [starts[0]]
    for idx in range(count - 1):
        edges.append((ends[idx] + starts[idx + 1]) * 0.5)
    edges.append(ends[-1])
    return edges


def scale_edges(edges: list[float], source_low: float, source_high: float, target_low: float, target_high: float) -> list[float]:
    span = max(1.0, source_high - source_low)
    return [target_low + (edge - source_low) / span * (target_high - target_low) for edge in edges]


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float, float, float], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    if not text:
        return
    x0, y0, x1, y1 = xy
    if x1 - x0 < 24 or y1 - y0 < 14:
        return
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x0 + max(2, (x1 - x0 - tw) / 2), y0 + max(1, (y1 - y0 - th) / 2)), text, font=font, fill=fill)


def render_table_image(
    structure: TableStructure,
    width: int,
    height: int,
    placeholder_mode: str,
    style: str,
    title: str,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22)
    cell_font = load_font(13)
    small_font = load_font(11)
    draw.text((24, 18), title, fill=(22, 22, 22), font=title_font)

    n_rows, n_cols = structure.shape
    if n_rows <= 0 or n_cols <= 0:
        draw.text((24, 70), "empty structure", fill=(160, 0, 0), font=title_font)
        return canvas

    table_box = structure.table_box
    if table_box is None:
        x0 = min((box.x0 for box in structure.cols), default=0.0)
        x1 = max((box.x1 for box in structure.cols), default=float(n_cols))
        y0 = min((box.y0 for box in structure.rows), default=0.0)
        y1 = max((box.y1 for box in structure.rows), default=float(n_rows))
    else:
        x0, y0, x1, y1 = table_box.x0, table_box.y0, table_box.x1, table_box.y1

    max_left, max_top = 32.0, 72.0
    max_right, max_bottom = float(width - 32), float(height - 34)
    max_w = max_right - max_left
    max_h = max_bottom - max_top
    source_aspect = max(0.12, min(8.0, (x1 - x0) / max(1.0, y1 - y0)))
    target_aspect = max_w / max(1.0, max_h)
    if target_aspect > source_aspect:
        table_h = max_h
        table_w = table_h * source_aspect
    else:
        table_w = max_w
        table_h = table_w / source_aspect
    left = max_left + (max_w - table_w) * 0.5
    right = left + table_w
    top = max_top
    bottom = top + table_h
    x_edges = scale_edges(grid_edges(structure.cols, n_cols, x0, x1, "x"), x0, x1, left, right)
    y_edges = scale_edges(grid_edges(structure.rows, n_rows, y0, y1, "y"), y0, y1, top, bottom)

    spans = {(span.row, span.col): span for span in structure.spans}
    covered: set[tuple[int, int]] = set()
    for span in structure.spans:
        for row in range(span.row, min(n_rows, span.row + span.rowspan)):
            for col in range(span.col, min(n_cols, span.col + span.colspan)):
                if (row, col) != (span.row, span.col):
                    covered.add((row, col))

    for row in range(n_rows):
        for col in range(n_cols):
            if (row, col) in covered:
                continue
            span = spans.get((row, col), SpanCell(row, col, 1, 1))
            x_left = x_edges[col]
            x_right = x_edges[min(n_cols, col + span.colspan)]
            y_top = y_edges[row]
            y_bottom = y_edges[min(n_rows, row + span.rowspan)]
            fill = (246, 249, 252) if row == 0 else (255, 255, 255)
            if span.rowspan > 1 or span.colspan > 1:
                fill = (236, 245, 255)
            draw.rectangle((x_left, y_top, x_right, y_bottom), fill=fill)
            text_center(
                draw,
                (x_left, y_top, x_right, y_bottom),
                latex_placeholder(row, col, placeholder_mode),
                cell_font,
                (35, 35, 35),
            )

    line = (30, 30, 30)
    light = (215, 215, 215)
    if style == "grid":
        for x in x_edges:
            draw.line((x, top, x, bottom), fill=light, width=1)
        for y in y_edges:
            draw.line((left, y, right, y), fill=light, width=1)
        draw.rectangle((left, top, right, bottom), outline=line, width=2)
    else:
        draw.line((left, top, right, top), fill=line, width=3)
        if n_rows > 1:
            draw.line((left, y_edges[1], right, y_edges[1]), fill=line, width=2)
        draw.line((left, bottom, right, bottom), fill=line, width=3)
        for y in y_edges[2:-1]:
            draw.line((left, y, right, y), fill=(232, 232, 232), width=1)

    meta = f"{n_rows} rows x {n_cols} cols, {len(structure.spans)} spans"
    draw.text((24, height - 22), meta, fill=(90, 90, 90), font=small_font)
    return canvas


def find_image(image_root: Path | None, image_name: str) -> Path | None:
    if image_root is None:
        return None
    candidate = Path(image_name)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    direct = image_root / candidate.name
    if direct.exists():
        return direct
    stem = candidate.stem or candidate.name
    for suffix in IMAGE_SUFFIXES:
        match = image_root / f"{stem}{suffix}"
        if match.exists():
            return match
    return None


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    image = image.convert("RGB")
    ratio = min(max_width / image.width, max_height / image.height)
    size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return image.resize(size, Image.Resampling.LANCZOS)


def render_comparison(original_path: Path, rendered: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (246, 246, 246))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22)
    left_w = width // 2
    original = fit_image(Image.open(original_path), left_w - 48, height - 90)
    draw.text((24, 18), "Original", fill=(22, 22, 22), font=title_font)
    canvas.paste(original, (24, 72))
    canvas.paste(rendered, (left_w, 0))
    draw.line((left_w - 1, 0, left_w - 1, height), fill=(210, 210, 210), width=2)
    return canvas


def render_record(
    record: dict[str, Any],
    output_dir: Path,
    image_root: Path | None,
    width: int,
    height: int,
    placeholder_mode: str,
    style: str,
) -> dict[str, Any]:
    stem = Path(str(record.get("image", "table"))).stem
    structure = build_structure(record)
    latex = generate_latex(structure, placeholder_mode=placeholder_mode, booktabs=(style == "booktabs"))
    image = render_table_image(
        structure,
        width=width,
        height=height,
        placeholder_mode=placeholder_mode,
        style=style,
        title="LaTeX-style structure render",
    )

    (output_dir / "latex").mkdir(parents=True, exist_ok=True)
    (output_dir / "renders").mkdir(parents=True, exist_ok=True)
    (output_dir / "json").mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "latex" / f"{stem}.tex"
    render_path = output_dir / "renders" / f"{stem}_latex.png"
    tex_path.write_text(latex + "\n", encoding="utf-8")
    image.save(render_path)

    comparison_path = None
    original_path = find_image(image_root, str(record.get("image", "")))
    if original_path is not None:
        (output_dir / "comparisons").mkdir(parents=True, exist_ok=True)
        comparison_path = output_dir / "comparisons" / f"{stem}_comparison.png"
        render_comparison(original_path, image, width=width * 2, height=height).save(comparison_path)

    summary = {
        "image": record.get("image"),
        "shape": list(structure.shape),
        "spans": len(structure.spans),
        "latex": str(tex_path),
        "render": str(render_path),
        "comparison": str(comparison_path) if comparison_path is not None else None,
    }
    (output_dir / "json" / f"{stem}_render.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render standard structure JSON as LaTeX-style table images.")
    parser.add_argument("--structure-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--placeholder-mode", choices=["coords", "blank"], default="coords")
    parser.add_argument("--style", choices=["booktabs", "grid"], default="booktabs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_structure_records(args.structure_json)
    if args.limit is not None and args.limit > 0:
        records = records[: args.limit]
    summaries = [
        render_record(
            record=record,
            output_dir=args.output_dir,
            image_root=args.image_root,
            width=args.width,
            height=args.height,
            placeholder_mode=args.placeholder_mode,
            style=args.style,
        )
        for record in records
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summaries)} LaTeX-style render(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
