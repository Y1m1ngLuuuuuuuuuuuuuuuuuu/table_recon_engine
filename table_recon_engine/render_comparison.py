from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    image = image.convert("RGB")
    ratio = min(max_width / image.width, max_height / image.height)
    size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return image.resize(size, Image.Resampling.LANCZOS)


def cell_text(row: int, col: int, mode: str) -> str:
    return "" if mode == "blank" else f"r{row + 1}c{col + 1}"


def render_table(structure: dict, width: int, height: int, placeholder_mode: str) -> Image.Image:
    rows, cols = structure.get("shape", [0, 0])
    rows = int(rows)
    cols = int(cols)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    cell_font = load_font(14)

    draw.text((24, 16), "LaTeX-style reconstruction", fill=(20, 20, 20), font=title_font)
    if rows <= 0 or cols <= 0:
        draw.text((24, 70), "empty structure", fill=(160, 0, 0), font=title_font)
        return canvas

    left, top = 24, 64
    right, bottom = width - 24, height - 24
    cell_w = (right - left) / cols
    cell_h = min(36.0, max(18.0, (bottom - top) / rows))
    table_h = cell_h * rows
    if table_h > bottom - top:
        cell_h = (bottom - top) / rows
        table_h = bottom - top

    spans = {
        (int(span["row"]), int(span["col"])): (
            max(1, int(span["rowspan"])),
            max(1, int(span["colspan"])),
        )
        for span in structure.get("spans", [])
    }
    covered: set[tuple[int, int]] = set()
    for (row, col), (rowspan, colspan) in spans.items():
        for rr in range(row, min(rows, row + rowspan)):
            for cc in range(col, min(cols, col + colspan)):
                if (rr, cc) != (row, col):
                    covered.add((rr, cc))

    for row in range(rows):
        for col in range(cols):
            if (row, col) in covered:
                continue
            rowspan, colspan = spans.get((row, col), (1, 1))
            x0 = left + col * cell_w
            y0 = top + row * cell_h
            x1 = left + min(cols, col + colspan) * cell_w
            y1 = top + min(rows, row + rowspan) * cell_h
            fill = (248, 248, 248) if (row + col) % 2 else (255, 255, 255)
            outline = (35, 35, 35)
            if rowspan > 1 or colspan > 1:
                fill = (235, 243, 255)
                outline = (30, 90, 180)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=2 if fill != (255, 255, 255) else 1)
            text = cell_text(row, col, placeholder_mode)
            if text and cell_w >= 28 and cell_h >= 16:
                bbox = draw.textbbox((0, 0), text, font=cell_font)
                tx = x0 + max(2, (x1 - x0 - (bbox[2] - bbox[0])) / 2)
                ty = y0 + max(1, (y1 - y0 - (bbox[3] - bbox[1])) / 2)
                draw.text((tx, ty), text, fill=(30, 30, 30), font=cell_font)

    return canvas


def make_comparison(record_path: Path, output_path: Path, width: int, height: int, placeholder_mode: str) -> None:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    original = fit_image(Image.open(record["image"]), width // 2 - 36, height - 72)
    rendered = render_table(record, width // 2 - 24, height, placeholder_mode)

    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    draw.text((24, 16), "Original table image", fill=(20, 20, 20), font=title_font)
    canvas.paste(original, (24, 64))
    canvas.paste(rendered, (width // 2 + 12, 0))
    draw.line((width // 2, 0, width // 2, height), fill=(210, 210, 210), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create original-vs-LaTeX-style comparison images.")
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument("--placeholder-mode", choices=["coords", "blank"], default="coords")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = sorted((args.demo_dir / "json").glob("*.json"))[: args.limit]
    for record in records:
        make_comparison(
            record_path=record,
            output_path=args.output_dir / f"{record.stem}_comparison.jpg",
            width=args.width,
            height=args.height,
            placeholder_mode=args.placeholder_mode,
        )
    print(f"Wrote {len(records)} comparison image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
