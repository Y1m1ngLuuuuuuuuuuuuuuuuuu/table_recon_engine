from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from table_recon_engine.structure_json import CLASS_TO_ID, normalize_class_name


NODE_FEATURE_DIM = 17
EDGE_FEATURE_DIM = 9


@dataclass(slots=True)
class GridBox:
    x0: float
    y0: float
    x1: float
    y1: float
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

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(slots=True)
class LogicalSpan:
    row: int
    col: int
    rowspan: int
    colspan: int
    score: float = 1.0


@dataclass(slots=True)
class GraphSample:
    image: str
    width: int
    height: int
    rows: list[GridBox]
    cols: list[GridBox]
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_labels: torch.Tensor | None
    edge_keys: list[tuple[int, int, str]]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.cols)


def _box_from_obj(obj: dict[str, Any]) -> GridBox | None:
    raw = obj.get("bbox")
    if not raw or len(raw) != 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in raw]
    score = float(obj.get("confidence", obj.get("score", 1.0)))
    return GridBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), score)


def _objects(record: dict[str, Any], class_name: str) -> list[dict[str, Any]]:
    return [
        obj
        for obj in record.get("objects", [])
        if normalize_class_name(str(obj.get("class", "")), obj.get("class_id")) == class_name
    ]


def _boxes(record: dict[str, Any], class_name: str, axis: str) -> list[GridBox]:
    boxes = [box for obj in _objects(record, class_name) if (box := _box_from_obj(obj)) is not None]
    if axis == "x":
        boxes.sort(key=lambda item: item.cx)
    else:
        boxes.sort(key=lambda item: item.cy)
    return boxes


def _axis_overlap(box: GridBox, other: GridBox, axis: str) -> float:
    if axis == "x":
        start = max(box.x0, other.x0)
        end = min(box.x1, other.x1)
        denom = min(max(box.width, 1.0), max(other.width, 1.0))
    else:
        start = max(box.y0, other.y0)
        end = min(box.y1, other.y1)
        denom = min(max(box.height, 1.0), max(other.height, 1.0))
    return max(0.0, end - start) / denom


def _span_from_obj(
    obj: dict[str, Any],
    rows: list[GridBox],
    cols: list[GridBox],
    overlap_threshold: float,
) -> LogicalSpan | None:
    logical = obj.get("logical_span")
    if isinstance(logical, dict):
        try:
            row = int(logical["row"])
            col = int(logical["col"])
            rowspan = int(logical["rowspan"])
            colspan = int(logical["colspan"])
        except (KeyError, TypeError, ValueError):
            return None
        if row < 0 or col < 0 or rowspan <= 0 or colspan <= 0:
            return None
        if row >= len(rows) or col >= len(cols):
            return None
        return LogicalSpan(
            row=row,
            col=col,
            rowspan=min(rowspan, len(rows) - row),
            colspan=min(colspan, len(cols) - col),
            score=float(obj.get("confidence", obj.get("score", 1.0))),
        )

    box = _box_from_obj(obj)
    if box is None:
        return None
    covered_rows = [idx for idx, row in enumerate(rows) if _axis_overlap(box, row, "y") >= overlap_threshold]
    covered_cols = [idx for idx, col in enumerate(cols) if _axis_overlap(box, col, "x") >= overlap_threshold]
    if not covered_rows or not covered_cols:
        return None
    return LogicalSpan(
        row=min(covered_rows),
        col=min(covered_cols),
        rowspan=max(covered_rows) - min(covered_rows) + 1,
        colspan=max(covered_cols) - min(covered_cols) + 1,
        score=float(obj.get("confidence", obj.get("score", 1.0))),
    )


def extract_logical_spans(
    record: dict[str, Any],
    rows: list[GridBox],
    cols: list[GridBox],
    overlap_threshold: float = 0.60,
) -> list[LogicalSpan]:
    spans: list[LogicalSpan] = []
    seen: set[tuple[int, int, int, int]] = set()
    for obj in sorted(_objects(record, "table spanning cell"), key=lambda item: float(item.get("confidence", item.get("score", 1.0))), reverse=True):
        span = _span_from_obj(obj, rows=rows, cols=cols, overlap_threshold=overlap_threshold)
        if span is None or (span.rowspan == 1 and span.colspan == 1):
            continue
        key = (span.row, span.col, span.rowspan, span.colspan)
        if key in seen:
            continue
        seen.add(key)
        spans.append(span)
    return spans


def span_component_map(n_rows: int, n_cols: int, spans: list[LogicalSpan]) -> list[list[int]]:
    comp = [[-1 for _ in range(n_cols)] for _ in range(n_rows)]
    for idx, span in enumerate(spans):
        row_end = min(n_rows, span.row + span.rowspan)
        col_end = min(n_cols, span.col + span.colspan)
        for row in range(span.row, row_end):
            for col in range(span.col, col_end):
                comp[row][col] = idx
    return comp


def projected_bbox(rows: list[GridBox], cols: list[GridBox], row: int, col: int, rowspan: int, colspan: int) -> list[float]:
    row_items = rows[row : row + rowspan]
    col_items = cols[col : col + colspan]
    return [
        min(item.x0 for item in col_items),
        min(item.y0 for item in row_items),
        max(item.x1 for item in col_items),
        max(item.y1 for item in row_items),
    ]


def _image_density(image: Image.Image | None, box: GridBox) -> tuple[float, float, float]:
    if image is None or box.area <= 0:
        return 0.0, 0.0, 0.0
    x0 = max(0, int(round(box.x0)))
    y0 = max(0, int(round(box.y0)))
    x1 = min(image.width, int(round(box.x1)))
    y1 = min(image.height, int(round(box.y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0, 0.0
    gray = image.crop((x0, y0, x1, y1)).convert("L")
    data = torch.as_tensor(list(gray.getdata()), dtype=torch.float32).view(gray.height, gray.width)
    darkness = (255.0 - data) / 255.0
    ink = float((darkness > 0.18).float().mean().item())
    h_projection = float(darkness.mean(dim=1).max().item()) if darkness.numel() else 0.0
    v_projection = float(darkness.mean(dim=0).max().item()) if darkness.numel() else 0.0
    return ink, h_projection, v_projection


def _load_image(path: Path | None) -> Image.Image | None:
    if path is None or not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except OSError:
        return None


def find_image(image_root: Path | None, image_name: str) -> Path | None:
    if image_root is None:
        return None
    direct = image_root / Path(image_name).name
    if direct.exists():
        return direct
    stem = Path(image_name).stem
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        candidate = image_root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _header_flags(record: dict[str, Any], rows: list[GridBox], cols: list[GridBox]) -> tuple[set[int], set[int]]:
    header_rows: set[int] = set()
    projected_cols: set[int] = set()
    for obj in _objects(record, "table column header"):
        box = _box_from_obj(obj)
        if box is None:
            continue
        header_rows.update(idx for idx, row in enumerate(rows) if _axis_overlap(box, row, "y") >= 0.45)
    for obj in _objects(record, "table projected row header"):
        box = _box_from_obj(obj)
        if box is None:
            continue
        projected_cols.update(idx for idx, col in enumerate(cols) if _axis_overlap(box, col, "x") >= 0.45)
    return header_rows, projected_cols


def _node_features(
    record: dict[str, Any],
    rows: list[GridBox],
    cols: list[GridBox],
    image: Image.Image | None,
) -> torch.Tensor:
    width = max(1.0, float(record.get("width", image.width if image else 1)))
    height = max(1.0, float(record.get("height", image.height if image else 1)))
    header_rows, projected_cols = _header_flags(record, rows, cols)
    n_rows = max(1, len(rows))
    n_cols = max(1, len(cols))
    features: list[list[float]] = []
    for row_idx, row_box in enumerate(rows):
        for col_idx, col_box in enumerate(cols):
            cell = GridBox(col_box.x0, row_box.y0, col_box.x1, row_box.y1)
            ink, h_projection, v_projection = _image_density(image, cell)
            features.append(
                [
                    cell.x0 / width,
                    cell.y0 / height,
                    cell.x1 / width,
                    cell.y1 / height,
                    cell.cx / width,
                    cell.cy / height,
                    cell.width / width,
                    cell.height / height,
                    row_idx / max(n_rows - 1, 1),
                    col_idx / max(n_cols - 1, 1),
                    row_box.height / height,
                    col_box.width / width,
                    float(row_idx in header_rows),
                    float(col_idx in projected_cols),
                    ink,
                    h_projection,
                    v_projection,
                ]
            )
    return torch.tensor(features, dtype=torch.float32)


def _edge_tensors(
    node_features: torch.Tensor,
    rows: list[GridBox],
    cols: list[GridBox],
    comp: list[list[int]] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[tuple[int, int, str]]]:
    n_rows = len(rows)
    n_cols = len(cols)
    edge_index: list[list[int]] = []
    edge_features: list[list[float]] = []
    labels: list[float] = []
    edge_keys: list[tuple[int, int, str]] = []

    def add_edge(row: int, col: int, other_row: int, other_col: int, orientation: str) -> None:
        src = row * n_cols + col
        dst = other_row * n_cols + other_col
        a = node_features[src]
        b = node_features[dst]
        edge_index.append([src, dst])
        edge_features.append(
            [
                1.0 if orientation == "h" else 0.0,
                1.0 if orientation == "v" else 0.0,
                float(b[4] - a[4]),
                float(b[5] - a[5]),
                float(abs(b[6] - a[6])),
                float(abs(b[7] - a[7])),
                float((a[14] + b[14]) * 0.5),
                float(max(a[15], b[15])),
                float(max(a[16], b[16])),
            ]
        )
        edge_keys.append((src, dst, orientation))
        if comp is not None:
            left = comp[row][col]
            right = comp[other_row][other_col]
            labels.append(1.0 if left >= 0 and left == right else 0.0)

    for row in range(n_rows):
        for col in range(n_cols - 1):
            add_edge(row, col, row, col + 1, "h")
    for row in range(n_rows - 1):
        for col in range(n_cols):
            add_edge(row, col, row + 1, col, "v")

    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_feature_tensor = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
        edge_feature_tensor = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32) if comp is not None else None
    return edge_index_tensor, edge_feature_tensor, label_tensor, edge_keys


def build_graph_sample(
    record: dict[str, Any],
    image_root: Path | None = None,
    label_record: dict[str, Any] | None = None,
    span_overlap_threshold: float = 0.60,
    use_visual_features: bool = False,
) -> GraphSample | None:
    rows = _boxes(record, "table row", "y")
    cols = _boxes(record, "table column", "x")
    if not rows or not cols:
        return None

    image_path = find_image(image_root, str(record.get("image", ""))) if use_visual_features else None
    image = _load_image(image_path)
    node_features = _node_features(record, rows, cols, image)
    comp = None
    if label_record is not None:
        label_rows = _boxes(label_record, "table row", "y")
        label_cols = _boxes(label_record, "table column", "x")
        if len(label_rows) == len(rows) and len(label_cols) == len(cols):
            spans = extract_logical_spans(label_record, label_rows, label_cols, overlap_threshold=span_overlap_threshold)
            comp = span_component_map(len(rows), len(cols), spans)

    edge_index, edge_features, edge_labels, edge_keys = _edge_tensors(node_features, rows, cols, comp)
    return GraphSample(
        image=str(record.get("image", "")),
        width=int(record.get("width", image.width if image else 0)),
        height=int(record.get("height", image.height if image else 0)),
        rows=rows,
        cols=cols,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_labels=edge_labels,
        edge_keys=edge_keys,
    )


def spans_from_merge_logits(
    sample: GraphSample,
    logits: torch.Tensor,
    threshold: float = 0.50,
    min_span_area: int = 2,
) -> list[LogicalSpan]:
    n_rows, n_cols = sample.shape
    if n_rows <= 0 or n_cols <= 0:
        return []
    parent = list(range(n_rows * n_cols))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    probs = torch.sigmoid(logits.detach().cpu()).flatten()
    for edge_idx, prob in enumerate(probs):
        if float(prob) < threshold:
            continue
        src, dst, _orientation = sample.edge_keys[edge_idx]
        union(src, dst)

    groups: dict[int, list[int]] = {}
    for idx in range(n_rows * n_cols):
        groups.setdefault(find(idx), []).append(idx)

    spans: list[LogicalSpan] = []
    occupied: set[tuple[int, int]] = set()
    for cells in sorted(groups.values(), key=lambda items: min(items)):
        if len(cells) < min_span_area:
            continue
        rows = [cell // n_cols for cell in cells]
        cols = [cell % n_cols for cell in cells]
        row0, row1 = min(rows), max(rows)
        col0, col1 = min(cols), max(cols)
        area = (row1 - row0 + 1) * (col1 - col0 + 1)
        if area != len(cells):
            continue
        coords = {(row, col) for row in range(row0, row1 + 1) for col in range(col0, col1 + 1)}
        if coords & occupied:
            continue
        occupied.update(coords)
        spans.append(LogicalSpan(row=row0, col=col0, rowspan=row1 - row0 + 1, colspan=col1 - col0 + 1))
    return spans


def merge_spans_into_record(
    record: dict[str, Any],
    sample: GraphSample,
    spans: list[LogicalSpan],
    keep_existing_spans: bool = False,
) -> dict[str, Any]:
    objects = []
    for obj in record.get("objects", []):
        class_name = normalize_class_name(str(obj.get("class", "")), obj.get("class_id"))
        if class_name != "table spanning cell" or keep_existing_spans:
            objects.append(obj)
    for span in spans:
        box = projected_bbox(sample.rows, sample.cols, span.row, span.col, span.rowspan, span.colspan)
        objects.append(
            {
                "class": "table spanning cell",
                "class_id": CLASS_TO_ID["table spanning cell"],
                "bbox": box,
                "projected_bbox": box,
                "confidence": span.score,
                "logical_span": {
                    "row": span.row,
                    "col": span.col,
                    "rowspan": span.rowspan,
                    "colspan": span.colspan,
                },
            }
        )
    output = dict(record)
    output["objects"] = objects
    return output
