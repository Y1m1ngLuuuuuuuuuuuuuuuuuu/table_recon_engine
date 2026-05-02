import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from table_recon_engine.utils.tokenizer import HTMLTokenizer, merge_pubtabnet_tokens


def _read_records(annotation_path: Path) -> list[dict[str, Any]]:
    text = annotation_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if annotation_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "annotations" in data:
        return list(data["annotations"])
    raise ValueError(f"Unsupported annotation format in {annotation_path}")


def _extract_structure_tokens(record: dict[str, Any]) -> list[str]:
    if "tokens" in record:
        return list(record["tokens"])
    html = record.get("html", {})
    structure = html.get("structure", {})
    if "tokens" in structure:
        return list(structure["tokens"])
    if "structure" in record and isinstance(record["structure"], list):
        return list(record["structure"])
    raise KeyError("Cannot find structure tokens. Expected record['html']['structure']['tokens'].")


def _extract_cells(record: dict[str, Any]) -> list[dict[str, Any]]:
    if "cells" in record:
        return list(record["cells"])
    html = record.get("html", {})
    return list(html.get("cells", []))


def _image_name(record: dict[str, Any]) -> str:
    for key in ("filename", "file_name", "image_path", "img_path", "image"):
        if key in record:
            return str(record[key])
    raise KeyError("Cannot find image filename in annotation record.")


def _normalize_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0 = min(max(x0 / width, 0.0), 1.0)
    y0 = min(max(y0 / height, 0.0), 1.0)
    x1 = min(max(x1 / width, 0.0), 1.0)
    y1 = min(max(y1 / height, 0.0), 1.0)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


class PubTabNetDataset(Dataset):
    """Small PubTabNet-style subset loader with normalized bbox targets."""

    def __init__(
        self,
        annotation_path: str | Path,
        image_root: str | Path,
        tokenizer: HTMLTokenizer,
        image_size: tuple[int, int] = (512, 512),
        max_seq_len: int = 512,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.records = _read_records(self.annotation_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        image_path = self.image_root / _image_name(record)
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size
        image_tensor = self._to_tensor(image)

        tokens = merge_pubtabnet_tokens(_extract_structure_tokens(record))
        tokens = tokens[: self.max_seq_len - 2]
        input_ids = self.tokenizer.encode(tokens, add_special=True)[:-1]
        target_ids = self.tokenizer.encode(tokens, add_special=True)[1:]

        boxes, box_mask = self._build_box_targets(
            tokens=tokens,
            cells=_extract_cells(record),
            width=orig_width,
            height=orig_height,
        )

        return {
            "image": image_tensor,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "box_mask": torch.tensor(box_mask, dtype=torch.bool),
        }

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize(self.image_size, Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _build_box_targets(
        self,
        tokens: list[str],
        cells: list[dict[str, Any]],
        width: int,
        height: int,
    ) -> tuple[list[list[float]], list[bool]]:
        boxes = [[0.0, 0.0, 0.0, 0.0] for _ in range(len(tokens) + 1)]
        box_mask = [False for _ in range(len(tokens) + 1)]

        bboxes = [
            _normalize_bbox(cell["bbox"], width, height)
            for cell in cells
            if "bbox" in cell and cell["bbox"] is not None
        ]
        bbox_idx = 0
        for target_pos, token in enumerate(tokens, start=0):
            if HTMLTokenizer.is_cell_token(token) and bbox_idx < len(bboxes):
                boxes[target_pos] = bboxes[bbox_idx]
                box_mask[target_pos] = True
                bbox_idx += 1
        return boxes, box_mask


def collate_tsr_batch(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    images = torch.stack([item["image"] for item in batch], dim=0)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    target_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    boxes = torch.zeros((len(batch), max_len, 4), dtype=torch.float32)
    box_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for row, item in enumerate(batch):
        seq_len = item["input_ids"].numel()
        input_ids[row, :seq_len] = item["input_ids"]
        target_ids[row, :seq_len] = item["target_ids"]
        boxes[row, :seq_len] = item["boxes"]
        box_mask[row, :seq_len] = item["box_mask"]

    return {
        "images": images,
        "input_ids": input_ids,
        "target_ids": target_ids,
        "boxes": boxes,
        "box_mask": box_mask,
    }
