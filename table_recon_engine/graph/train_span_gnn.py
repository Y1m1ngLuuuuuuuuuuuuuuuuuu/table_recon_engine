from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from table_recon_engine.graph.grid_graph import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GraphSample,
    build_graph_sample,
)
from table_recon_engine.graph.span_gnn import SpanEdgeClassifier
from table_recon_engine.structure_json import image_key, load_structure_records


class SpanGraphDataset(Dataset[GraphSample]):
    def __init__(
        self,
        input_json: Path,
        label_json: Path,
        image_root: Path | None,
        max_samples: int | None = None,
        seed: int = 42,
        use_visual_features: bool = False,
    ) -> None:
        input_records = load_structure_records(input_json)
        label_records = {image_key(record): record for record in load_structure_records(label_json)}
        if max_samples is not None and max_samples > 0:
            rng = random.Random(seed)
            rng.shuffle(input_records)
            input_records = input_records[:max_samples]

        self.samples: list[GraphSample] = []
        skipped = 0
        for record in input_records:
            label_record = label_records.get(image_key(record))
            if label_record is None:
                skipped += 1
                continue
            sample = build_graph_sample(
                record,
                image_root=image_root,
                label_record=label_record,
                use_visual_features=use_visual_features,
            )
            if sample is None or sample.edge_labels is None or sample.edge_labels.numel() == 0:
                skipped += 1
                continue
            self.samples.append(sample)
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GraphSample:
        return self.samples[index]


def collate_graphs(samples: list[GraphSample]) -> dict[str, torch.Tensor]:
    node_features = []
    edge_features = []
    edge_labels = []
    edge_indexes = []
    node_offset = 0
    for sample in samples:
        node_features.append(sample.node_features)
        edge_features.append(sample.edge_features)
        if sample.edge_labels is None:
            raise ValueError("Training samples must include edge labels.")
        edge_labels.append(sample.edge_labels)
        edge_indexes.append(sample.edge_index + node_offset)
        node_offset += sample.node_features.shape[0]
    return {
        "node_features": torch.cat(node_features, dim=0),
        "edge_features": torch.cat(edge_features, dim=0),
        "edge_labels": torch.cat(edge_labels, dim=0),
        "edge_index": torch.cat(edge_indexes, dim=1),
    }


def select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def edge_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.50) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    truth = labels >= 0.5
    tp = int((preds & truth).sum().item())
    fp = int((preds & ~truth).sum().item())
    fn = int((~preds & truth).sum().item())
    tn = int((~preds & ~truth).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def merge_metrics(total: dict[str, float], part: dict[str, float]) -> None:
    for key in ("tp", "fp", "fn", "tn"):
        total[key] = total.get(key, 0.0) + part[key]


def finalize_metrics(total: dict[str, float]) -> dict[str, float]:
    tp = int(total.get("tp", 0))
    fp = int(total.get("fp", 0))
    fn = int(total.get("fn", 0))
    tn = int(total.get("tn", 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def run_epoch(
    model: SpanEdgeClassifier,
    loader: DataLoader[dict[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_edges = 0
    totals: dict[str, float] = {}
    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        batch = move_batch(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(batch["node_features"], batch["edge_index"], batch["edge_features"])
            loss = criterion(logits, batch["edge_labels"])
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        count = int(batch["edge_labels"].numel())
        total_loss += float(loss.item()) * count
        total_edges += count
        merge_metrics(totals, edge_metrics(logits.detach().cpu(), batch["edge_labels"].detach().cpu()))
        progress.set_postfix(loss=total_loss / max(total_edges, 1))
    metrics = finalize_metrics(totals)
    metrics["loss"] = total_loss / max(total_edges, 1)
    metrics["edges"] = total_edges
    return metrics


def positive_weight(dataset: SpanGraphDataset) -> torch.Tensor:
    positives = 0.0
    negatives = 0.0
    for sample in dataset.samples:
        labels = sample.edge_labels
        if labels is None:
            continue
        positives += float(labels.sum().item())
        negatives += float((labels.numel() - labels.sum()).item())
    return torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)


def save_checkpoint(
    path: Path,
    model: SpanEdgeClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
            "model_config": {
                "node_dim": NODE_FEATURE_DIM,
                "edge_dim": EDGE_FEATURE_DIM,
                "hidden_dim": args.hidden_dim,
                "message_layers": args.message_layers,
                "dropout": args.dropout,
            },
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the second-layer spanning-cell graph edge classifier.")
    parser.add_argument("--train-input-json", type=Path, required=True, help="Base-grid JSON used as model input.")
    parser.add_argument("--train-label-json", type=Path, required=True, help="Ground-truth structure JSON used for edge labels.")
    parser.add_argument("--val-input-json", type=Path, required=True)
    parser.add_argument("--val-label-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--use-visual-features", action="store_true", help="Add per-cell ink-density features from images.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_data = SpanGraphDataset(
        input_json=args.train_input_json,
        label_json=args.train_label_json,
        image_root=args.image_root,
        max_samples=args.max_train_samples,
        seed=args.seed,
        use_visual_features=args.use_visual_features,
    )
    val_data = SpanGraphDataset(
        input_json=args.val_input_json,
        label_json=args.val_label_json,
        image_root=args.image_root,
        max_samples=args.max_val_samples,
        seed=args.seed + 1,
        use_visual_features=args.use_visual_features,
    )
    if len(train_data) == 0 or len(val_data) == 0:
        raise RuntimeError("No usable graph samples. Check row/column detections and label JSON paths.")

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_graphs)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_graphs)
    model = SpanEdgeClassifier(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        message_layers=args.message_layers,
        dropout=args.dropout,
    ).to(device)
    pos_weight = positive_weight(train_data).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "device": str(device),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_skipped": train_data.skipped,
        "val_skipped": val_data.skipped,
        "pos_weight": float(pos_weight.item()),
        "use_visual_features": args.use_visual_features,
        "epochs": [],
    }
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, desc=f"train {epoch}/{args.epochs}")
        val_metrics = run_epoch(model, val_loader, criterion, device, None, desc=f"val {epoch}/{args.epochs}")
        epoch_record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history["epochs"].append(epoch_record)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_p={val_metrics['precision']:.4f} val_r={val_metrics['recall']:.4f}"
        )
        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, epoch_record, args)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, epoch_record, args)

    (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"best_f1={best_f1:.4f} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
