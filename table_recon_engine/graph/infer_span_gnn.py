from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from table_recon_engine.graph.grid_graph import build_graph_sample, merge_spans_into_record, spans_from_merge_logits
from table_recon_engine.graph.span_gnn import build_model_from_checkpoint
from table_recon_engine.graph.train_span_gnn import select_device
from table_recon_engine.structure_json import load_structure_records, write_structure_records


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()
    return model


def infer_record(
    record: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    image_root: Path | None,
    threshold: float,
    keep_existing_spans: bool,
    use_visual_features: bool,
) -> tuple[dict[str, Any], int]:
    sample = build_graph_sample(
        record,
        image_root=image_root,
        label_record=None,
        use_visual_features=use_visual_features,
    )
    if sample is None or sample.edge_index.numel() == 0:
        return record, 0
    with torch.no_grad():
        logits = model(
            sample.node_features.to(device),
            sample.edge_index.to(device),
            sample.edge_features.to(device),
        )
    spans = spans_from_merge_logits(sample, logits, threshold=threshold)
    return merge_spans_into_record(record, sample, spans, keep_existing_spans=keep_existing_spans), len(spans)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer spanning cells from a base-grid structure JSON with a graph edge classifier.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--use-visual-features", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--keep-existing-spans", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--pretty-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model = load_model(args.checkpoint, device)
    records = load_structure_records(args.input_json)
    outputs = []
    span_count = 0
    skipped = 0
    for record in tqdm(records, desc="span graph infer"):
        output, spans = infer_record(
            record=record,
            model=model,
            device=device,
            image_root=args.image_root,
            threshold=args.threshold,
            keep_existing_spans=args.keep_existing_spans,
            use_visual_features=args.use_visual_features,
        )
        outputs.append(output)
        span_count += spans
        if spans == 0:
            skipped += 1
    write_jsonl = args.output_json.suffix.lower() == ".jsonl" and not args.pretty_json
    write_structure_records(args.output_json, outputs, jsonl=write_jsonl)
    manifest = {
        "input_json": str(args.input_json),
        "output_json": str(args.output_json),
        "checkpoint": str(args.checkpoint),
        "image_root": str(args.image_root) if args.image_root else None,
        "threshold": args.threshold,
        "keep_existing_spans": args.keep_existing_spans,
        "records": len(outputs),
        "predicted_spans": span_count,
        "records_without_spans": skipped,
        "device": str(device),
        "use_visual_features": args.use_visual_features,
    }
    manifest_path = args.output_json.with_suffix(args.output_json.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(outputs)} record(s) with {span_count} graph span(s) to {args.output_json}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
