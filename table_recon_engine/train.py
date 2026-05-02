import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from table_recon_engine.custom_ops import giou_loss
from table_recon_engine.models import TSREngine
from table_recon_engine.utils.dataset import PubTabNetDataset, collate_tsr_batch
from table_recon_engine.utils.tokenizer import HTMLTokenizer, default_html_tokens, merge_pubtabnet_tokens


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_tokenizer(annotation_path: Path, vocab_path: Path | None) -> HTMLTokenizer:
    if vocab_path and vocab_path.exists():
        return HTMLTokenizer.load(vocab_path)

    tokenizer = HTMLTokenizer(default_html_tokens())
    text = annotation_path.read_text(encoding="utf-8").strip()
    if text:
        if annotation_path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            records = data if isinstance(data, list) else data.get("annotations", [])
        for record in records:
            html = record.get("html", {})
            structure = html.get("structure", {})
            tokens = record.get("tokens") or structure.get("tokens") or record.get("structure") or []
            tokenizer.add_tokens(merge_pubtabnet_tokens(list(tokens)))

    if vocab_path:
        tokenizer.save(vocab_path)
    return tokenizer


def train_one_epoch(
    model: TSREngine,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss: nn.Module,
    device: torch.device,
    bbox_weight: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_box = 0.0
    steps = 0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_boxes = batch["boxes"].to(device)
        box_mask = batch["box_mask"].to(device)

        outputs = model(images, input_ids)
        logits = outputs["logits"]
        pred_boxes = outputs["boxes"]

        loss_ce = ce_loss(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        if box_mask.any():
            loss_box = giou_loss(pred_boxes[box_mask], target_boxes[box_mask])
        else:
            loss_box = logits.new_tensor(0.0)
        loss = loss_ce + bbox_weight * loss_box

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_ce += float(loss_ce.detach().cpu())
        total_box += float(loss_box.detach().cpu())
        steps += 1

    denom = max(steps, 1)
    return {
        "loss": total_loss / denom,
        "ce_loss": total_ce / denom,
        "giou_loss": total_box / denom,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight TSR Engine.")
    parser.add_argument("--annotations", type=Path, required=True, help="PubTabNet subset json/jsonl.")
    parser.add_argument("--image-root", type=Path, required=True, help="Root folder of table images.")
    parser.add_argument("--vocab-path", type=Path, default=Path("vocab.json"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tsr_engine.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bbox-weight", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"Using device: {device}")

    tokenizer = build_tokenizer(args.annotations, args.vocab_path)
    dataset = PubTabNetDataset(
        annotation_path=args.annotations,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_size=tuple(args.image_size),
        max_seq_len=args.max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_tsr_batch, pad_id=tokenizer.pad_id),
    )

    model = TSREngine(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        pad_token_id=tokenizer.pad_id,
        max_len=args.max_seq_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            ce_loss=ce_loss,
            device=device,
            bbox_weight=args.bbox_weight,
        )
        print(
            f"epoch={epoch:03d} "
            f"loss={metrics['loss']:.4f} "
            f"ce={metrics['ce_loss']:.4f} "
            f"giou={metrics['giou_loss']:.4f}"
        )
        torch.save(
            {
                "model": model.state_dict(),
                "tokenizer": tokenizer.id_to_token,
                "epoch": epoch,
                "metrics": metrics,
                "args": vars(args),
            },
            args.output,
        )


if __name__ == "__main__":
    main()
