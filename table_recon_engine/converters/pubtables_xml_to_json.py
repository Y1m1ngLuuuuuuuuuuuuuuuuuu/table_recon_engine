from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from table_recon_engine.structure_json import pubtables_xml_to_record, write_structure_records


def read_filelist(extracted_dir: Path, split: str) -> list[str]:
    filelist = extracted_dir / f"{split}_filelist.txt"
    return [Path(line.strip()).name for line in filelist.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert_split(
    extracted_dir: Path,
    output_dir: Path,
    split: str,
    max_samples: int | None,
    seed: int,
    jsonl: bool,
) -> dict[str, int | str]:
    xml_names = read_filelist(extracted_dir, split)
    rng = random.Random(seed)
    rng.shuffle(xml_names)
    if max_samples is not None:
        xml_names = xml_names[:max_samples]

    records = []
    skipped = 0
    objects = 0
    for xml_name in xml_names:
        xml_path = extracted_dir / xml_name
        if not xml_path.exists():
            skipped += 1
            continue
        try:
            record = pubtables_xml_to_record(xml_path)
        except (OSError, ValueError):
            skipped += 1
            continue
        records.append(record)
        objects += len(record["objects"])

    suffix = ".jsonl" if jsonl else ".json"
    output_path = output_dir / "annotations" / f"{split}{suffix}"
    write_structure_records(output_path, records, jsonl=jsonl)
    return {
        "split": split,
        "records": len(records),
        "objects": objects,
        "skipped": skipped,
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PubTables-1M Structure XML to standard structure JSON.")
    parser.add_argument("--extracted-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Write pretty JSON instead of compact JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = not args.json
    train = convert_split(
        extracted_dir=args.extracted_dir,
        output_dir=args.output_dir,
        split="train",
        max_samples=args.train_samples,
        seed=args.seed,
        jsonl=jsonl,
    )
    val = convert_split(
        extracted_dir=args.extracted_dir,
        output_dir=args.output_dir,
        split="val",
        max_samples=args.val_samples,
        seed=args.seed + 1,
        jsonl=jsonl,
    )
    manifest = {"train": train, "val": val}
    manifest_path = args.output_dir / "annotations" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
