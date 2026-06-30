#!/usr/bin/env python3
"""Download embedding training/eval datasets to ./data/ (local disk)."""

from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    all_nli_dir = DATA / "all-nli-pair-class"
    if all_nli_dir.exists():
        print(f"Already exists: {all_nli_dir}")
    else:
        print("Downloading AllNLI (pair-class) — ~139 MB on disk...")
        print("  Source: https://huggingface.co/datasets/sentence-transformers/all-nli")
        ds = load_dataset("sentence-transformers/all-nli", "pair-class")
        ds.save_to_disk(str(all_nli_dir))
        for split in ds:
            print(f"  {split}: {len(ds[split]):,} rows")

    sts_dir = DATA / "stsbenchmark"
    if sts_dir.exists():
        print(f"Already exists: {sts_dir}")
    else:
        print("Downloading STS Benchmark — evaluation set...")
        print("  Source: https://huggingface.co/datasets/mteb/stsbenchmark-sts")
        sts = load_dataset("mteb/stsbenchmark-sts")
        sts.save_to_disk(str(sts_dir))
        for split in sts:
            print(f"  {split}: {len(sts[split]):,} rows")

    print("\nDone. Data saved under:", DATA)


if __name__ == "__main__":
    main()
