"""
save_wikitext_local.py — Download WikiText-103-raw-v1 and save to disk.

Run once locally:
    python save_wikitext_local.py

Output: wikitext_local/  (train/ + validation/ in Arrow format)

Then zip and upload to Kaggle:
    zip -r wikitext_local.zip wikitext_local/
Upload as a Kaggle dataset named: wikitext-local
It will mount at: /kaggle/input/wikitext-local/wikitext_local/
"""

import pathlib
from datasets import load_dataset

OUT = pathlib.Path("wikitext_local")
OUT.mkdir(exist_ok=True)

for split in ["train", "validation"]:
    dst = OUT / split
    if dst.exists():
        print(f"{split}: already exists at {dst}, skipping")
        continue
    print(f"Downloading {split} ...", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    print(f"  {len(ds):,} rows — saving to {dst} ...", flush=True)
    ds.save_to_disk(str(dst))
    print(f"  saved.")

print("\nDone. Zip and upload to Kaggle:")
print("  Windows (PowerShell): Compress-Archive wikitext_local wikitext_local.zip")
print("  Linux/Mac:            zip -r wikitext_local.zip wikitext_local/")
print("  Kaggle dataset name:  wikitext-local")
