#!/usr/bin/env python3
"""
extract_seq00.py
----------------
Extracts ONLY sequence 00 from data_odometry_velodyne.zip into:
  data/sequences/00/velodyne/*.bin

Then deletes the original zip to reclaim ~72 GB of disk space.
"""

import zipfile
import os
import sys
from pathlib import Path

ZIP_PATH   = Path(r"C:\Users\Rohan\Downloads\data_odometry_velodyne.zip")
DEST_DIR   = Path("data/sequences/00/velodyne")
SEQ_PREFIX = "dataset/sequences/00/velodyne/"   # path inside the zip

def main():
    if not ZIP_PATH.exists():
        print(f"[ERROR] Zip not found at: {ZIP_PATH}")
        sys.exit(1)

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] Scanning zip for sequence 00 entries...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        all_names = zf.namelist()

        # Find all .bin files belonging to sequence 00
        seq00_files = [
            n for n in all_names
            if n.startswith(SEQ_PREFIX) and n.endswith(".bin")
        ]

        if not seq00_files:
            # Try alternate path structure (some downloads vary)
            alt_prefix = "sequences/00/velodyne/"
            seq00_files = [
                n for n in all_names
                if n.startswith(alt_prefix) and n.endswith(".bin")
            ]
            if seq00_files:
                global SEQ_PREFIX
                SEQ_PREFIX = alt_prefix

        if not seq00_files:
            print("[ERROR] Could not find sequence 00 entries in the zip!")
            print("  Top-level entries found:")
            for n in all_names[:20]:
                print(f"    {n}")
            sys.exit(1)

        total = len(seq00_files)
        print(f"[1/3] Found {total:,} scan files in sequence 00.")
        print(f"[2/3] Extracting to: {DEST_DIR.resolve()}")

        for i, name in enumerate(seq00_files):
            # Strip the prefix so we just get the filename (e.g., 000000.bin)
            filename = Path(name).name
            dest_file = DEST_DIR / filename

            if dest_file.exists():
                # Skip already extracted files (allows resuming)
                continue

            with zf.open(name) as src, open(dest_file, "wb") as dst:
                dst.write(src.read())

            if (i + 1) % 500 == 0 or (i + 1) == total:
                pct = 100.0 * (i + 1) / total
                print(f"  {i+1:,}/{total:,}  ({pct:.1f}%)", end="\r")

    print(f"\n  ✓ All {total:,} scans extracted.")

    # Verify count
    extracted = list(DEST_DIR.glob("*.bin"))
    print(f"  ✓ Files on disk: {len(extracted):,}")

    print(f"\n[3/3] Deleting zip to free disk space...")
    try:
        ZIP_PATH.unlink()
        print(f"  ✓ Deleted: {ZIP_PATH}")
    except Exception as e:
        print(f"  [WARN] Could not delete zip automatically: {e}")
        print(f"         You can manually delete: {ZIP_PATH}")

    print(f"\n{'='*55}")
    print(f"  Done! {len(extracted):,} LiDAR scans ready in:")
    print(f"  {DEST_DIR.resolve()}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
