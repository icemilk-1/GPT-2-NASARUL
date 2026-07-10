"""Verify C-MAPSS dataset files are present in data/CMaps/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "CMaps"

REQUIRED = [
    "train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt",
    "test_FD001.txt", "test_FD002.txt", "test_FD003.txt", "test_FD004.txt",
    "RUL_FD001.txt", "RUL_FD002.txt", "RUL_FD003.txt", "RUL_FD004.txt",
]


def check() -> bool:
    missing = [f for f in REQUIRED if not (DATA_DIR / f).exists()]
    if missing:
        print("ERROR: Missing C-MAPSS files in data/CMaps/:")
        for f in missing:
            print(f"  - {f}")
        print("\nDownload guide: data/README.md")
        print("NASA: https://www.nasa.gov/intelligent-systems-division/prognostics-center-of-excellence-data-set-repository/")
        return False
    print(f"OK: All {len(REQUIRED)} C-MAPSS files found in {DATA_DIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
