#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revert the Rowmax-H15 patch (restore <file>.orig backups, remove pot_h15.py).
Usage: python revert_patch.py --fa4-root /path/to/flash-attention"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_patch import do_revert, CUTE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa4-root", required=True)
    a = ap.parse_args()
    root = Path(a.fa4_root).resolve()
    if not (root / CUTE).is_dir():
        print(f"ERROR: {root}/{CUTE} not found")
        return 1
    return do_revert(root)


if __name__ == "__main__":
    sys.exit(main())
