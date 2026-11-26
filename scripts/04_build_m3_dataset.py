"""Build the milestone 3 dataset (observables -> stereo-free SELFIES).

Usage:
    python scripts/04_build_m3_dataset.py [--split formula|random] [--variants 5]

Needs data/processed/qm9.csv from 01_prepare_qm9.py. Writes data/processed/m3/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rotml.ml.dataset import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qm9", type=Path, default=Path("data/processed/qm9.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/m3"))
    ap.add_argument("--split", choices=["formula", "random"], default="formula")
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    meta = build_dataset(args.qm9, args.out, n_variants=args.variants, split=args.split,
                         seed=args.seed)
    print(json.dumps({k: meta[k] for k in ("sizes", "n_formulas", "max_len")}, indent=1))
    print(f"vocab size {len(meta['vocab'])}; wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
