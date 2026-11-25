"""Milestone 2: error of cheap geometry methods vs QM9 DFT rotational constants.

Usage:
    OMP_NUM_THREADS=1 python scripts/03_cheap_method_error.py [--n 2000] [--workers 4]

Outputs (results/m2/):
    constants_by_method.csv   A, B, C per molecule and method, plus timings
    error_summary.csv         error statistics and tolerances per method

Figures and the link to milestone 1 are made by 03b_analyze_cheap_methods.py.
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from rotml.analysis.cheap_methods import evaluate_molecule
from rotml.data.qm9 import RAW_DIR, SDF_NAME, iter_sdf
from rotml.physics.rotor import is_linear

DATA = Path("data/processed/qm9.csv")
OUT = Path("results/m2")


def _work(args):
    sdf_index, ref_abc, k = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    (_, mol), = list(iter_sdf(RAW_DIR / SDF_NAME, [int(sdf_index)]))
    return int(sdf_index), evaluate_molecule(mol, np.asarray(ref_abc), k_conformers=k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10, help="conformers per molecule")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qm9 = pd.read_csv(DATA)
    pool_df = qm9[qm9.sanitized & ~is_linear(qm9.A, qm9.B)]
    sample = pool_df.sample(n=args.n, random_state=args.seed)
    jobs = [(r.sdf_index, (r.A, r.B, r.C), args.k) for r in sample.itertuples()]

    t0 = time.time()
    rows = {}
    with Pool(args.workers) as pool:
        for i, (idx, res) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4), 1):
            rows[idx] = res
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} molecules  {time.time() - t0:.0f}s", flush=True)

    res = pd.DataFrame.from_dict(rows, orient="index")
    res.index.name = "sdf_index"
    table = sample.set_index("sdf_index")[["mol_id", "formula", "n_heavy", "smiles", "A", "B", "C"]]
    table = table.join(res)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "constants_by_method.csv")
    print(f"done: {len(table)} molecules in {time.time() - t0:.0f}s "
          f"({(table.status == 'ok').sum()} ok)")


if __name__ == "__main__":
    main()
