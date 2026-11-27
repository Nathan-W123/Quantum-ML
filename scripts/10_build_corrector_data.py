"""Training data for the geometry corrector: GFN2-xTB vs DFT geometries of the same conformer.

Usage:
    python scripts/10_build_corrector_data.py [--n-train 16000 --n-val 1000 --n-test 1500] \
        [--workers 4] [--time-budget-min 100]

For QM9 molecules from the milestone 3 formula split, relax the DFT geometry
with GFN2-xTB (same conformer, atoms in the same order) and store both
geometries with the DFT rotational constants. The corrector is trained on
"train" molecules only, so test formulas stay unseen.

Resumable: results are appended to data/processed/corrector/xtb_relaxed.jsonl;
re-run to continue. When all molecules are done, writes corrector.npz.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rotml.analysis.cheap_methods import xtb_relax  # noqa: E402
from rotml.data.qm9 import RAW_DIR, SDF_NAME, iter_records  # noqa: E402

OUT = Path("data/processed/corrector")
ELEMENT_INDEX = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
MAX_ATOMS = 29


def _work(job: tuple) -> dict:
    mol_id, split, symbols, coords = job
    t = time.perf_counter()
    try:
        g = xtb_relax(symbols, np.asarray(coords))
    except Exception:  # noqa: BLE001
        g = None
    return {"mol_id": mol_id, "split": split, "symbols": symbols, "dft_xyz": coords,
            "xtb_xyz": None if g is None else g.coords.round(6).tolist(),
            "seconds": time.perf_counter() - t}


def pack(rows: list[dict], qm9: pd.DataFrame, out: Path) -> None:
    rows = [r for r in rows if r["xtb_xyz"] is not None]
    n = len(rows)
    z = np.zeros((n, MAX_ATOMS), np.int64)
    dft = np.zeros((n, MAX_ATOMS, 3), np.float32)
    xtb = np.zeros((n, MAX_ATOMS, 3), np.float32)
    for i, r in enumerate(rows):
        k = len(r["symbols"])
        z[i, :k] = [ELEMENT_INDEX[s] for s in r["symbols"]]
        dft[i, :k] = r["dft_xyz"]
        xtb[i, :k] = r["xtb_xyz"]
    ref = qm9.set_index("mol_id").loc[[r["mol_id"] for r in rows], ["A", "B", "C"]].to_numpy()
    np.savez_compressed(out, z=z, dft_xyz=dft, xtb_xyz=xtb, ref_abc=ref.astype(np.float64),
                        split=np.array([r["split"] for r in rows]),
                        mol_id=np.array([r["mol_id"] for r in rows]))
    print(f"wrote {out}: {n} molecules "
          f"({', '.join(f'{s} {sum(r['split'] == s for r in rows)}' for s in ('train', 'val', 'test'))})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget-min", type=float, default=float("inf"))
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "xtb_relaxed.jsonl"

    qm9 = pd.read_csv("data/processed/qm9.csv")
    mols = pd.read_csv("data/processed/m3/m3_molecules.csv").merge(qm9[["mol_id", "sdf_index"]], on="mol_id")
    chosen = pd.concat([mols[mols.split == s].sample(n=n, random_state=args.seed)
                        for s, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test))])
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["mol_id"]] = r
    todo = chosen[~chosen.mol_id.isin(done)]
    print(f"{len(done)} cached, {len(todo)} to do", flush=True)

    if len(todo):
        split_of = dict(zip(todo.sdf_index, zip(todo.mol_id, todo.split)))
        jobs = []
        for i, rec in iter_records(RAW_DIR / SDF_NAME, sorted(int(x) for x in todo.sdf_index)):
            mol_id, split = split_of[i]
            jobs.append((mol_id, split, rec.symbols, rec.coords.round(6).tolist()))
        with mp.get_context("spawn").Pool(args.workers) as pool, cache.open("a") as fh:
            for n, res in enumerate(pool.imap_unordered(_work, jobs, chunksize=8), 1):
                fh.write(json.dumps(res) + "\n")
                done[res["mol_id"]] = res
                if n % 500 == 0:
                    fh.flush()
                    print(f"[{time.time() - t0:.0f}s] {len(done)}/{len(chosen)}", flush=True)
                if time.time() - t0 > args.time_budget_min * 60:
                    pool.terminate()
                    print("time budget reached; re-run to continue", flush=True)
                    return
    rows = [done[m] for m in chosen.mol_id if m in done]
    print(f"xTB failed for {sum(r['xtb_xyz'] is None for r in rows)} molecules", flush=True)
    pack(rows, qm9, OUT / "corrector.npz")


if __name__ == "__main__":
    main()
