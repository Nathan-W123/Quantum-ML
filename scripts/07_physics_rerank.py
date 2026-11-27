"""Re-rank the inverse model's candidates with physics (first milestone 5 prototype).

Usage:
    python scripts/07_physics_rerank.py --run runs/m3_cpu_medium --out runs/m3_rerank --n 200 \
        [--top 30] [--workers 4] [--time-budget-min 100]

Stages, each resumable (re-run the same command to continue):

1. candidates.json - model candidates (beam 10 + 100 samples) for n test
   molecules, the first n of the molecules 06_eval_m3.py evaluates, with
   0.5% noise on A, B, C and the formula given.
2. physics.jsonl - A, B, C of the conformers of every distinct candidate in
   each molecule's top --top model candidates (stereo-
   isomers -> ETKDG -> MMFF94 -> GFN2-xTB), cached by SMILES and computed in
   parallel. Stops cleanly at the time budget.
3. summary.csv, per_molecule.csv - recall@K for several rankings of the same
   candidate lists:
     model            the model's log-probability alone
     xtb fit          GFN2-xTB fit to the observed constants alone
     model + xtb      log-probability + Student-t log-likelihood of the xTB fit
     mmff fit / model + mmff   the same with MMFF94 (100x cheaper, 2x less accurate)
     gnn fit / model + gnn     the same with GFN2-xTB geometries corrected toward DFT by the
                               geometry corrector (--corrector; needs --keep-geometries)
     DFT lookup (oracle)       candidates that are QM9 molecules, ranked by their
                               DFT constants: what exact physics with a known
                               conformer would do. Candidates outside QM9 rank last.
   All rankings re-order the same top --top list; "in candidates" (the true
   structure is in that list) is the ceiling for every re-ranking.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per xTB worker

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rotml.analysis.rerank import (XTB_BIAS, XTB_SIGMA, candidate_constants,  # noqa: E402
                                   fit_sq_error, physics_loglik)

MMFF_SIGMA = 0.027  # robust per-constant spread of MMFF94 vs B3LYP, searched conformer (milestone 2)
KS = (1, 5, 10, 30)


def stage_candidates(args, path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text())
    import torch

    from rotml.ml import features as F
    from rotml.ml.dataset import M3Data
    from rotml.ml.decode import generate_candidates
    from rotml.ml.model import ModelConfig, Seq2Seq

    torch.set_num_threads(os.cpu_count() or 1)
    ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    data = M3Data.load(Path(ckpt["train_config"]["data_dir"]))
    vocab = data.vocab
    model = Seq2Seq(ModelConfig(**ckpt["model_config"]), pad_id=vocab.pad)
    model.load_state_dict(ckpt["model"])
    model.eval()

    idx = data.indices("test")
    idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))][: args.n]
    noise_t = torch.full((len(idx),), args.noise, dtype=torch.float64)
    g = torch.Generator().manual_seed(int(args.noise * 1e6))
    abc, mu = F.perturb(data.abc[idx], data.mu[idx], noise_t, g)
    present = F.condition_mask(len(idx), True, False)
    values, present = data.standardize(abc, mu, data.counts[idx], noise_t, present)
    formulas = pd.read_csv(Path(ckpt["train_config"]["data_dir"]) / "m3_molecules.csv").formula
    ranked = []
    t0 = time.time()
    for s in range(0, len(idx), 32):
        ranked += generate_candidates(model, values[s:s + 32], present[s:s + 32], vocab, beam=10,
                                      n_samples=args.samples, generator=torch.Generator().manual_seed(s))
        print(f"[{time.time() - t0:.0f}s] candidates for {len(ranked)}/{len(idx)} molecules", flush=True)
    rows = [{"mol_index": int(i), "target": data.smiles[int(i)], "formula": formulas[int(i)],
             "obs_abc": abc[k].tolist(), "true_abc": data.abc[int(i)].tolist(),
             "candidates": [[smi, float(lp)] for smi, lp in ranked[k]]}
            for k, i in enumerate(idx.tolist())]
    path.write_text(json.dumps(rows))
    return rows


def _work(job: tuple) -> dict:
    smiles, keep_geometries, *rest = job
    adaptive = bool(rest and rest[0])
    try:
        return candidate_constants(smiles, keep_geometries=keep_geometries, adaptive=adaptive)
    except Exception as exc:  # noqa: BLE001 - keep the pool alive on odd molecules
        return {"smiles": smiles, "status": f"error: {type(exc).__name__}", "mmff": [], "xtb": []}


def stage_physics(rows: list[dict], path: Path, workers: int, budget_s: float, t_start: float,
                  top: int, keep_geometries: bool = False, adaptive: bool = False) -> bool:
    done = set()
    if path.exists():
        done = {json.loads(line)["smiles"] for line in path.read_text().splitlines() if line.strip()}
    todo = sorted({smi for r in rows for smi, _ in r["candidates"][:top]} - done)
    total = len(done) + len(todo)
    print(f"physics: {len(done)}/{total} distinct candidates cached, {len(todo)} to do", flush=True)
    if not todo:
        return True
    with mp.get_context("spawn").Pool(workers) as pool, path.open("a") as fh:
        jobs = [(smi, keep_geometries, adaptive) for smi in todo]
        for n, res in enumerate(pool.imap_unordered(_work, jobs, chunksize=1), 1):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if n % 200 == 0 or n == len(todo):
                print(f"[{time.time() - t_start:.0f}s] physics {len(done) + n}/{total}", flush=True)
            if time.time() - t_start > budget_s:
                pool.terminate()
                print(f"time budget reached after {len(done) + n}/{total}; re-run to continue", flush=True)
                return False
    return True


def rank_of(target: str, smiles: list[str], score: np.ndarray) -> float:
    """1-based rank of target under descending score (stable: ties keep model order)."""
    if target not in smiles:
        return math.inf
    order = np.argsort(-score, kind="stable")
    return float(np.where(np.array(smiles)[order] == target)[0][0] + 1)


def corrected_constants(phys: dict, corrector_path: Path) -> tuple[dict, float]:
    """GNN-corrected A, B, C for every stored xTB geometry, and the scoring scale sigma.

    sigma is fixed from validation data, not test results: the corrector's
    same-conformer validation spread, times 1.28 (the ratio of searched- to
    same-conformer spread for GFN2-xTB in milestone 2).
    """
    import torch

    from rotml.ml.corrector import Z_OF_SYMBOL, GeometryCorrector, log_constants, masses

    ckpt = torch.load(corrector_path, map_location="cpu", weights_only=False)
    model = GeometryCorrector(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    metrics = pd.read_csv(corrector_path.parent / "metrics.csv")
    val_sd = metrics.query("split == 'val' and method == 'GNN-corrected'").robust_sd_per_constant_pct.item() / 100
    keys, zs, xs = [], [], []
    for smi, r in phys.items():
        if any(a not in Z_OF_SYMBOL for a in r.get("symbols", [])):
            continue  # element the corrector was not trained on (e.g. P): no corrected constants
        for k, xyz in enumerate(r.get("xtb_xyz", [])):
            keys.append((smi, k))
            zs.append([Z_OF_SYMBOL[a] for a in r["symbols"]])
            xs.append(xyz)
    out: dict[str, list] = {}
    with torch.no_grad():
        for s in range(0, len(keys), 512):
            n = max(len(z) for z in zs[s:s + 512])
            z = torch.zeros(len(zs[s:s + 512]), n, dtype=torch.long)
            x = torch.zeros(len(zs[s:s + 512]), n, 3)
            for i, (zi, xi) in enumerate(zip(zs[s:s + 512], xs[s:s + 512])):
                z[i, :len(zi)] = torch.tensor(zi)
                x[i, :len(zi)] = torch.tensor(xi)
            abc = torch.exp(log_constants(model(z, x), masses(z))).numpy()
            for (smi, _), a in zip(keys[s:s + 512], abc):
                out.setdefault(smi, []).append(a.tolist())
    return out, 1.28 * val_sd


def stage_rank(rows: list[dict], physics_path: Path, data_dir: Path, noise: float, out: Path,
               top: int, corrector: Path | None = None, summary_name: str = "summary.csv") -> None:
    phys = {}
    for line in physics_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            phys[r["smiles"]] = r
    gnn, sig_gnn = corrected_constants(phys, corrector) if corrector else ({}, math.nan)
    if corrector:
        print(f"corrected {sum(map(len, gnn.values()))} geometries; GNN scoring sigma "
              f"{100 * sig_gnn:.2f}% per constant", flush=True)
        sig_gnn = math.hypot(sig_gnn, noise)
    # DFT constants of every QM9 molecule in the dataset, by stereo-free canonical SMILES.
    mols = pd.read_csv(data_dir / "m3_molecules.csv")
    abc_all = np.load(data_dir / "m3_dataset.npz")["abc"]
    dft = {}
    for smi, abc in zip(mols.smiles, abc_all):
        dft.setdefault(smi, []).append(abc)

    sig_xtb = math.hypot(XTB_SIGMA, noise)
    sig_mmff = math.hypot(MMFF_SIGMA, noise)
    per_mol, stats = [], {"candidates": [], "xtb_ok": [], "in_qm9": [], "seconds": []}
    for r in rows:
        smiles = [c[0] for c in r["candidates"][:top]]
        logp = np.array([c[1] for c in r["candidates"][:top]])
        obs = np.array(r["obs_abc"])
        xtb_err = np.array([fit_sq_error(phys[s]["xtb"], obs, XTB_BIAS) for s in smiles])
        mmff_err = np.array([fit_sq_error(phys[s]["mmff"], obs) for s in smiles])
        dft_err = np.array([fit_sq_error(dft[s], obs) if s in dft else math.inf for s in smiles])
        xtb_ll = np.array([physics_loglik(e, sig_xtb) for e in xtb_err])
        mmff_ll = np.array([physics_loglik(e, sig_mmff) for e in mmff_err])
        scores = {
            "model": logp,
            "xtb fit": -xtb_err,
            "model + xtb": logp + xtb_ll,
            "mmff fit": -mmff_err,
            "model + mmff": logp + mmff_ll,
        }
        if corrector:
            gnn_err = np.array([fit_sq_error(gnn.get(s, []), obs) for s in smiles])
            scores["gnn fit"] = -gnn_err
            scores["model + gnn"] = logp + np.array([physics_loglik(e, sig_gnn) for e in gnn_err])
        scores["DFT lookup (oracle)"] = -dft_err
        t = r["target"]
        row = {"mol_index": r["mol_index"], "target": t, "formula": r["formula"],
               "n_candidates": len(smiles), "in_candidates": t in smiles}
        if t in smiles:
            k = smiles.index(t)
            row["true_xtb_rms_err_pct"] = 100 * math.sqrt(xtb_err[k] / 3) if math.isfinite(xtb_err[k]) else math.inf
            row["true_xtb_status"] = phys[t]["status"]
            if corrector:
                row["true_gnn_rms_err_pct"] = (100 * math.sqrt(gnn_err[k] / 3) if math.isfinite(gnn_err[k])
                                               else math.inf)
        row |= {f"rank {name}": rank_of(t, smiles, s) for name, s in scores.items()}
        per_mol.append(row)
        stats["candidates"].append(len(smiles))
        stats["xtb_ok"].append(np.mean(np.isfinite(xtb_err)))
        stats["in_qm9"].append(np.mean([s in dft for s in smiles]))
        stats["seconds"].append(sum(phys[s].get("seconds", 0.0) for s in smiles))

    pm = pd.DataFrame(per_mol)
    pm.to_csv(out / summary_name.replace("summary", "per_molecule"), index=False, float_format="%.4g")
    names = [c.removeprefix("rank ") for c in pm.columns if c.startswith("rank ")]
    summary = [{"ranking": "in candidates (ceiling)", **{f"recall@{k}": pm.in_candidates.mean() for k in KS}}]
    summary += [{"ranking": n, **{f"recall@{k}": float((pm[f"rank {n}"] <= k).mean()) for k in KS}}
                for n in names]
    s = pd.DataFrame(summary)
    s.insert(1, "n", len(pm))
    s.to_csv(out / summary_name, index=False, float_format="%.4g")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"mean candidates {np.mean(stats['candidates']):.1f}; with xTB constants "
          f"{100 * np.mean(stats['xtb_ok']):.1f}%; candidates that are QM9 molecules "
          f"{100 * np.mean(stats['in_qm9']):.1f}%; physics CPU time per molecule "
          f"{np.mean(stats['seconds']):.0f} s (summed over its candidates, before caching)")
    errs = pm.true_xtb_rms_err_pct.dropna()
    print(f"true structure's xTB rms error (bias-corrected), when proposed: median {errs.median():.2f}%, "
          f"90th pct {errs.quantile(0.9):.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--top", type=int, default=30, help="physics for each molecule's top-N model candidates")
    ap.add_argument("--cache", default="physics.jsonl", help="physics cache file name inside --out")
    ap.add_argument("--corrector", type=Path, default=None,
                    help="geometry corrector checkpoint (scripts/11_train_corrector.py)")
    ap.add_argument("--summary-name", default="summary.csv")
    ap.add_argument("--adaptive-search", action="store_true",
                    help="wider conformer/stereoisomer search (use a separate --cache)")
    ap.add_argument("--keep-geometries", action="store_true",
                    help="also store GFN2-xTB coordinates (needed for the geometry corrector)")
    ap.add_argument("--time-budget-min", type=float, default=float("inf"))
    args = ap.parse_args()
    t_start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = stage_candidates(args, args.out / "candidates.json")
    if len(rows) != args.n:
        raise SystemExit(f"{args.out}/candidates.json has {len(rows)} molecules, not --n {args.n}")
    if not stage_physics(rows, args.out / args.cache, args.workers,
                         args.time_budget_min * 60, t_start, args.top, args.keep_geometries,
                         args.adaptive_search):
        return
    import torch
    ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    stage_rank(rows, args.out / args.cache, Path(ckpt["train_config"]["data_dir"]), args.noise,
               args.out, args.top, args.corrector, args.summary_name)


if __name__ == "__main__":
    main()
