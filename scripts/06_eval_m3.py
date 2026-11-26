"""Evaluate a trained milestone 3 model on held-out molecules.

Usage:
    python scripts/06_eval_m3.py --run runs/m3_baseline [--n-eval 1000] [--samples 1000]

For each noise level and information condition, generates ranked candidate
structures (beam search + sampling, deduplicated by canonical SMILES) for test
molecules and reports recall@K: how often the true structure is among the top
K. Matching is on constitution (stereo-free canonical SMILES).

Also reports the library-lookup reference: rank all ~127k dataset molecules
(including the test ones, with exact DFT constants) by chi-square fit to the
same noisy observations. That needs a library containing the answer, so it is
close to an upper bound; the generative model has no library and has never
seen these formulas.

A second, lower reference is "random isomer": with the formula known, pick K
random QM9 molecules of that formula (expected recall min(1, K / n_isomers)).
A model that uses the rotational constants must beat it.

Control (--decoy-constants): each molecule's A, B, C and dipole are replaced
by those of a different QM9 molecule with the same formula, while recall is
still scored against the true molecule. A model that uses the constants should
fall to about the random-isomer level; one that only uses the formula should
not change.

Writes to <run>/eval/: summary.csv, per_molecule_ranks.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rotml.ml import features as F
from rotml.ml.dataset import M3Data
from rotml.ml.decode import generate_candidates
from rotml.ml.metrics import library_lookup_ranks, recall_at_k, top1_formula_accuracy
from rotml.ml.model import ModelConfig, Seq2Seq


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.001, 0.005, 0.01])
    ap.add_argument("--conditions", nargs="+", default=list(F.CONDITIONS))
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--decoy-constants", action="store_true",
                    help="control: observe the constants of a different isomer of the same formula")
    args = ap.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.run / args.checkpoint, map_location="cpu", weights_only=False)
    data_dir = args.data_dir or Path(ckpt["train_config"]["data_dir"])
    data = M3Data.load(data_dir)
    vocab = data.vocab
    assert vocab.itos == ckpt["vocab"], "checkpoint vocabulary does not match the dataset"
    model = Seq2Seq(ModelConfig(**ckpt["model_config"]), pad_id=vocab.pad)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    idx = data.indices(args.split)
    idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))][: args.n_eval]
    targets = [data.smiles[int(i)] for i in idx]
    formulas = pd.read_csv(data_dir / "m3_molecules.csv").formula.to_numpy()
    isomer_counts = pd.Series(formulas).value_counts()
    n_isomers = np.array([isomer_counts[formulas[int(i)]] for i in idx])
    obs_idx = idx
    if args.decoy_constants:
        by_formula: dict[str, list[int]] = {}
        for j, f in enumerate(formulas):
            by_formula.setdefault(f, []).append(j)
        rng = np.random.default_rng(0)
        decoys = []
        for i in idx.tolist():
            others = [j for j in by_formula[formulas[i]] if j != i]
            decoys.append(int(rng.choice(others)) if others else i)
        obs_idx = torch.as_tensor(decoys, dtype=idx.dtype)
        print(f"decoy constants: {int((obs_idx != idx).sum())}/{len(idx)} molecules get another isomer's constants")
    ks = [k for k in (1, 5, 10, 50, 100, 500, 1000) if k <= max(args.beam, args.samples + args.beam)]

    out_dir = args.run / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, per_mol = [], {"mol_index": idx.numpy(), "smiles": targets}
    t0 = time.time()
    for noise in args.noise:
        g = torch.Generator().manual_seed(int(noise * 1e6))
        noise_t = torch.full((len(idx),), noise, dtype=torch.float64)
        abc, mu = F.perturb(data.abc[obs_idx], data.mu[obs_idx], noise_t, g)
        for cond in args.conditions:
            formula, dipole = F.CONDITIONS[cond]
            present = F.condition_mask(len(idx), formula, dipole)
            values, present = data.standardize(abc, mu, data.counts[idx], noise_t, present)
            ranked = []
            for s in range(0, len(idx), args.batch):
                ranked += generate_candidates(
                    model, values[s:s + args.batch].to(device), present[s:s + args.batch].to(device),
                    vocab, beam=args.beam, n_samples=args.samples, temperature=args.temperature,
                    generator=torch.Generator(device=device).manual_seed(s) if device.type == "cuda"
                    else torch.Generator().manual_seed(s))
            model_rank = [next((r + 1 for r, (smi, _) in enumerate(c) if smi == t), np.inf)
                          for c, t in zip(ranked, targets)]
            lib_rank = library_lookup_ranks(abc, mu, idx, data.abc, data.mu, formulas,
                                            sigma_rel=noise, use_formula=formula, use_dipole=dipole)
            row = {"noise_pct": 100 * noise, "condition": cond, "n": len(idx),
                   "mean_candidates": float(np.mean([len(c) for c in ranked])),
                   "top1_formula_acc": top1_formula_accuracy(ranked, targets)}
            row |= {f"model {k}": v for k, v in recall_at_k(ranked, targets, ks).items()}
            row |= {f"library recall@{k}": float(np.mean(lib_rank <= k)) for k in ks}
            pool = n_isomers if formula else np.full(len(idx), len(formulas))
            row |= {f"random isomer recall@{k}": float(np.mean(np.minimum(1.0, k / pool))) for k in ks}
            rows.append(row)
            per_mol[f"model_rank {cond} @{noise}"] = model_rank
            per_mol[f"library_rank {cond} @{noise}"] = lib_rank
            print(f"[{time.time() - t0:.0f}s] noise {100 * noise:g}% | {cond:22s} | "
                  + " ".join(f"{k.split()[-1]}={v:.3f}" for k, v in row.items() if k.startswith("model "))
                  + f" | library@1={row['library recall@1']:.3f} @10={row['library recall@10']:.3f}",
                  flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False, float_format="%.4g")
    pd.DataFrame(per_mol).to_csv(out_dir / "per_molecule_ranks.csv", index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
