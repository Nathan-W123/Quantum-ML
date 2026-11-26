"""Structure recall without a known formula, using proposed formulas.

Usage:
    python scripts/09_eval_formula_proposals.py --run runs/m3_cpu_medium \
        --formula-model runs/formula/model.pt [--n-eval 500] [--top-formulas 10] [--dipole]

Two-step pipeline for an unknown formula:
  1. The formula model proposes the top M formulas from A, B, C (and the
     dipole if --dipole).
  2. The inverse model is run once per proposed formula (beam + sampling),
     exactly as in the "+ formula" condition.
Candidates are pooled and ranked by log p(formula) + log p(structure | formula).

Uses the same test molecules and noise draws as 06_eval_m3.py, so the result
compares directly with its "ABC" (and "ABC + |mu|") rows. Writes
<run>/eval_formula_proposals/summary.csv and per_molecule_ranks.csv.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rotml.ml import features as F
from rotml.ml.dataset import M3Data
from rotml.ml.decode import generate_candidates
from rotml.ml.formula import FormulaNet
from rotml.ml.metrics import recall_at_k
from rotml.ml.model import ModelConfig, Seq2Seq

KS = (1, 5, 10, 50, 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--formula-model", type=Path, required=True)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--top-formulas", type=int, default=10)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--dipole", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-name", default="eval_formula_proposals")
    args = ap.parse_args()

    ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    data_dir = Path(ckpt["train_config"]["data_dir"])
    data = M3Data.load(data_dir)
    vocab = data.vocab
    model = Seq2Seq(ModelConfig(**ckpt["model_config"]), pad_id=vocab.pad)
    model.load_state_dict(ckpt["model"])
    model.eval()
    fck = torch.load(args.formula_model, map_location="cpu", weights_only=False)
    fmodel = FormulaNet(**fck["config"])
    fmodel.load_state_dict(fck["model"])
    fmodel.eval()

    # Same molecules and noise as 06_eval_m3.py.
    idx = data.indices("test")
    idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))][: args.n_eval]
    targets = [data.smiles[int(i)] for i in idx]
    noise_t = torch.full((len(idx),), args.noise, dtype=torch.float64)
    g = torch.Generator().manual_seed(int(args.noise * 1e6))
    abc, mu = F.perturb(data.abc[idx], data.mu[idx], noise_t, g)

    # 1. Formula proposals.
    present = F.condition_mask(len(idx), False, args.dipole)
    values, present = data.standardize(abc, mu, data.counts[idx], noise_t, present)
    proposals = fmodel.propose(values, present, top=args.top_formulas)
    true_counts = [tuple(int(x) for x in c) for c in data.counts[idx].tolist()]
    formula_rank = np.array([next((r + 1 for r, (c, _) in enumerate(p) if c == t), np.inf)
                             for p, t in zip(proposals, true_counts)])

    # 2. Inverse model once per (molecule, proposed formula).
    pairs = [(m, c, lp) for m, props in enumerate(proposals) for c, lp in props]
    m_idx = torch.tensor([p[0] for p in pairs])
    counts = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    present_f = F.condition_mask(len(pairs), True, args.dipole)
    values_f, present_f = data.standardize(abc[m_idx], mu[m_idx], counts, noise_t[m_idx], present_f)
    t0 = time.time()
    per_pair = []
    for s in range(0, len(pairs), args.batch):
        per_pair += generate_candidates(model, values_f[s:s + args.batch], present_f[s:s + args.batch],
                                        vocab, beam=args.beam, n_samples=args.samples,
                                        generator=torch.Generator().manual_seed(s))
        if (s // args.batch) % 10 == 0:
            print(f"[{time.time() - t0:.0f}s] {min(s + args.batch, len(pairs))}/{len(pairs)} "
                  "(molecule, formula) pairs decoded", flush=True)

    pooled: list[dict[str, float]] = [{} for _ in idx]
    for (m, _, f_lp), cands in zip(pairs, per_pair):
        for smi, s_lp in cands:
            score = f_lp + s_lp
            if score > pooled[m].get(smi, -math.inf):
                pooled[m][smi] = score
    ranked = [sorted(d.items(), key=lambda kv: -kv[1]) for d in pooled]

    model_rank = [next((r + 1 for r, (smi, _) in enumerate(c) if smi == t), np.inf)
                  for c, t in zip(ranked, targets)]
    cond = "ABC + |mu|" if args.dipole else "ABC"
    row = {"condition": f"{cond}, top-{args.top_formulas} proposed formulas", "noise_pct": 100 * args.noise,
           "n": len(idx), "decodes_per_molecule": args.top_formulas * (args.beam + args.samples),
           "mean_candidates": float(np.mean([len(c) for c in ranked]))}
    row |= {f"formula coverage@{k}": float(np.mean(formula_rank <= k))
            for k in (1, 5, 10, 20, 50) if k <= args.top_formulas}
    row |= {f"model {k}": v for k, v in recall_at_k(ranked, targets, list(KS)).items()}
    out = args.run / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out / "summary.csv", index=False, float_format="%.4g")
    pd.DataFrame({"mol_index": idx.numpy(), "smiles": targets, "formula_rank": formula_rank,
                  "model_rank": model_rank}).to_csv(out / "per_molecule_ranks.csv", index=False)
    for k, v in row.items():
        print(f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
