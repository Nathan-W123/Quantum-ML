"""Train the formula-proposal model (rotational constants -> molecular formula).

Usage:
    python scripts/08_train_formula_model.py --out runs/formula [--epochs 40]

Trained on the same formula split as the inverse model, with the same noise
(0.1-1% on A, B, C, log-uniform) and the dipole shown half the time. Reports
formula coverage@K (is the true formula among the top K proposals?) on the
validation and test molecules at 0.5% noise, without and with the dipole.
Writes <out>/model.pt, history.csv and coverage.csv.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch

from rotml.ml import features as F
from rotml.ml.dataset import M3Data
from rotml.ml.formula import FormulaNet

KS = (1, 5, 10, 20, 50)


def batch_features(data: M3Data, idx: torch.Tensor, g: torch.Generator, noise: float | None,
                   dipole: bool | None):
    n = len(idx)
    if noise is None:
        noise_t = 10 ** (torch.rand(n, generator=g, dtype=torch.float64) - 3)  # 0.1-1%
    else:
        noise_t = torch.full((n,), noise, dtype=torch.float64)
    present = (F.random_condition_mask(n, 0.0, 0.5, g) if dipole is None
               else F.condition_mask(n, False, dipole))
    return data.features(idx, noise_t, present, g)


@torch.no_grad()
def coverage(model: FormulaNet, data: M3Data, split: str, n: int, dipole: bool) -> dict:
    idx = data.indices(split)
    idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))][:n]
    g = torch.Generator().manual_seed(1)
    values, present = batch_features(data, idx, g, 0.005, dipole)
    model.eval()
    nll = model.nll(values, present, data.counts[idx]).item()
    props = []
    for s in range(0, len(idx), 256):
        props += model.propose(values[s:s + 256], present[s:s + 256], top=max(KS))
    true = [tuple(int(x) for x in c) for c in data.counts[idx].tolist()]
    ranks = [next((r + 1 for r, (c, _) in enumerate(p) if c == t), math.inf) for p, t in zip(props, true)]
    return {"split": split, "dipole": dipole, "n": len(idx), "nll": nll,
            **{f"coverage@{k}": sum(r <= k for r in ranks) / len(ranks) for k in KS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed/m3"))
    ap.add_argument("--out", type=Path, default=Path("runs/formula"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--independent", action="store_true",
                    help="predict element counts independently (no conditioning on earlier elements)")
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    data = M3Data.load(args.data_dir)
    train = data.indices("train")
    model = FormulaNet(dropout=args.dropout, autoregressive=not args.independent)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(args.seed)
    history, best = [], math.inf
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = train[torch.randperm(len(train), generator=g)]
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            values, present = batch_features(data, idx, g, None, None)
            loss = model.nll(values, present, data.counts[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        val = coverage(model, data, "val", 2000, False)
        row = {"epoch": epoch, "time_s": time.time() - t0, "train_nll": tot / len(perm),
               "val_nll": val["nll"], **{k: v for k, v in val.items() if k.startswith("coverage")}}
        history.append(row)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)
        if val["nll"] < best:
            best = val["nll"]
            torch.save({"model": model.state_dict(), "config": model.config, "epoch": epoch},
                       args.out / "model.pt")
        pd.DataFrame(history).to_csv(args.out / "history.csv", index=False, float_format="%.4g")

    ckpt = torch.load(args.out / "model.pt", weights_only=False)
    model.load_state_dict(ckpt["model"])
    rows = [coverage(model, data, split, n, dip)
            for split, n in (("val", 6390), ("test", 13020)) for dip in (False, True)]
    cov = pd.DataFrame(rows)
    cov.to_csv(args.out / "coverage.csv", index=False, float_format="%.4g")
    print(f"best epoch {ckpt['epoch']}")
    print(cov.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
