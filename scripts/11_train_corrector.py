"""Train the geometry corrector: GFN2-xTB geometry -> DFT-quality geometry and constants.

Usage:
    python scripts/11_train_corrector.py [--epochs 30] [--out runs/corrector]

Needs data/processed/corrector/corrector.npz (scripts/10_build_corrector_data.py).
Trains on "train" molecules, selects the epoch by validation error, and reports
the rotational-constant error on held-out-formula "test" molecules for:
  raw xTB          constants of the GFN2-xTB geometry
  xTB - bias       the same with the median per-constant bias removed (what
                   the milestone 5 prototype uses)
  GNN-corrected    constants of the corrected geometry
Writes <out>/corrector.pt, history.csv, metrics.csv.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rotml.ml.corrector import GeometryCorrector, corrector_loss, log_constants, masses

DATA = Path("data/processed/corrector/corrector.npz")


def load(split: str, z: np.ndarray, split_arr: np.ndarray, *arrays):
    keep = split_arr == split
    return [torch.from_numpy(a[keep]) for a in (z, *arrays)]


def trim(z, *xs):
    n = int((z > 0).sum(1).max())
    return [z[:, :n]] + [x[:, :n] for x in xs]


@torch.no_grad()
def predict_log_abc(model, z, x, batch: int = 256) -> torch.Tensor:
    model.eval()
    out = []
    for s in range(0, len(z), batch):
        zb, xb = trim(z[s:s + batch], x[s:s + batch])
        out.append(log_constants(model(zb, xb), masses(zb)))
    return torch.cat(out)


def error_stats(err: np.ndarray, name: str, split: str) -> dict:
    """err: (n, 3) log errors. Per-molecule rms over A, B, C, in percent."""
    rms = 100 * np.sqrt((err ** 2).mean(1))
    return {"split": split, "method": name, "n": len(err),
            **{f"rms_err_q{int(q * 100)}_pct": float(np.quantile(rms, q)) for q in (0.5, 0.68, 0.9, 0.95)},
            "robust_sd_per_constant_pct": float(100 * 1.4826 * np.median(np.abs(err - np.median(err, 0))))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=Path("runs/corrector"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.data)
    arrays = (npz["xtb_xyz"], npz["dft_xyz"], npz["ref_abc"])
    data = {s: load(s, npz["z"], npz["split"], *arrays) for s in ("train", "val", "test")}
    print({s: len(v[0]) for s, v in data.items()}, flush=True)

    # Baseline bias: median log(xTB / DFT) per constant on training molecules.
    z_tr, x_tr, _, abc_tr = data["train"]
    raw_tr = log_constants(x_tr.double(), masses(z_tr)) - torch.log(abc_tr)
    bias = raw_tr.median(0).values
    print("train median xTB bias (%):", (100 * bias).numpy().round(2), flush=True)

    model = GeometryCorrector(d=args.d, n_layers=args.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(z_tr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    z_va, x_va, _, abc_va = data["val"]
    g = torch.Generator().manual_seed(args.seed)
    history, best = [], math.inf
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(z_tr), generator=g)
        sums = torch.zeros(3)
        for s in range(0, len(perm), args.batch):
            i = perm[s:s + args.batch]
            zb, xb, rb = trim(z_tr[i], x_tr[i], data["train"][2][i])
            loss, l_abc, l_dist = corrector_loss(model, zb, xb, rb, abc_tr[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            sums += torch.tensor([loss.item(), l_abc.item(), l_dist.item()]) * len(i)
        err_va = (predict_log_abc(model, z_va, x_va) - torch.log(abc_va)).numpy()
        val_mse = float((err_va ** 2).mean())
        st = error_stats(err_va, "GNN", "val")
        row = {"epoch": epoch, "time_s": time.time() - t0,
               "train_loss": sums[0].item() / len(perm), "train_abc_mse": sums[1].item() / len(perm),
               "train_dist_mse": sums[2].item() / len(perm), "val_abc_mse": val_mse,
               "val_rms_q50_pct": st["rms_err_q50_pct"], "val_rms_q90_pct": st["rms_err_q90_pct"]}
        history.append(row)
        print(" ".join(f"{k}={v:.4g}" for k, v in row.items()), flush=True)
        if val_mse < best:
            best = val_mse
            torch.save({"model": model.state_dict(), "config": model.config, "epoch": epoch,
                        "xtb_bias": bias.tolist()}, args.out / "corrector.pt")
        pd.DataFrame(history).to_csv(args.out / "history.csv", index=False, float_format="%.5g")

    ckpt = torch.load(args.out / "corrector.pt", weights_only=False)
    model.load_state_dict(ckpt["model"])
    rows = []
    for split in ("val", "test"):
        z, x, _, abc = data[split]
        raw = (log_constants(x.double(), masses(z)) - torch.log(abc)).numpy()
        rows.append(error_stats(raw, "raw xTB", split))
        rows.append(error_stats(raw - bias.numpy(), "xTB - bias", split))
        rows.append(error_stats((predict_log_abc(model, z, x) - torch.log(abc)).numpy(), "GNN-corrected", split))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out / "metrics.csv", index=False, float_format="%.4g")
    print(f"best epoch {ckpt['epoch']}")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
