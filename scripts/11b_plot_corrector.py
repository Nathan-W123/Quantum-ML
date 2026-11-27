"""Plot the rotational-constant error of GFN2-xTB geometries before and after the GNN correction.

Usage:
    python scripts/11b_plot_corrector.py [--run runs/corrector] [--out results/m5_corrector]

Cumulative distribution of the per-molecule rms error over A, B, C on the
held-out-formula test molecules of data/processed/corrector/corrector.npz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter

from rotml.ml.corrector import GeometryCorrector, log_constants, masses
from rotml.plotting import SERIES, TEXT_SECONDARY, apply_style


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/corrector"))
    ap.add_argument("--data", type=Path, default=Path("data/processed/corrector/corrector.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/m5_corrector"))
    args = ap.parse_args()
    ckpt = torch.load(args.run / "corrector.pt", weights_only=False)
    model = GeometryCorrector(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    npz = np.load(args.data)
    keep = npz["split"] == "test"
    z = torch.from_numpy(npz["z"][keep])
    x = torch.from_numpy(npz["xtb_xyz"][keep])
    ref = torch.log(torch.from_numpy(npz["ref_abc"][keep]))
    with torch.no_grad():
        raw = (log_constants(x.double(), masses(z)) - ref).numpy()
        gnn = torch.cat([log_constants(model(z[s:s + 256], x[s:s + 256]), masses(z[s:s + 256]))
                         for s in range(0, len(z), 256)]) - ref
    curves = [("GFN2-xTB, bias removed", raw - np.array(ckpt["xtb_bias"]), SERIES[0], "-"),
              ("GFN2-xTB + GNN correction", gnn.numpy(), SERIES[1], "-")]

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, err, color, ls in curves:
        rms = np.sort(100 * np.sqrt((err ** 2).mean(1)))
        ax.plot(rms, 100 * np.arange(1, len(rms) + 1) / len(rms), color=color, ls=ls, label=label)
        med = np.median(rms)
        ax.annotate(f"median {med:.2f}%", (med, 50), xytext=(8, -14 if "GNN" in label else 8),
                    textcoords="offset points", fontsize=9, color=TEXT_SECONDARY)
    ax.axhline(50, color=TEXT_SECONDARY, lw=0.8, ls=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_xlim(0.05, 20)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("rms error of A, B, C vs B3LYP (same conformer)")
    ax.set_ylabel("Test molecules with smaller error")
    ax.set_title(f"Correcting GFN2-xTB geometries ({len(z)} test molecules, unseen formulas)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "error_cdf.png", dpi=200)
    print(f"wrote {args.out / 'error_cdf.png'}")


if __name__ == "__main__":
    main()
