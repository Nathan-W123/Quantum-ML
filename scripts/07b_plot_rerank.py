"""Plot top-1 recall of each candidate ranking from 07_physics_rerank.py.

Usage:
    python scripts/07b_plot_rerank.py --results results/m5_rerank
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from rotml.plotting import SERIES, TEXT, TEXT_SECONDARY, apply_style

LABELS = {
    "model": "Model alone",
    "mmff fit": "MMFF94 fit",
    "model + mmff": "Model + MMFF94",
    "xtb fit": "GFN2-xTB fit",
    "model + xtb": "Model + GFN2-xTB",
    "DFT lookup (oracle)": "Exact DFT constants (oracle)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/m5_rerank"))
    args = ap.parse_args()
    s = pd.read_csv(args.results / "summary.csv").set_index("ranking")
    ceiling = 100 * s.loc["in candidates (ceiling)", "recall@1"]
    names = list(LABELS)
    vals = [100 * s.loc[n, "recall@1"] for n in names]

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = range(len(names))[::-1]
    colors = [TEXT_SECONDARY if "oracle" in n else SERIES[0] for n in names]
    ax.barh(list(y), vals, height=0.6, color=colors)
    for yi, v in zip(y, vals):
        if ceiling - v < 6:  # label inside the bar so it does not collide with the ceiling line
            ax.text(v - 0.8, yi, f"{v:.1f}%", va="center", ha="right", fontsize=9, color="white")
        else:
            ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", fontsize=9, color=TEXT)
    ax.axvline(ceiling, color=TEXT_SECONDARY, ls=(0, (4, 2)), lw=1.5)
    ax.text(ceiling + 0.8, len(names) - 0.45, f"right structure among\nthe 30 candidates: {ceiling:.1f}%",
            va="top", fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_yticks(list(y), [LABELS[n] for n in names])
    ax.set_xlim(0, 70)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    n = int(s["n"].iloc[0])
    ax.set_xlabel(f"Right structure ranked first ({n} test molecules, formula given, 0.5% noise)")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_title("Re-ranking the model's top 30 candidates with physics")
    fig.tight_layout()
    out = args.results / "top1_by_ranking.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
