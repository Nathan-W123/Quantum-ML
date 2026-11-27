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

# (summary file, ranking, label); rows whose file is missing are skipped.
ROWS = [
    ("summary.csv", "model", "Model alone"),
    ("summary.csv", "model + mmff", "Model + MMFF94"),
    ("summary.csv", "model + xtb", "Model + GFN2-xTB"),
    ("summary_gnn.csv", "model + gnn", "Model + GFN2-xTB + GNN correction"),
    ("summary_adaptive.csv", "model + gnn", "... + wider conformer search"),
    ("summary.csv", "DFT lookup (oracle)", "Exact DFT constants (oracle)"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/m5_rerank"))
    args = ap.parse_args()
    tables = {f: pd.read_csv(args.results / f).set_index("ranking")
              for f in {r[0] for r in ROWS} if (args.results / f).exists()}
    rows = [r for r in ROWS if r[0] in tables and r[1] in tables[r[0]].index]
    s = tables["summary.csv"]
    ceiling = 100 * s.loc["in candidates (ceiling)", "recall@1"]
    names = [r[2] for r in rows]
    vals = [100 * tables[f].loc[k, "recall@1"] for f, k, _ in rows]

    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(names) + 1.5))
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
    ax.set_yticks(list(y), names)
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
