"""Plot recall@K for a milestone 3 run against its reference baselines.

Usage:
    python scripts/06b_plot_m3.py --results results/m3_cpu_medium

Reads eval_test_summary.csv (and eval_decoy_summary.csv if present) from the
results directory and writes recall_at_k.png next to them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from rotml.plotting import SERIES, SURFACE, TEXT_SECONDARY, apply_style

KS = [1, 5, 10, 50, 100]
PANELS = [("ABC", "A, B, C only"), ("ABC + formula", "A, B, C + formula")]


def curve(row: pd.Series, prefix: str) -> list[float]:
    return [100 * row[f"{prefix} recall@{k}"] for k in KS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/m3_cpu_medium"))
    args = ap.parse_args()

    test = pd.read_csv(args.results / "eval_test_summary.csv").set_index("condition")
    decoy_path = args.results / "eval_decoy_summary.csv"
    decoy = pd.read_csv(decoy_path).set_index("condition") if decoy_path.exists() else None

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (cond, title) in zip(axes, PANELS):
        row = test.loc[cond]
        series = [
            ("Model", curve(row, "model"), SERIES[0], "-", "o"),
        ]
        if decoy is not None and cond in decoy.index:
            series.append(("Model, decoy constants", curve(decoy.loc[cond], "model"), SERIES[1], "-", "s"))
        series += [
            ("Random isomer", curve(row, "random isomer"), SERIES[2], "-", "^"),
            ("Library lookup (reference)", curve(row, "library"), TEXT_SECONDARY, (0, (4, 2)), None),
        ]
        for label, y, color, ls, marker in series:
            ax.plot(KS, y, color=color, ls=ls, marker=marker, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        model_y = series[0][1]
        ax.annotate(f"{model_y[-1]:.0f}%", (KS[-1], model_y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.minorticks_off()
        ax.set_ylim(0, 102)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_xlabel("K (candidates kept)")
        ax.set_title(title)
    axes[0].set_ylabel("True structure in top K")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
    n = int(test["n"].iloc[0])
    noise = test["noise_pct"].iloc[0]
    fig.suptitle(f"Held-out formulas: {n} test molecules, {noise:g}% noise on A, B, C",
                 x=0.06, ha="left", fontsize=10, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    out = args.results / "recall_at_k.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
