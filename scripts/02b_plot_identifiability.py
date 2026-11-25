"""Figures for milestone 1 (reads results/m1/*.csv written by 02_identifiability.py)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from rotml.plotting import SERIES, SURFACE, TEXT, TEXT_SECONDARY, apply_style

M1 = Path("results/m1")
CONDITIONS = ["ABC", "ABC + formula", "ABC + |mu|", "ABC + formula + |mu|"]
LABELS = {
    "ABC": "A, B, C only",
    "ABC + formula": "+ formula",
    "ABC + |mu|": "+ dipole |μ| (±0.2 D)",
    "ABC + formula + |mu|": "+ formula + |μ|",
}
COLORS = dict(zip(CONDITIONS, SERIES))


def pct(x, _pos):
    return f"{x * 100:g}%"


def plot_unique_vs_tolerance(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for cond in CONDITIONS:
        s = summary[summary.condition == cond]
        ax.plot(s.delta, s.frac_unique, marker="o", color=COLORS[cond], label=LABELS[cond],
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for x, y, text in [(0.001, 0.04, "best composite\nmethods ~0.1%"), (0.01, 0.30, "typical DFT\n~1%")]:
        ax.axvline(x, color=TEXT_SECONDARY, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(x * 1.08, y, text, color=TEXT_SECONDARY, fontsize=8.5, va="bottom")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xticks([0.001, 0.0025, 0.005, 0.01, 0.02, 0.05])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Relative tolerance on each rotational constant")
    ax.set_ylabel("QM9 molecules uniquely identified")
    ax.set_title("How often A, B, C pick out a single QM9 molecule")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.18))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_candidates_vs_tolerance(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for cond in CONDITIONS:
        s = summary[summary.condition == cond]
        ax.plot(s.delta, s.p90 + 1, marker="o", color=COLORS[cond], label=LABELS[cond],
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xticks([0.001, 0.0025, 0.005, 0.01, 0.02, 0.05])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _p: f"{y:g}"))
    ax.set_xlabel("Relative tolerance on each rotational constant")
    ax.set_ylabel("Candidates consistent with the data\n(90th percentile, incl. the true one)")
    ax.set_title("Candidate-list size a structure search must handle")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_unique_vs_size(by_size: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    d = by_size[by_size.n_heavy >= 4]
    for cond in CONDITIONS:
        ax.plot(d.n_heavy, d[f"unique: {cond}"], marker="o", color=COLORS[cond], label=LABELS[cond],
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.02)
    ax.set_xticks(d.n_heavy)
    ax.set_xticklabels([f"{n}\n(n={c:,})" for n, c in zip(d.n_heavy, d.n_molecules)])
    ax.set_xlabel("Heavy atoms (C, N, O, F), with number of QM9 molecules")
    ax.set_ylabel("Uniquely identified at 1% tolerance")
    ax.set_title("Ambiguity grows with molecule size")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    apply_style()
    summary = pd.read_csv(M1 / "summary.csv")
    by_size = pd.read_csv(M1 / "by_heavy_atoms.csv")
    plot_unique_vs_tolerance(summary, M1 / "unique_vs_tolerance.png")
    plot_candidates_vs_tolerance(summary, M1 / "candidates_vs_tolerance.png")
    plot_unique_vs_size(by_size, M1 / "unique_vs_size.png")
    print("figures written to", M1, "| text color", TEXT)


if __name__ == "__main__":
    main()
