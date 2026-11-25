"""Analyze milestone 2 and connect it to milestone 1.

Reads results/m2/constants_by_method.csv and writes:
    results/m2/error_summary.csv   bias, spread, tolerances and timing per method
    results/m2/funnel_link.csv     candidates surviving a stage run at each
                                   method's 95%-recall tolerance (via the M1 counts)
    results/m2/recall_vs_tolerance.png

A method's "tolerance at X% recall" is the relative tolerance on each rotational
constant within which X% of molecules have all of A, B, C reproduced. A funnel
stage run at that tolerance keeps the true structure X% of the time.
Systematic bias (one scale factor per constant) is fitted on half the
molecules and removed from the other half before computing bias-corrected
tolerances.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from rotml.analysis.identifiability import count_matches
from rotml.plotting import SERIES, SURFACE, TEXT_SECONDARY, apply_style

M2 = Path("results/m2")
QM9 = Path("data/processed/qm9.csv")

METHODS = {
    "mmff_dft": ("MMFF94", "relaxed from DFT geometry"),
    "xtb_dft": ("GFN2-xTB", "relaxed from DFT geometry"),
    "mmff_graph_best": ("MMFF94", "from graph, best-matching conformer"),
    "mmff_graph_lowest": ("MMFF94", "from graph, lowest-energy conformer"),
    "xtb_graph_best": ("GFN2-xTB", "from graph, best-matching conformer"),
    "xtb_graph_lowest": ("GFN2-xTB", "from graph, lowest-energy conformer"),
}
TIME_COL = {
    "mmff_dft": "t_mmff_dft",
    "xtb_dft": "t_xtb_dft",
    "mmff_graph_best": "t_mmff_graph",
    "mmff_graph_lowest": "t_mmff_graph",
    "xtb_graph_best": ["t_mmff_graph", "t_xtb_graph"],
    "xtb_graph_lowest": ["t_mmff_graph", "t_xtb_graph"],
}
RECALLS = (0.90, 0.95, 0.99)


def log_errors(d: pd.DataFrame, method: str) -> np.ndarray:
    pred = d[[f"{method}_{c}" for c in "ABC"]].to_numpy(dtype=float)
    ref = d[["A", "B", "C"]].to_numpy(dtype=float)
    return np.log(pred / ref)


def tolerance_at(max_err: np.ndarray, recall: float) -> float:
    """Relative tolerance at which `recall` of molecules match (failures count as misses)."""
    filled = np.where(np.isnan(max_err), np.inf, max_err)
    return float(np.expm1(np.quantile(filled, recall)))


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    fit = rng.random(len(d)) < 0.5
    rows = []
    for m, (name, setting) in METHODS.items():
        r = log_errors(d, m)
        bias = np.nanmedian(r[fit], axis=0)
        raw = np.max(np.abs(r[~fit]), axis=1)
        corr = np.max(np.abs(r[~fit] - bias), axis=1)
        tcols = TIME_COL[m] if isinstance(TIME_COL[m], list) else [TIME_COL[m]]
        row = {
            "method": m,
            "name": name,
            "setting": setting,
            "failed_frac": float(np.mean(np.isnan(r).any(axis=1))),
            "bias_A_pct": 100 * np.expm1(bias[0]),
            "bias_B_pct": 100 * np.expm1(bias[1]),
            "bias_C_pct": 100 * np.expm1(bias[2]),
            "median_abs_err_pct": 100 * np.expm1(np.nanmedian(np.abs(r))),
            "sec_per_molecule": float(d[tcols].sum(axis=1).mean()),
        }
        for q in RECALLS:
            row[f"tol_{int(q * 100)}_raw_pct"] = 100 * tolerance_at(raw, q)
            row[f"tol_{int(q * 100)}_corrected_pct"] = 100 * tolerance_at(corr, q)
        rows.append(row)
    return pd.DataFrame(rows)


def funnel_link(summary: pd.DataFrame) -> pd.DataFrame:
    """Candidates within each method's 95%-recall tolerance, counted on all of QM9."""
    qm9 = pd.read_csv(QM9)
    A, B, C = qm9.A.to_numpy(), qm9.B.to_numpy(), qm9.C.to_numpy()
    rows = []
    for s in summary.itertuples():
        delta = s.tol_95_corrected_pct / 100
        for cond, groups in (("ABC", None), ("ABC + formula", qm9.formula.to_numpy())):
            n = count_matches(A, B, C, delta, groups=groups) + 1  # include the true one
            rows.append({
                "method": s.method, "name": s.name, "setting": s.setting,
                "tol_95_corrected_pct": s.tol_95_corrected_pct, "condition": cond,
                "candidates_median": float(np.median(n)), "candidates_p90": float(np.quantile(n, 0.9)),
                "survivor_frac_median": float(np.median(n) / len(qm9)),
            })
    return pd.DataFrame(rows)


def plot_recall(d: pd.DataFrame, out: Path) -> None:
    apply_style()
    grid = np.geomspace(0.001, 0.5, 200)
    color = {"MMFF94": SERIES[0], "GFN2-xTB": SERIES[1]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    panels = [
        ("Same conformer: relaxed from DFT geometry", ["mmff_dft", "xtb_dft"]),
        ("Structure search: from graph, ≤10 conformers", ["mmff_graph_best", "mmff_graph_lowest",
                                                          "xtb_graph_best", "xtb_graph_lowest"]),
    ]
    for ax, (title, methods) in zip(axes, panels):
        for m in methods:
            name, setting = METHODS[m]
            err = np.max(np.abs(log_errors(d, m)), axis=1)
            err = np.where(np.isnan(err), np.inf, err)
            recall = [(err <= np.log1p(g)).mean() for g in grid]
            dashed = "lowest" in m
            label = name if "dft" in m else f"{name}, {'lowest energy' if dashed else 'best match'}"
            ax.plot(grid, recall, color=color[name], ls=(0, (4, 2)) if dashed else "-", label=label)
        ax.axhline(0.95, color=TEXT_SECONDARY, lw=1, ls=(0, (3, 3)))
        ax.text(0.0011, 0.955, "95% recall", color=TEXT_SECONDARY, fontsize=8.5, va="bottom")
        ax.set_xscale("log")
        ax.set_xticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x * 100:g}%"))
        ax.minorticks_off()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Tolerance on each rotational constant")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Molecules whose DFT A, B, C\nare reproduced within tolerance")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylim(0, 1.02)
    for line in axes[0].lines + axes[1].lines:
        line.set_markeredgecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    d = pd.read_csv(M2 / "constants_by_method.csv")
    d = d[d.status == "ok"]
    summary = summarize(d)
    summary.to_csv(M2 / "error_summary.csv", index=False, float_format="%.4g")
    link = funnel_link(summary)
    link.to_csv(M2 / "funnel_link.csv", index=False, float_format="%.4g")
    plot_recall(d, M2 / "recall_vs_tolerance.png")
    pd.set_option("display.width", 200)
    print(summary.drop(columns=["setting"]).round(3).to_string(index=False))
    print(link.drop(columns=["setting"]).round(3).to_string(index=False))
    print("n molecules:", len(d), " mean conformers:", round(d.n_conformers.mean(), 2))


if __name__ == "__main__":
    main()
