"""Milestone 1: identifiability of QM9 molecules from rotational constants.

For every molecule, count how many *other* QM9 molecules have A, B and C all
within a relative tolerance delta, under four information conditions:

    ABC                 rotational constants only
    ABC + formula       molecular formula also known (e.g. from mass spec)
    ABC + |mu|          dipole magnitude also known, to +/- 0.2 D
    ABC + formula + |mu|

Outputs (results/m1/):
    summary.csv                     statistics per condition and tolerance
    by_heavy_atoms.csv              fraction unique vs molecule size, delta = 1%
    identifiability_per_molecule.csv  raw counts (gitignored; regenerated)

Figures are made by scripts/02b_plot_identifiability.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from rotml.analysis.identifiability import count_matches, summarize_counts

DATA = Path("data/processed/qm9.csv")
OUT = Path("results/m1")
DELTAS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
MU_TOL = 0.2  # Debye


def main() -> None:
    t0 = time.time()
    qm9 = pd.read_csv(DATA)
    A, B, C = qm9.A.to_numpy(), qm9.B.to_numpy(), qm9.C.to_numpy()
    formula = qm9.formula.to_numpy()
    mu = {"mu": (qm9.mu.to_numpy(), MU_TOL)}

    conditions = {
        "ABC": dict(),
        "ABC + formula": dict(groups=formula),
        "ABC + |mu|": dict(extra=mu),
        "ABC + formula + |mu|": dict(groups=formula, extra=mu),
    }

    per_mol = qm9[["mol_id", "formula", "n_heavy"]].copy()
    summary = []
    for name, kwargs in conditions.items():
        for delta in DELTAS:
            counts = count_matches(A, B, C, delta, **kwargs)
            per_mol[f"{name} @ {delta}"] = counts
            summary.append({"condition": name, "delta": delta, **summarize_counts(counts)})
            s = summary[-1]
            print(f"{name:22s} delta={delta:<7} unique={s['frac_unique']:.3f} "
                  f"median={s['median']:<7.0f} p90={s['p90']:<7.0f} max={s['max']}")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    per_mol.to_csv(OUT / "identifiability_per_molecule.csv", index=False)

    rows = []
    for n_heavy, grp in per_mol.groupby("n_heavy"):
        row = {"n_heavy": n_heavy, "n_molecules": len(grp)}
        for name in conditions:
            row[f"unique: {name}"] = (grp[f"{name} @ 0.01"] == 0).mean()
            row[f"median: {name}"] = grp[f"{name} @ 0.01"].median()
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "by_heavy_atoms.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
