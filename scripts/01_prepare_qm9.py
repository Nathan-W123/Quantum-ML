"""Build data/processed/qm9.csv from the raw MoleculeNet QM9 files.

Download first (about 45 MB):

    mkdir -p data/raw/qm9_molnet
    curl -L -o data/raw/qm9_molnet/qm9.zip \
        https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
    (cd data/raw/qm9_molnet && unzip qm9.zip)

Also checks that rotational constants recomputed from the SDF geometries
agree with the reference values in the CSV. Bonds are re-perceived from the
geometries because the SDF's bond orders and charges are unreliable (see
rotml.data.qm9).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rotml.data.qm9 import build_table
from rotml.physics.rotor import is_linear

OUT = Path("data/processed/qm9.csv")


def main() -> None:
    t0 = time.time()
    table = build_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)

    nonlinear = ~is_linear(table.A, table.B)
    rel = {}
    for c in "ABC":
        ref = table.loc[nonlinear, c].to_numpy()
        geom = table.loc[nonlinear, f"{c}_geom"].to_numpy()
        rel[c] = np.abs(geom - ref) / ref
    print(f"molecules: {len(table)}  (linear: {(~nonlinear).sum()}, "
          f"no valid neutral graph from xyz: {(~table.graph_ok).sum()}, "
          f"charged graph in SDF: {(table.sdf_net_charge != 0).sum()})  in {time.time() - t0:.0f}s")
    for c, r in rel.items():
        print(f"  {c}: geometry vs reference  median rel. diff {np.median(r):.2e}, "
              f"99.9th pct {np.quantile(r, 0.999):.2e}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
