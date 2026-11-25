"""How many molecules are indistinguishable from each other by their observables?

Two molecules "match" at relative tolerance delta when every rotational
constant agrees within that tolerance, measured symmetrically in log space:

    |ln X_i - ln X_j| <= ln(1 + delta)   for X in (A, B, C)

This is a Chebyshev (L-infinity) ball, so a k-d tree counts matches for all
molecules without forming the O(N^2) pair list. Optional extra observables
(e.g. dipole magnitude) are added as further coordinates, scaled so their own
tolerance maps onto the same ball radius. Optional groups (e.g. molecular
formula) restrict matches to molecules in the same group.

Linear molecules have A = 0 and are matched only with other linear molecules,
on B alone; a linear rotor's spectrum is distinguishable from a non-linear one
by its line pattern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from rotml.physics.rotor import is_linear


def _count_in_ball(points: np.ndarray, radius: float) -> np.ndarray:
    """Number of *other* points within the Chebyshev ball around each point."""
    if len(points) == 1:
        return np.zeros(1, dtype=np.int64)
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius, p=np.inf, return_length=True)
    return np.asarray(counts, dtype=np.int64) - 1  # exclude self


def count_matches(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    delta: float,
    groups: np.ndarray | None = None,
    extra: dict[str, tuple[np.ndarray, float]] | None = None,
) -> np.ndarray:
    """For each molecule, count the other molecules matching it within delta.

    Parameters
    ----------
    A, B, C : rotational constants (GHz); A = 0 marks linear molecules.
    delta : relative tolerance on each rotational constant (0.01 = 1%).
    groups : optional labels; molecules only match within the same label.
    extra : optional {name: (values, absolute_tolerance)} extra observables.
    """
    A, B, C = (np.asarray(x, dtype=float) for x in (A, B, C))
    n = len(A)
    radius = np.log1p(delta)
    linear = is_linear(A, B)

    with np.errstate(divide="ignore"):
        rot = np.column_stack([np.log(A), np.log(B), np.log(C)])
    rot[linear, 0] = 0.0  # A is not used for linear rotors
    rot[linear, 2] = 0.0  # B = C for linear rotors

    extra_cols = []
    for values, tol in (extra or {}).values():
        extra_cols.append(np.asarray(values, dtype=float) * (radius / tol))
    feats = np.column_stack([rot, *extra_cols]) if extra_cols else rot

    # Linear and non-linear molecules never match each other.
    labels = pd.Series(linear.astype(int).astype(str))
    if groups is not None:
        labels = labels + "|" + pd.Series(np.asarray(groups)).astype(str)

    counts = np.zeros(n, dtype=np.int64)
    for _, idx in labels.groupby(labels).indices.items():
        counts[idx] = _count_in_ball(feats[idx], radius)
    return counts


def summarize_counts(counts: np.ndarray) -> dict[str, float]:
    """Summary statistics of per-molecule match counts."""
    counts = np.asarray(counts)
    return {
        "frac_unique": float(np.mean(counts == 0)),
        "frac_le_10": float(np.mean(counts <= 10)),
        "frac_le_100": float(np.mean(counts <= 100)),
        "median": float(np.median(counts)),
        "mean": float(np.mean(counts)),
        "p90": float(np.quantile(counts, 0.90)),
        "p99": float(np.quantile(counts, 0.99)),
        "max": int(np.max(counts)),
    }
