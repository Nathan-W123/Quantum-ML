"""Rigid-rotor quantities from 3D geometries.

Rotational constants follow from the principal moments of inertia:

    B [MHz] = 505379.07 / I [amu * Angstrom^2]

with the convention A >= B >= C (I_a <= I_b <= I_c). Masses are those of the
most abundant isotope, which is what quantum-chemistry codes (and QM9) use.
"""

from __future__ import annotations

import numpy as np

# h / (8 pi^2) expressed in MHz * amu * Angstrom^2.
INERTIA_TO_MHZ = 505379.07

# Most-abundant-isotope masses (amu), matching Gaussian defaults used by QM9.
ISOTOPE_MASS = {
    "H": 1.00782503,
    "C": 12.0,
    "N": 14.00307401,
    "O": 15.99491462,
    "F": 18.99840322,
    "Si": 27.97692653,
    "P": 30.97376199,
    "S": 31.97207100,
    "Cl": 34.96885268,
}

# A principal moment below this (amu A^2) is treated as zero (linear molecule).
LINEAR_MOMENT_THRESHOLD = 1e-3


def masses_for(symbols: list[str]) -> np.ndarray:
    """Isotopic masses (amu) for a list of element symbols."""
    try:
        return np.array([ISOTOPE_MASS[s] for s in symbols], dtype=float)
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"No isotope mass for element {exc.args[0]!r}") from None


def inertia_tensor(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Inertia tensor (amu A^2) about the centre of mass."""
    coords = np.asarray(coords, dtype=float)
    masses = np.asarray(masses, dtype=float)
    com = (masses[:, None] * coords).sum(axis=0) / masses.sum()
    r = coords - com
    r2 = (r**2).sum(axis=1)
    tensor = np.eye(3) * (masses * r2).sum() - np.einsum("i,ij,ik->jk", masses, r, r)
    return tensor


def principal_moments(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Principal moments of inertia I_a <= I_b <= I_c (amu A^2)."""
    moments = np.linalg.eigvalsh(inertia_tensor(coords, masses))
    return np.clip(np.sort(moments), 0.0, None)


def rotational_constants(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Rotational constants (A, B, C) in GHz, with A >= B >= C.

    For a linear molecule I_a = 0 and A is returned as 0.0, the convention
    QM9 uses.
    """
    moments = principal_moments(coords, masses)
    constants = np.zeros(3)
    nonzero = moments > LINEAR_MOMENT_THRESHOLD
    constants[nonzero] = INERTIA_TO_MHZ / moments[nonzero] / 1000.0
    return constants


def is_linear(A: np.ndarray | float, B: np.ndarray | float) -> np.ndarray:
    """Linear (or numerically near-linear) rotors: A = 0 or A/B > 1e4."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return (A == 0) | (A > 1e4 * B)


def ray_asymmetry(A, B, C) -> np.ndarray:
    """Ray's asymmetry parameter kappa = (2B - A - C) / (A - C).

    -1 is a prolate symmetric top, +1 an oblate symmetric top. Undefined
    (NaN) for spherical tops and linear molecules.
    """
    A, B, C = (np.asarray(x, dtype=float) for x in (A, B, C))
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (2 * B - A - C) / (A - C)
    return np.where(np.isfinite(kappa), kappa, np.nan)


def inertial_defect(A, B, C) -> np.ndarray:
    """Inertial defect Delta = I_c - I_a - I_b (amu A^2).

    Close to zero for planar molecules; increasingly negative with mass out
    of the molecular plane. NaN for linear molecules (A = 0).
    """
    A, B, C = (np.asarray(x, dtype=float) for x in (A, B, C))
    with np.errstate(divide="ignore"):
        Ia, Ib, Ic = (INERTIA_TO_MHZ / (x * 1000.0) for x in (A, B, C))
    delta = Ic - Ia - Ib
    return np.where(A > 0, delta, np.nan)
