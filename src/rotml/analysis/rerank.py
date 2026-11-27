"""Physics re-ranking of proposed structures (first prototype of milestone 5).

The inverse model proposes stereo-free candidate structures. For each one we
compute rotational constants from 3D geometries and score how well they fit
the observed constants:

1. Enumerate stereoisomers (diastereomers only: enantiomers have identical
   rotational constants), up to a cap.
2. Embed ETKDG conformers for each, relax with MMFF94, drop duplicates, then
   relax each with GFN2-xTB.
3. Score a candidate by its best-fitting conformer. A molecule in a cold
   spectrum is in one specific conformer, and the search does not know which.

GFN2-xTB constants are systematically ~1.1-1.7% larger than B3LYP ones
(milestone 2); that bias is divided out before scoring.
"""

from __future__ import annotations

import math
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from rotml.analysis.cheap_methods import etkdg_conformers, mmff_relax, unique_geometries, xtb_relax
from rotml.physics.rotor import masses_for, rotational_constants

# Median log(GFN2-xTB / B3LYP) for A, B, C, from milestone 2 (relaxed from the DFT geometry).
XTB_BIAS = np.array([0.0113, 0.0158, 0.0166])
# Robust spread of GFN2-xTB vs B3LYP per constant for a searched conformer (milestone 2).
XTB_SIGMA = 0.014


def diastereomers(mol: Chem.Mol, max_isomers: int = 8, seed: int = 42,
                  try_embedding: bool = False) -> list[Chem.Mol]:
    """Stereoisomers of mol with one representative per enantiomer pair.

    try_embedding keeps only stereoisomers RDKit can embed in 3D: in strained
    cages most diastereomers are impossible, and sampling among them misses
    the feasible ones.
    """
    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True, tryEmbedding=try_embedding)
    kept: dict[str, Chem.Mol] = {}
    for iso in EnumerateStereoisomers(mol, options=opts):
        iso = Chem.Mol(iso)
        iso.RemoveAllConformers()  # a 3D conformer (from tryEmbedding) would override the chiral tags
        smi = Chem.MolToSmiles(iso)
        mirror = Chem.Mol(iso)
        for atom in mirror.GetAtoms():
            atom.InvertChirality()
        key = min(smi, Chem.MolToSmiles(mirror))
        kept.setdefault(key, iso)
    isomers = [kept[k] for k in sorted(kept)]
    if len(isomers) > max_isomers:
        rng = np.random.default_rng(seed)
        isomers = [isomers[i] for i in sorted(rng.choice(len(isomers), max_isomers, replace=False))]
    return isomers


def candidate_constants(smiles: str, n_conformers: int = 10, max_isomers: int = 8,
                        seed: int = 42, use_xtb: bool = True, keep_geometries: bool = False,
                        adaptive: bool = False) -> dict:
    """Rotational constants (GHz) of a candidate's relaxed conformers.

    Returns {"smiles", "status", "n_isomers", "mmff": [[A, B, C, E], ...],
    "xtb": [[A, B, C, E], ...], "seconds"}. Energies are per isomer, so they
    are only comparable within one stereoisomer. With keep_geometries, also
    "symbols" and "xtb_xyz" (one coordinate list per "xtb" row; every
    stereoisomer has the same atom order).

    adaptive=True is a wider search for hard cases: only stereoisomers that can
    be embedded (all of them if none can), up to 16 of them; 10 + 5 conformers
    per rotatable bond (up to 30), half of them embedded permissively
    (random coordinates, no torsion preferences) for rotamer diversity and
    strained cages.
    """
    t0 = time.perf_counter()
    out = {"smiles": smiles, "status": "ok", "n_isomers": 0, "mmff": [], "xtb": []}
    if keep_geometries:
        out["xtb_xyz"] = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out["status"] = "bad_smiles"
        return out
    if adaptive:
        n_conformers = min(30, 10 + 5 * rdMolDescriptors.CalcNumRotatableBonds(mol))
        max_isomers = max(max_isomers, 16)
        isomers = (diastereomers(mol, max_isomers, seed, try_embedding=True)
                   or diastereomers(mol, max_isomers, seed))
    else:
        isomers = diastereomers(mol, max_isomers, seed)
    out["n_isomers"] = len(isomers)
    k = max(3 if adaptive else 2, math.ceil(n_conformers / len(isomers)))
    for iso in isomers:
        molh = Chem.AddHs(iso)
        symbols = [a.GetSymbol() for a in molh.GetAtoms()]
        masses = masses_for(symbols)
        etkdg_conformers(molh, k, seed)
        if adaptive:
            etkdg_conformers(molh, k, seed + 1, permissive=True, keep_existing=True)
        if molh.GetNumConformers() == 0:
            continue
        geoms = [mmff_relax(molh, cid) for cid in range(molh.GetNumConformers())]
        with np.errstate(divide="ignore", invalid="ignore"):  # linear geometries have A = 0
            geoms = unique_geometries([g for g in geoms if g is not None], masses)
        for g in geoms:
            out["mmff"].append([*rotational_constants(g.coords, masses), g.energy])
            if use_xtb:
                x = xtb_relax(symbols, g.coords)
                if x is not None:
                    out["xtb"].append([*rotational_constants(x.coords, masses), x.energy])
                    if keep_geometries:
                        out["symbols"] = symbols
                        out["xtb_xyz"].append(np.round(x.coords, 5).tolist())
    if not out["mmff"]:
        out["status"] = "no_embedding"
    elif use_xtb and not out["xtb"]:
        out["status"] = "xtb_failed"
    out["seconds"] = time.perf_counter() - t0
    return out


def fit_sq_error(calc: np.ndarray, obs_abc: np.ndarray, bias: np.ndarray | None = None) -> float:
    """Smallest sum over A, B, C of squared log errors, over a candidate's conformers.

    calc: (n_conf, >=3) array whose first three columns are A, B, C (GHz).
    Returns inf when there are no conformers or a constant is not positive
    (linear geometry).
    """
    calc = np.asarray(calc, dtype=float)
    if calc.size == 0:
        return math.inf
    abc = np.sort(calc.reshape(len(calc), -1)[:, :3], axis=1)[:, ::-1]
    obs = np.sort(np.asarray(obs_abc, dtype=float))[::-1]
    ok = (abc > 0).all(axis=1)
    if not ok.any():
        return math.inf
    r = np.log(abc[ok] / obs)
    if bias is not None:
        r = r - bias
    return float((r ** 2).sum(axis=1).min())


def physics_loglik(sq_err: float, sigma: float, nu: float = 3.0) -> float:
    """Student-t log-likelihood (up to a constant) of a squared log error over 3 constants.

    Heavy tails because cheap-method errors have them (milestone 2: the 95th
    percentile is ~9x the median), so one bad conformer search does not
    eliminate the right answer outright.
    """
    if not math.isfinite(sq_err):
        return -math.inf
    return -0.5 * (nu + 3) * math.log1p(sq_err / (nu * sigma ** 2))
