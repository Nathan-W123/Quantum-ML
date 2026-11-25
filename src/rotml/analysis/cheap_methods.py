"""Milestone 2: how accurate are cheap geometry methods for rotational constants?

Reference values are QM9's B3LYP/6-31G(2df,p) rotational constants. For each
molecule we compute A, B, C with cheaper methods, in two settings:

* "from DFT geometry": start at the QM9 geometry and relax with the cheap
  method. The conformer is (almost always) the same, so this isolates the
  method's intrinsic error.
* "from graph": start from the molecular graph only (with stereo), as a
  structure search would. Embed K conformers with ETKDG, relax them, and
  report both the lowest-energy conformer and the best-matching one. The
  best-matching error is what a funnel stage needs: a candidate survives if
  any of its conformers matches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from rotml.data.qm9 import graph_from_xyz
from rotml.physics.rotor import masses_for, rotational_constants

XTB_FMAX = 0.01  # eV / Angstrom
XTB_MAX_STEPS = 400


@dataclass
class Geometry:
    coords: np.ndarray
    energy: float


def mmff_relax(mol: Chem.Mol, conf_id: int = -1) -> Geometry | None:
    """MMFF94 relaxation of one conformer (modifies mol in place)."""
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    ff.Minimize(maxIts=2000)
    coords = mol.GetConformer(conf_id).GetPositions()
    return Geometry(coords, ff.CalcEnergy())


def xtb_relax(symbols: list[str], coords: np.ndarray, charge: int = 0) -> Geometry | None:
    """GFN2-xTB relaxation (tblite + ASE BFGS). Energy in eV."""
    from ase import Atoms
    from ase.optimize import BFGS
    from tblite.ase import TBLite

    atoms = Atoms(symbols, positions=coords)
    atoms.calc = TBLite(method="GFN2-xTB", charge=charge, verbosity=0)
    try:
        opt = BFGS(atoms, logfile=None)
        converged = opt.run(fmax=XTB_FMAX, steps=XTB_MAX_STEPS)
        if not converged:
            return None
        return Geometry(atoms.get_positions(), atoms.get_potential_energy())
    except Exception:  # noqa: BLE001 - SCF failures etc.
        return None


def etkdg_conformers(mol: Chem.Mol, k: int, seed: int) -> list[int]:
    """Embed k conformers from the graph alone (QM9 coordinates discarded).

    No RMSD pruning at embedding time: duplicates are removed after
    relaxation instead (see unique_geometries), when they are real duplicates.

    Note: RDKit gives k *identical* conformers when randomSeed == 0, so the
    seed must be non-zero.
    """
    if seed == 0:
        raise ValueError("randomSeed=0 makes RDKit embed identical conformers; use a non-zero seed")
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    mol.RemoveAllConformers()
    return list(AllChem.EmbedMultipleConfs(mol, numConfs=k, params=params))


def unique_geometries(geoms: list[Geometry], masses: np.ndarray, rel_tol: float = 1e-3
                      ) -> list[Geometry]:
    """Drop relaxed conformers whose A, B, C all match an earlier one within rel_tol."""
    kept: list[Geometry] = []
    kept_abc: list[np.ndarray] = []
    for g in sorted(geoms, key=lambda g: g.energy):
        abc = rotational_constants(g.coords, masses)
        if any(np.all(np.abs(np.log(abc / k)) < rel_tol) for k in kept_abc):
            continue
        kept.append(g)
        kept_abc.append(abc)
    return kept


def _abc(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return rotational_constants(coords, masses)


def evaluate_molecule(symbols: list[str], dft_coords: np.ndarray, ref_abc: np.ndarray,
                      k_conformers: int = 10, seed: int = 42) -> dict:
    """Rotational constants of one QM9 molecule under every cheap method.

    The molecular graph is re-perceived from the DFT geometry (QM9 molecules
    are neutral, so charge 0). Returns a flat dict: {method}_{A|B|C} plus
    per-method wall time (s) and the number of distinct conformers.
    """
    out: dict = {}
    mol = graph_from_xyz(symbols, dft_coords, charge=0)
    if mol is None:
        return {"status": "no_graph"}
    masses = masses_for(symbols)
    charge = 0

    def record(name: str, abc: np.ndarray | None) -> None:
        for c, v in zip("ABC", abc if abc is not None else [np.nan] * 3):
            out[f"{name}_{c}"] = v

    # Same conformer: relax from the DFT geometry.
    t = time.perf_counter()
    m = Chem.Mol(mol)
    g = mmff_relax(m)
    out["t_mmff_dft"] = time.perf_counter() - t
    record("mmff_dft", None if g is None else _abc(g.coords, masses))

    t = time.perf_counter()
    g = xtb_relax(symbols, dft_coords, charge)
    out["t_xtb_dft"] = time.perf_counter() - t
    record("xtb_dft", None if g is None else _abc(g.coords, masses))

    # From graph: ETKDG conformers, relaxed with MMFF, then with GFN2-xTB.
    t = time.perf_counter()
    m = Chem.Mol(mol)
    conf_ids = etkdg_conformers(m, k_conformers, seed)
    mmff = [mmff_relax(m, cid) for cid in conf_ids]
    mmff = unique_geometries([g for g in mmff if g is not None], masses)
    out["t_mmff_graph"] = time.perf_counter() - t
    out["n_conformers"] = len(mmff)

    t = time.perf_counter()
    xtb = [xtb_relax(symbols, g.coords, charge) for g in mmff]
    xtb = [g for g in xtb if g is not None]
    out["t_xtb_graph"] = time.perf_counter() - t

    for name, geoms in (("mmff_graph", mmff), ("xtb_graph", xtb)):
        if not geoms:
            record(f"{name}_lowest", None)
            record(f"{name}_best", None)
            continue
        abcs = np.array([_abc(g.coords, masses) for g in geoms])
        energies = np.array([g.energy for g in geoms])
        record(f"{name}_lowest", abcs[np.argmin(energies)])
        errs = np.max(np.abs(np.log(abcs / ref_abc)), axis=1)
        record(f"{name}_best", abcs[np.argmin(errs)])

    out["status"] = "ok"
    return out
