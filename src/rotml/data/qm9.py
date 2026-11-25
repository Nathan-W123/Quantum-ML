"""Loading QM9 (MoleculeNet distribution: gdb9.sdf + gdb9.sdf.csv).

The SDF holds the B3LYP/6-31G(2df,p) geometries (4-decimal coordinates); the
CSV holds the reference properties, including rotational constants A, B, C
(GHz) and dipole moment mu (Debye).

Caution: the SDF's *bond orders and formal charges are wrong for about 19% of
molecules* (e.g. the imine CCN=C(C)OCCO is stored with single bonds and an
N+ charge). Read naively, RDKit "repairs" such records by adding implicit
hydrogens, which silently changes the molecule and its formula. We therefore
use the SDF only for atoms and coordinates, re-perceive bonds from the
geometry with total charge 0 (all QM9 molecules are neutral), and compute
formulas from the explicit atoms.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

from rotml.physics.rotor import masses_for, rotational_constants

RAW_DIR = Path("data/raw/qm9_molnet")
SDF_NAME = "gdb9.sdf"
CSV_NAME = "gdb9.sdf.csv"

RDLogger.DisableLog("rdApp.*")


@dataclass
class QM9Record:
    mol_id: str
    symbols: list[str]
    coords: np.ndarray


def iter_sdf(sdf_path: Path, indices: list[int] | None = None):
    """Yield (index, unsanitized RDKit mol) from the SDF. Use only atoms and coordinates."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    order = range(len(supplier)) if indices is None else indices
    for i in order:
        mol = supplier[i]
        if mol is not None:
            yield i, mol


def iter_records(sdf_path: Path, indices: list[int] | None = None):
    """Yield (index, QM9Record) with atoms and coordinates only."""
    for i, mol in iter_sdf(sdf_path, indices):
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        yield i, QM9Record(mol.GetProp("_Name"), symbols, mol.GetConformer().GetPositions())


def hill_formula(symbols: list[str]) -> str:
    """Hill-notation formula from explicit atoms (C, H first, then alphabetical)."""
    counts = Counter(symbols)
    order = [e for e in ("C", "H") if e in counts] + sorted(e for e in counts if e not in ("C", "H"))
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order)


def graph_from_xyz(symbols: list[str], coords: np.ndarray, charge: int = 0) -> Chem.Mol | None:
    """Molecule with bonds perceived from 3D coordinates (explicit H, 3D conformer).

    Returns None if RDKit cannot assign a valid neutral Lewis structure.
    Stereochemistry is assigned from the 3D geometry.
    """
    lines = [str(len(symbols)), ""]
    lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, coords)]
    mol = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - RDKit raises several exception types
        return None
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def summarize_record(rec: QM9Record, sdf_mol: Chem.Mol | None = None) -> dict:
    """Formula, canonical stereo SMILES (from re-perceived bonds) and geometry constants."""
    A, B, C = rotational_constants(rec.coords, masses_for(rec.symbols))
    mol = graph_from_xyz(rec.symbols, rec.coords)
    row = {
        "mol_id": rec.mol_id,
        "formula": hill_formula(rec.symbols),
        "n_atoms": len(rec.symbols),
        "n_heavy": sum(s != "H" for s in rec.symbols),
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)) if mol is not None else None,
        "graph_ok": mol is not None,
        "A_geom": A,
        "B_geom": B,
        "C_geom": C,
    }
    if sdf_mol is not None:
        row["sdf_net_charge"] = sum(a.GetFormalCharge() for a in sdf_mol.GetAtoms())
    return row


def build_table(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Merge geometry-derived descriptors with the reference property CSV."""
    props = pd.read_csv(raw_dir / CSV_NAME, usecols=["mol_id", "A", "B", "C", "mu"])
    rows = []
    for i, sdf_mol in iter_sdf(raw_dir / SDF_NAME):
        rec = QM9Record(sdf_mol.GetProp("_Name"), [a.GetSymbol() for a in sdf_mol.GetAtoms()],
                        sdf_mol.GetConformer().GetPositions())
        rows.append(summarize_record(rec, sdf_mol) | {"sdf_index": i})
    geom = pd.DataFrame(rows)
    return props.merge(geom, on="mol_id", how="inner", validate="one_to_one")
