import numpy as np
import pytest
from rdkit import Chem

from rotml.analysis.cheap_methods import (
    Geometry,
    etkdg_conformers,
    mmff_relax,
    unique_geometries,
)
from rotml.physics.rotor import masses_for


def _butanol():
    return Chem.AddHs(Chem.MolFromSmiles("CCCCO"))


def test_seed_zero_is_rejected():
    with pytest.raises(ValueError):
        etkdg_conformers(_butanol(), 5, seed=0)


def test_conformer_search_finds_distinct_minima():
    mol = _butanol()
    ids = etkdg_conformers(mol, 10, seed=42)
    assert len(ids) == 10
    masses = masses_for([a.GetSymbol() for a in mol.GetAtoms()])
    geoms = [mmff_relax(mol, cid) for cid in ids]
    unique = unique_geometries(geoms, masses)
    assert 2 <= len(unique) <= 10  # butanol has several rotamers
    energies = [g.energy for g in unique]
    assert energies == sorted(energies)


def test_unique_geometries_drops_exact_duplicates():
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.5, 0.3]])
    masses = masses_for(["C", "O", "N"])
    g = Geometry(coords, 1.0)
    shifted = Geometry(coords + 5.0, 2.0)  # same shape, translated
    assert len(unique_geometries([g, shifted], masses)) == 1
