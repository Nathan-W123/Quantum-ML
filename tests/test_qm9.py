import numpy as np
from rdkit import Chem

from rotml.data.qm9 import graph_from_xyz, hill_formula

# QM9 gdb_121852: the MoleculeNet SDF stores this imine with single bonds and
# an N+ charge, which RDKit turns into a protonated amine with extra hydrogens.
GDB_121852 = """C 0.4789 0.9285 0.5022
C 0.1309 -0.5128 0.1209
N -1.3063 -0.6746 0.0135
C -1.8412 -1.1501 -1.0288
C -1.2033 -1.6371 -2.3102
O -3.1902 -1.2967 -1.0995
C -3.9513 -0.9323 0.0610
C -5.3984 -1.2463 -0.2750
O -5.5957 -2.6209 -0.5456
H 1.5566 1.0395 0.6584
H 0.1728 1.6236 -0.2862
H -0.0397 1.2147 1.4215
H 0.6779 -0.7952 -0.7911
H 0.4845 -1.1848 0.9146
H -0.7385 -2.6169 -2.1593
H -1.9608 -1.7318 -3.0888
H -0.4255 -0.9510 -2.6523
H -3.6046 -1.5149 0.9203
H -3.8044 0.1293 0.2874
H -6.0297 -0.9958 0.5839
H -5.7199 -0.6187 -1.1217
H -4.9445 -2.8572 -1.2155"""


def _parse(block):
    symbols, coords = [], []
    for line in block.splitlines():
        s, x, y, z = line.split()
        symbols.append(s)
        coords.append([float(x), float(y), float(z)])
    return symbols, np.array(coords)


def test_hill_formula():
    assert hill_formula(["O", "H", "C", "H", "C", "N", "H"]) == "C2H3NO"
    assert hill_formula(["H", "O", "H"]) == "H2O"


def test_bonds_reperceived_from_geometry():
    symbols, coords = _parse(GDB_121852)
    mol = graph_from_xyz(symbols, coords)
    assert mol is not None
    assert Chem.GetFormalCharge(mol) == 0
    assert hill_formula(symbols) == "C6H13NO2"
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    assert Chem.CanonSmiles(smiles, useChiral=0) == Chem.CanonSmiles("CCN=C(C)OCCO", useChiral=0)
