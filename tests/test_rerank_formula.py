import math

import numpy as np
import pytest

from rotml.analysis.rerank import candidate_constants, diastereomers, fit_sq_error, physics_loglik
from rotml.data.qm9 import hill_formula

torch = pytest.importorskip("torch")

from rotml.ml.formula import FormulaNet, formula_string, is_valid_formula  # noqa: E402


def test_fit_sq_error():
    obs = np.array([10.0, 5.0, 4.0])
    assert fit_sq_error([[4.0, 5.0, 10.0, -1.0]], obs) == pytest.approx(0.0)  # order-free, extra columns ignored
    calc = [[10.0, 5.0, 4.0 * 1.02], [10.0, 5.0, 4.0 * 1.01]]
    assert fit_sq_error(calc, obs) == pytest.approx(math.log(1.01) ** 2)  # best conformer wins
    bias = np.full(3, math.log(1.01))
    assert fit_sq_error([obs * 1.01], obs, bias) == pytest.approx(0.0)
    assert math.isinf(fit_sq_error([], obs))
    assert math.isinf(fit_sq_error([[0.0, 5.0, 5.0]], obs))  # linear geometry


def test_physics_loglik_is_heavy_tailed_and_monotonic():
    s = 0.014
    vals = [physics_loglik(e, s) for e in (0.0, s**2, 9 * s**2, 100 * s**2)]
    assert vals == sorted(vals, reverse=True) and vals[0] == 0.0
    # Student-t penalty grows like log, not quadratically.
    assert physics_loglik(100 * s**2, s) > -0.5 * 100
    assert physics_loglik(math.inf, s) == -math.inf


def test_diastereomers_merge_enantiomers():
    from rdkit import Chem
    # 2,3-butanediol: (R,R), (S,S), meso -> 2 diastereomers.
    assert len(diastereomers(Chem.MolFromSmiles("CC(O)C(C)O"))) == 2
    assert len(diastereomers(Chem.MolFromSmiles("CCO"))) == 1
    # E/Z isomers have different constants and are both kept.
    assert len(diastereomers(Chem.MolFromSmiles("CC=CC"))) == 2


def test_candidate_constants_small_molecule():
    out = candidate_constants("O=C1CC1", n_conformers=2, use_xtb=False)
    assert out["status"] == "ok" and out["n_isomers"] == 1
    abc = np.array(out["mmff"])[0, :3]
    assert np.all(abc > 0) and abc[0] >= abc[1] >= abc[2]
    assert candidate_constants("not a smiles")["status"] == "bad_smiles"


def test_formula_helpers_match_dataset_convention():
    assert is_valid_formula((7, 6, 0, 2, 0))       # C7H6O2
    assert not is_valid_formula((7, 7, 0, 2, 0))   # odd H count
    assert not is_valid_formula((1, 6, 0, 0, 0))   # CH6: negative unsaturation
    assert not is_valid_formula((9, 8, 1, 0, 0))   # 10 heavy atoms
    for symbols in (["C"] * 7 + ["H"] * 7 + ["N", "O"], ["H", "H", "O"], ["H"] * 3 + ["N"], ["C", "F", "F", "F", "H"]):
        counts = tuple(symbols.count(e) for e in ("C", "H", "N", "O", "F"))
        assert formula_string(counts) == hill_formula(symbols)


@pytest.mark.parametrize("autoregressive", [True, False])
def test_formula_net_learns_and_proposes_valid_formulas(autoregressive):
    torch.manual_seed(0)
    from rotml.ml import features as F
    n = 64
    values = torch.randn(n, F.N_OBS)
    present = F.condition_mask(n, False, False)
    counts = torch.tensor([[6, 6, 0, 1, 0], [4, 4, 2, 0, 0]], dtype=torch.float32).repeat(n // 2, 1)
    values[: n // 2 * 2 : 2, 0] += 3.0  # make the two formulas separable from log A
    net = FormulaNet(d=64, n_layers=1, n_freq=8, d_slot=16, dropout=0.0, autoregressive=autoregressive)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(150):
        loss = net.nll(values, present, counts)
        opt.zero_grad()
        loss.backward()
        opt.step()
    props = net.propose(values[:2], present[:2], top=5)
    assert [p[0][0] for p in props] == [(6, 6, 0, 1, 0), (4, 4, 2, 0, 0)]
    assert all(is_valid_formula(c) for p in props for c, _ in p)
    assert all(p[i][1] >= p[i + 1][1] for p in props for i in range(len(p) - 1))
