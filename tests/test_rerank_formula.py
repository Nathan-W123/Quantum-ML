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


def test_corrector_is_equivariant_and_starts_at_identity():
    from rotml.ml.corrector import GeometryCorrector, corrector_loss, log_constants, masses
    from rotml.physics.rotor import masses_for, rotational_constants

    torch.manual_seed(0)
    z = torch.tensor([[6, 8, 1, 1, 0], [6, 6, 1, 1, 1]])
    x = torch.randn(2, 5, 3)
    ref = rotational_constants(x[1].numpy().astype(float), masses_for(["C", "C", "H", "H", "H"]))
    assert np.allclose(np.exp(log_constants(x[1:2], masses(z[1:2])).numpy()[0]), ref, rtol=1e-6)

    net = GeometryCorrector(d=16, n_layers=1, n_rbf=8)
    assert torch.allclose(net(z, x), x)  # zero-initialised last layer: no correction yet
    for p in net.coord[-1].parameters():
        torch.nn.init.normal_(p, std=0.1)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    t = torch.randn(1, 1, 3)
    y, y2 = net(z, x), net(z, x @ q.T + t)
    real = z > 0
    assert torch.allclose(y2[real], (y @ q.T + t)[real], atol=1e-5)
    assert torch.allclose(log_constants(y, masses(z)), log_constants(y2, masses(z)), atol=1e-6)
    loss, _, _ = corrector_loss(net, z, x, x, torch.exp(log_constants(x, masses(z))))
    loss.backward()


def test_diastereomers_with_embedding_filter_merges_enantiomers():
    from rdkit import Chem
    # Strained cage: 20 diastereomers on paper, only 4 (all meso) can be embedded in 3D.
    mol = Chem.MolFromSmiles("C1C2C1C1OC2C2OC12")
    assert len(diastereomers(mol, max_isomers=100)) == 20
    assert len(diastereomers(mol, max_isomers=100, try_embedding=True)) == 4
    # With the embedding filter, chiral isomers are still merged with their mirror images.
    assert len(diastereomers(Chem.MolFromSmiles("CC(O)C(C)O"), try_embedding=True)) == 2


def test_adaptive_search_embeds_strained_cage_and_finds_rotamers():
    out = candidate_constants("c1cc2n(c1)C1CN2C1", use_xtb=False)
    assert out["status"] == "no_embedding"
    assert candidate_constants("c1cc2n(c1)C1CN2C1", use_xtb=False, adaptive=True)["status"] == "ok"
    # Both planar rotamers of 2-fluorobenzaldehyde.
    assert len(candidate_constants("O=Cc1ccccc1F", use_xtb=False, adaptive=True)["mmff"]) == 2


def test_adaptive_isomers_include_default_isomers():
    from rdkit import Chem
    from rotml.analysis.rerank import enantiomer_pair_key, search_isomers
    # RDKit's embedding test keeps only 1 of these 4 diastereomers, and not the
    # one in QM9; the adaptive search must still try all 4.
    for smi in ("C#CC12CC(C=O)C1O2", "CC1(O)C(O)C2OC21", "C1C2C1C1OC2C2OC12"):
        mol = Chem.MolFromSmiles(smi)
        default = {enantiomer_pair_key(m) for m in search_isomers(mol)}
        adaptive = {enantiomer_pair_key(m) for m in search_isomers(mol, adaptive=True)}
        assert default <= adaptive and len(adaptive) <= max(16, len(default))
    # Cage with 20 diastereomers: the 4 embeddable ones are added to the 8 sampled.
    cage = Chem.MolFromSmiles("C1C2C1C1OC2C2OC12")
    embeddable = {enantiomer_pair_key(m) for m in diastereomers(cage, 100, try_embedding=True)}
    assert embeddable <= {enantiomer_pair_key(m) for m in search_isomers(cage, adaptive=True)}
    # 16 diastereomers, none passing RDKit's embedding test: try all of them.
    assert len(search_isomers(Chem.MolFromSmiles("CC1C2C3NC3C12C#N"), adaptive=True)) == 16
