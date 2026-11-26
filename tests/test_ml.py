import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("selfies")

from rotml.ml import features as F  # noqa: E402
from rotml.ml.dataset import element_counts, split_by_formula  # noqa: E402
from rotml.ml.decode import beam_search, generate_candidates  # noqa: E402
from rotml.ml.metrics import library_lookup_ranks, recall_at_k  # noqa: E402
from rotml.ml.model import ModelConfig, Seq2Seq, count_parameters  # noqa: E402
from rotml.ml.tokens import Vocab, canonical_smiles, selfies_tokens, smiles_variants, tokens_to_canonical  # noqa: E402,E501


def test_selfies_round_trip_and_variants():
    for smi in ["CCO", "c1ccccc1O", "N#CC(=O)N1CC1C=O", "O=[N+]([O-])C"]:
        canon = canonical_smiles(smi)
        for variant in smiles_variants(smi, 5, seed=1):
            assert tokens_to_canonical(selfies_tokens(variant)) == canon


def test_vocab_encode_decode():
    toks = selfies_tokens("CCO")
    vocab = Vocab(sorted(set(toks)))
    ids = vocab.encode(toks, max_len=10)
    assert len(ids) == 10 and ids[0] == vocab.bos
    assert vocab.decode(ids[1:]) == toks
    assert vocab.encode(toks, max_len=3) is None


def test_element_counts():
    assert element_counts("C6H13NO2") == [6, 13, 1, 2, 0]
    assert element_counts("H3N") == [0, 3, 1, 0, 0]
    assert element_counts("CHF3") == [1, 1, 0, 0, 3]


def test_raw_features_sort_and_shape_parameters():
    abc = torch.tensor([[1.0, 5.0, 1.0], [10.0, 10.0, 2.0]], dtype=torch.float64)
    raw = F.raw_features(abc, torch.zeros(2, dtype=torch.float64),
                         torch.zeros(2, 5, dtype=torch.float64), torch.full((2,), 0.01))
    assert raw[0, 0] == pytest.approx(np.log(5.0))           # sorted: A is the largest
    assert raw[0, F.OBS_NAMES.index("kappa")] == pytest.approx(-1.0)  # prolate
    assert raw[1, F.OBS_NAMES.index("kappa")] == pytest.approx(1.0)   # oblate
    assert raw[0, F.OBS_NAMES.index("log_noise")] == pytest.approx(-2.0)


def test_split_by_formula_has_no_shared_formulas():
    formulas = pd.Series(np.random.default_rng(0).choice([f"C{i}H4" for i in range(60)], 3000))
    split = split_by_formula(formulas, val_frac=0.1, test_frac=0.2, seed=0)
    groups = [set(formulas[split == s]) for s in (0, 1, 2)]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    assert 0.15 < np.mean(split == 2) < 0.35


def _tiny_model(vocab_size, max_len=12):
    cfg = ModelConfig(vocab_size=vocab_size, max_len=max_len, n_obs=F.N_OBS, d_model=32,
                      n_heads=4, enc_layers=1, dec_layers=1, ffn_dim=64, dropout=0.0, n_freq=8)
    return Seq2Seq(cfg, pad_id=0)


def test_default_config_is_about_ten_million_parameters():
    cfg = ModelConfig(vocab_size=53, max_len=29, n_obs=F.N_OBS)
    assert 9.0e6 < count_parameters(Seq2Seq(cfg)) < 10.5e6


def test_masked_slots_do_not_affect_output():
    torch.manual_seed(0)
    model = _tiny_model(10).eval()
    values = torch.randn(2, F.N_OBS)
    present = F.condition_mask(2, formula=False, dipole=False)
    tokens = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])
    out1, _ = model(values, present, tokens)
    changed = values.clone()
    changed[:, F.FORMULA + F.DIPOLE] += 5.0  # only masked slots change
    out2, _ = model(changed, present, tokens)
    assert torch.allclose(out1, out2, atol=1e-5)
    present_f = F.condition_mask(2, formula=True, dipole=False)
    out3, _ = model(changed, present_f, tokens)
    assert not torch.allclose(out1, out3, atol=1e-3)


def test_overfit_single_example_and_beam_search_recovers_it():
    torch.manual_seed(0)
    toks = selfies_tokens("CC(=O)NC1CC1")
    vocab = Vocab(sorted(set(toks)))
    target = torch.tensor([vocab.encode(toks, max_len=16)])
    model = _tiny_model(len(vocab), max_len=16)
    values = torch.randn(1, F.N_OBS)
    present = F.condition_mask(1, formula=True, dipole=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        logits, _ = model(values, present, target[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), target[:, 1:].reshape(-1),
                                                 ignore_index=vocab.pad)
        opt.zero_grad()
        loss.backward()
        opt.step()
    seqs, scores = beam_search(model, values, present, vocab, beam=3)
    assert scores[0, 0] >= scores[0, 1]
    ranked = generate_candidates(model, values, present, vocab, beam=3, n_samples=5)
    assert ranked[0][0][0] == canonical_smiles("CC(=O)NC1CC1")
    assert recall_at_k(ranked, [canonical_smiles("CC(=O)NC1CC1")], ks=(1,))["recall@1"] == 1.0


def test_library_lookup_ranks_true_molecule_first_without_noise():
    rng = np.random.default_rng(0)
    lib_abc = torch.tensor(rng.uniform(1, 5, size=(200, 3)))
    lib_mu = torch.tensor(rng.uniform(0, 4, size=200))
    formulas = np.array(["C2H6O"] * 100 + ["C3H8"] * 100)
    q = torch.arange(0, 200, 7)
    ranks = library_lookup_ranks(lib_abc[q], lib_mu[q], q, lib_abc, lib_mu, formulas,
                                 sigma_rel=0.01, use_formula=True, use_dipole=True)
    assert np.all(ranks == 1)
