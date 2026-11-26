"""Evaluation metrics and the library-lookup reference."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from rotml.ml.features import DIPOLE_NOISE_DEBYE


def recall_at_k(ranked: list[list[tuple[str, float]]], targets: list[str],
                ks=(1, 5, 10, 50, 100)) -> dict[str, float]:
    """Fraction of inputs whose target SMILES is among the top-k candidates."""
    out = {}
    for k in ks:
        hits = [t in {smi for smi, _ in cands[:k]} for cands, t in zip(ranked, targets)]
        out[f"recall@{k}"] = float(np.mean(hits))
    return out


def formula_of(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else None


def top1_formula_accuracy(ranked, targets) -> float:
    hits = []
    for cands, t in zip(ranked, targets):
        hits.append(bool(cands) and formula_of(cands[0][0]) == formula_of(t))
    return float(np.mean(hits))


@torch.no_grad()
def library_lookup_ranks(query_abc: torch.Tensor, query_mu: torch.Tensor, query_idx: torch.Tensor,
                         lib_abc: torch.Tensor, lib_mu: torch.Tensor, lib_formula: np.ndarray,
                         sigma_rel: float, use_formula: bool, use_dipole: bool,
                         chunk: int = 256) -> np.ndarray:
    """Rank of the true molecule when the whole library is sorted by chi-square fit.

    This is a reference, not a competitor: it assumes a candidate library that
    contains the answer, with exact DFT constants for every entry. Its recall
    is close to the best any method can do at this noise level.
    """
    q_log = torch.log(torch.sort(query_abc, 1, descending=True).values)
    l_log = torch.log(torch.sort(lib_abc, 1, descending=True).values)
    ranks = np.empty(len(query_idx), dtype=np.int64)
    codes = {f: i for i, f in enumerate(np.unique(lib_formula))}
    lib_codes = torch.tensor([codes[f] for f in lib_formula])
    for s in range(0, len(query_idx), chunk):
        sl = slice(s, s + chunk)
        chi2 = (((q_log[sl, None, :] - l_log[None]) / sigma_rel) ** 2).sum(-1)
        if use_dipole:
            chi2 = chi2 + ((query_mu[sl, None] - lib_mu[None]) / DIPOLE_NOISE_DEBYE) ** 2
        if use_formula:
            same = lib_codes[None] == lib_codes[query_idx[sl]][:, None]
            chi2 = chi2.masked_fill(~same, float("inf"))
        true = chi2.gather(1, query_idx[sl, None])
        ranks[sl] = ((chi2 < true).sum(1) + 1).numpy()
    return ranks
