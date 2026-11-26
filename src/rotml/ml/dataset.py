"""Milestone 3 dataset: QM9 observables -> stereo-free SELFIES.

build_dataset() writes:
    m3_dataset.npz    A, B, C, mu, element counts, split labels, token ids
                      (n_variants SELFIES strings per molecule: canonical + randomized)
    m3_meta.json      vocabulary, max length, feature statistics, split sizes
    m3_molecules.csv  mol_id, canonical SMILES, formula, split

Splits are by molecular formula by default: every isomer of a test formula is
held out, so test molecules have formulas the model has never seen.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rotml.ml import features as F
from rotml.ml.tokens import Vocab, canonical_smiles, selfies_tokens, smiles_variants
from rotml.physics.rotor import is_linear

SPLITS = {"train": 0, "val": 1, "test": 2}


def element_counts(formula: str) -> list[int]:
    counts = dict.fromkeys(F.ELEMENTS, 0)
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if el not in counts:
            raise ValueError(f"unexpected element {el} in {formula}")
        counts[el] += int(n) if n else 1
    return [counts[e] for e in F.ELEMENTS]


def split_by_formula(formulas: pd.Series, val_frac: float, test_frac: float,
                     seed: int) -> np.ndarray:
    """Assign whole formula groups to test, then val, then train."""
    rng = np.random.default_rng(seed)
    sizes = formulas.value_counts()
    order = rng.permutation(sizes.index.to_numpy())
    n = len(formulas)
    assign, filled = {}, {"test": 0, "val": 0}
    for f in order:
        if filled["test"] < test_frac * n:
            assign[f] = "test"
            filled["test"] += sizes[f]
        elif filled["val"] < val_frac * n:
            assign[f] = "val"
            filled["val"] += sizes[f]
        else:
            assign[f] = "train"
    return formulas.map(assign).map(SPLITS).to_numpy()


def split_random(n: int, val_frac: float, test_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return np.where(u < test_frac, 2, np.where(u < test_frac + val_frac, 1, 0))


def build_dataset(qm9_csv: Path, out_dir: Path, n_variants: int = 5, split: str = "formula",
                  val_frac: float = 0.05, test_frac: float = 0.10, seed: int = 0) -> dict:
    qm9 = pd.read_csv(qm9_csv)
    keep = qm9.graph_ok & ~is_linear(qm9.A, qm9.B) & qm9.smiles.notna()
    qm9 = qm9[keep].reset_index(drop=True)
    qm9["canon"] = qm9.smiles.map(canonical_smiles)

    token_lists, ok = [], []
    for i, smi in enumerate(qm9.canon):
        variants = [t for t in (selfies_tokens(s) for s in smiles_variants(smi, n_variants, seed + i))
                    if t is not None] if smi else []
        token_lists.append(variants)
        ok.append(len(variants) > 0)
    qm9 = qm9[ok].reset_index(drop=True)
    token_lists = [t for t, keep_it in zip(token_lists, ok) if keep_it]

    alphabet = sorted({tok for variants in token_lists for toks in variants for tok in toks})
    vocab = Vocab(alphabet)
    max_len = max(len(toks) for variants in token_lists for toks in variants) + 2

    ids = np.full((len(qm9), n_variants, max_len), vocab.pad, dtype=np.int16)
    n_valid = np.zeros(len(qm9), dtype=np.int8)
    for i, variants in enumerate(token_lists):
        for j, toks in enumerate(variants[:n_variants]):
            ids[i, j] = vocab.encode(toks, max_len)
        n_valid[i] = min(len(variants), n_variants)

    if split == "formula":
        split_ids = split_by_formula(qm9.formula, val_frac, test_frac, seed)
    elif split == "random":
        split_ids = split_random(len(qm9), val_frac, test_frac, seed)
    else:
        raise ValueError(split)

    counts = np.array([element_counts(f) for f in qm9.formula], dtype=np.float32)
    abc = qm9[["A", "B", "C"]].to_numpy(np.float64)
    mu = qm9.mu.to_numpy(np.float64)

    # Feature statistics on clean training data (noise slot gets fixed stats).
    train = split_ids == 0
    raw = F.raw_features(torch.tensor(abc[train]), torch.tensor(mu[train]),
                         torch.tensor(counts[train], dtype=torch.float64),
                         torch.full((int(train.sum()),), 0.005, dtype=torch.float64))
    mean, std = raw.mean(0), raw.std(0).clamp_min(1e-6)
    mean[F.NOISE], std[F.NOISE] = -2.5, 0.5

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "m3_dataset.npz", abc=abc, mu=mu, counts=counts,
                        split=split_ids.astype(np.int8), tokens=ids, n_valid=n_valid)
    meta = {
        "vocab": vocab.itos, "max_len": max_len, "n_variants": n_variants, "split": split,
        "seed": seed, "obs_names": F.OBS_NAMES,
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "count_mean": counts[train].mean(0).tolist(),
        "count_std": counts[train].std(0).clip(1e-6).tolist(),
        "sizes": {k: int((split_ids == v).sum()) for k, v in SPLITS.items()},
        "n_formulas": {k: int(qm9.formula[split_ids == v].nunique()) for k, v in SPLITS.items()},
    }
    (out_dir / "m3_meta.json").write_text(json.dumps(meta, indent=1))
    mols = qm9[["mol_id", "canon", "formula"]].rename(columns={"canon": "smiles"})
    mols["split"] = pd.Series(split_ids).map({v: k for k, v in SPLITS.items()})
    mols.to_csv(out_dir / "m3_molecules.csv", index=False)
    return meta


@dataclass
class M3Data:
    abc: torch.Tensor        # (N, 3) float64
    mu: torch.Tensor         # (N,) float64
    counts: torch.Tensor     # (N, 5) float32
    split: torch.Tensor      # (N,) int8
    tokens: torch.Tensor     # (N, V, L) int64
    n_valid: torch.Tensor    # (N,) int64
    meta: dict
    smiles: list[str]

    @classmethod
    def load(cls, data_dir: Path) -> "M3Data":
        z = np.load(data_dir / "m3_dataset.npz")
        meta = json.loads((data_dir / "m3_meta.json").read_text())
        smiles = pd.read_csv(data_dir / "m3_molecules.csv").smiles.tolist()
        return cls(torch.from_numpy(z["abc"]), torch.from_numpy(z["mu"]),
                   torch.from_numpy(z["counts"]), torch.from_numpy(z["split"]),
                   torch.from_numpy(z["tokens"].astype(np.int64)),
                   torch.from_numpy(z["n_valid"].astype(np.int64)), meta, smiles)

    @property
    def vocab(self) -> Vocab:
        return Vocab.from_list(self.meta["vocab"])

    def indices(self, split: str) -> torch.Tensor:
        return torch.nonzero(self.split == SPLITS[split]).squeeze(1)

    def features(self, idx: torch.Tensor, noise_rel: torch.Tensor, present: torch.Tensor,
                 generator: torch.Generator | None = None, add_noise: bool = True
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardized slot values (B, N_OBS) and the presence mask."""
        abc, mu = self.abc[idx], self.mu[idx]
        if add_noise:
            abc, mu = F.perturb(abc, mu, noise_rel.double(), generator)
        return self.standardize(abc, mu, self.counts[idx], noise_rel, present)

    def standardize(self, abc: torch.Tensor, mu: torch.Tensor, counts: torch.Tensor,
                    noise_rel: torch.Tensor, present: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardized slot values from given (possibly already noisy) observables."""
        raw = F.raw_features(abc.double(), mu.double(), counts.double(), noise_rel.double())
        mean = torch.tensor(self.meta["feature_mean"])
        std = torch.tensor(self.meta["feature_std"])
        values = (raw - mean) / std
        return values.masked_fill(~present, 0.0), present

    def targets(self, idx: torch.Tensor, random_variant: bool,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """Token ids (B, L): a random SELFIES variant per molecule, or the canonical one."""
        if random_variant:
            u = torch.rand(len(idx), generator=generator)
            variant = (u * self.n_valid[idx]).long()
        else:
            variant = torch.zeros(len(idx), dtype=torch.long)
        return self.tokens[idx, variant]

    def count_targets(self, idx: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.meta["count_mean"])
        std = torch.tensor(self.meta["count_std"])
        return (self.counts[idx] - mean) / std
