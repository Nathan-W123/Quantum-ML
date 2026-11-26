"""SELFIES targets: tokenization, vocabulary and canonical comparison.

Targets are *stereo-free* (constitution only): enantiomers have identical
rotational spectra, so stereo labels would be partly unlearnable. Stereo and
conformers are a later step.
"""

from __future__ import annotations

import json

import selfies as sf
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIALS = [PAD, BOS, EOS]


def canonical_smiles(smiles: str | None) -> str | None:
    """Canonical, stereo-free SMILES, or None if RDKit cannot parse it."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def smiles_variants(smiles: str, n: int, seed: int) -> list[str]:
    """Canonical SMILES first, then up to n-1 distinct randomized SMILES."""
    canon = canonical_smiles(smiles)
    if canon is None:
        return []
    out = [canon]
    if n > 1:
        mol = Chem.MolFromSmiles(canon)
        for s in Chem.MolToRandomSmilesVect(mol, 4 * n, randomSeed=seed, isomericSmiles=False):
            if s not in out:
                out.append(s)
            if len(out) == n:
                break
    return out


def selfies_tokens(smiles: str) -> list[str] | None:
    try:
        return list(sf.split_selfies(sf.encoder(smiles)))
    except Exception:  # noqa: BLE001 - selfies raises EncoderError and others
        return None


def tokens_to_canonical(tokens: list[str]) -> str | None:
    """Decode SELFIES tokens to canonical stereo-free SMILES (None if invalid/empty)."""
    try:
        smiles = sf.decoder("".join(tokens))
    except Exception:  # noqa: BLE001
        return None
    return canonical_smiles(smiles)


class Vocab:
    def __init__(self, tokens: list[str]):
        self.itos = list(SPECIALS) + [t for t in tokens if t not in SPECIALS]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos = (self.stoi[t] for t in SPECIALS)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> list[int] | None:
        """[BOS] tokens [EOS] padded to max_len; None if too long or unknown token."""
        if len(tokens) + 2 > max_len or any(t not in self.stoi for t in tokens):
            return None
        ids = [self.bos] + [self.stoi[t] for t in tokens] + [self.eos]
        return ids + [self.pad] * (max_len - len(ids))

    def decode(self, ids) -> list[str]:
        """Token strings up to (not including) the first EOS; BOS/PAD skipped."""
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos:
                break
            if i in (self.bos, self.pad):
                continue
            out.append(self.itos[i])
        return out

    def to_json(self) -> str:
        return json.dumps(self.itos)

    @classmethod
    def from_list(cls, itos: list[str]) -> "Vocab":
        return cls([t for t in itos if t not in SPECIALS])
