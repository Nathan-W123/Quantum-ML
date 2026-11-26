"""Formula proposals from rotational constants, for when the formula is unknown.

The inverse transformer does well when it is given the formula, but without
it it rarely writes an unseen formula. This small model proposes formulas
instead, so the transformer can be run once per proposed formula.

It factorizes over elements:
    p(nC, nN, nO, nF, nH | obs) = p(nC | obs) p(nN | obs, nC) ... p(nH | obs, nC, nN, nO, nF)
Each factor is a categorical over counts. The formula split holds out whole
formulas, not element counts, so this can compose formulas never seen in
training. Beam search over the five factors, keeping only valid formulas,
gives a ranked list with log-probabilities.
"""

from __future__ import annotations

import torch
import torch.nn.functional as nnf
from torch import nn

from rotml.ml import features as F
from rotml.ml.model import FourierValueEmbedding

ORDER = ["C", "N", "O", "F", "H"]  # H last: valence and parity constrain it most
MAX_COUNT = {"C": 9, "N": 7, "O": 5, "F": 6, "H": 20}
INPUT_SLOTS = F.ROTATIONAL + F.DIPOLE + F.NOISE
_POS = [F.ELEMENTS.index(e) for e in ORDER]  # ORDER -> column in F.ELEMENTS order


def is_valid_formula(counts: tuple[int, ...]) -> bool:
    """1-9 heavy atoms and a non-negative integer degree of unsaturation (F.ELEMENTS order)."""
    c, h, n, o, f = counts
    heavy = c + n + o + f
    twice_dou = 2 * c + 2 + n - h - f
    return 1 <= heavy <= 9 and twice_dou >= 0 and twice_dou % 2 == 0


def formula_string(counts: tuple[int, ...]) -> str:
    """Formula string in the dataset's convention (rotml.data.qm9.hill_formula: C, H, then alphabetical)."""
    d = dict(zip(F.ELEMENTS, counts))
    return "".join(f"{e}{d[e] if d[e] > 1 else ''}" for e in ("C", "H", "F", "N", "O") if d[e])


class FormulaNet(nn.Module):
    def __init__(self, d: int = 256, n_layers: int = 3, n_freq: int = 32, d_slot: int = 64,
                 dropout: float = 0.1, autoregressive: bool = True, use_noise_slot: bool = True):
        super().__init__()
        self.config = {"d": d, "n_layers": n_layers, "n_freq": n_freq, "d_slot": d_slot,
                       "dropout": dropout, "autoregressive": autoregressive,
                       "use_noise_slot": use_noise_slot}
        self.autoregressive = autoregressive
        self.use_noise_slot = use_noise_slot
        n = len(INPUT_SLOTS)
        self.value_emb = FourierValueEmbedding(d_slot, n_freq)
        self.slot_emb = nn.Parameter(torch.zeros(n, d_slot))
        self.inp = nn.Linear(n * d_slot + n, d)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(2 * d, d)) for _ in range(n_layers))
        self.count_emb = nn.ModuleList(nn.Embedding(MAX_COUNT[e] + 1, d) for e in ORDER)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, MAX_COUNT[e] + 1))
            for e in ORDER)

    def trunk(self, values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        v, p = values[:, INPUT_SLOTS], present[:, INPUT_SLOTS]
        if not self.use_noise_slot:  # trained at a fixed noise range; ignore the stated noise
            p = p.clone()
            p[:, INPUT_SLOTS.index(F.NOISE[0])] = False
        x = (self.value_emb(v) + self.slot_emb) * p[..., None]
        h = self.inp(torch.cat([x.flatten(1), p.float()], dim=1))
        for block in self.blocks:
            h = h + block(h)
        return h

    def forward(self, values: torch.Tensor, present: torch.Tensor, counts: torch.Tensor
                ) -> list[torch.Tensor]:
        """Teacher-forced logits per element (in ORDER); counts are in F.ELEMENTS order."""
        h = self.trunk(values, present)
        logits = []
        for j, e in enumerate(ORDER):
            logits.append(self.heads[j](h))
            if self.autoregressive:
                h = h + self.count_emb[j](counts[:, _POS[j]].long().clamp(0, MAX_COUNT[e]))
        return logits

    def nll(self, values, present, counts) -> torch.Tensor:
        """Mean negative log-likelihood of the true formula (nats per molecule)."""
        logits = self(values, present, counts)
        return sum(nnf.cross_entropy(lg, counts[:, _POS[j]].long().clamp(0, MAX_COUNT[e]))
                   for j, (lg, e) in enumerate(zip(logits, ORDER)))

    @torch.no_grad()
    def propose(self, values: torch.Tensor, present: torch.Tensor, top: int = 20, beam: int = 256
                ) -> list[list[tuple[tuple[int, ...], float]]]:
        """Top valid formulas per input: [(counts in F.ELEMENTS order, log-prob), ...], best first."""
        self.eval()
        h = self.trunk(values, present)[:, None]  # (B, k, d)
        B = h.shape[0]
        score = torch.zeros(B, 1, device=h.device)
        counts = torch.zeros(B, 1, 0, dtype=torch.long, device=h.device)
        for j in range(len(ORDER)):
            lp = torch.log_softmax(self.heads[j](h), dim=-1)  # (B, k, V)
            V = lp.shape[-1]
            cand = (score[..., None] + lp).view(B, -1)
            score, flat = cand.topk(min(beam, cand.shape[1]), dim=1)
            src, n = flat // V, flat % V
            h = torch.gather(h, 1, src[..., None].expand(-1, -1, h.shape[-1]))
            if self.autoregressive:
                h = h + self.count_emb[j](n)
            counts = torch.cat([torch.gather(counts, 1, src[..., None].expand(-1, -1, counts.shape[-1])),
                                n[..., None]], dim=-1)
        out = []
        for c_b, s_b in zip(counts.tolist(), score.tolist()):
            kept = []
            for c, s in zip(c_b, s_b):
                by_el = dict(zip(ORDER, c))
                tup = tuple(by_el[e] for e in F.ELEMENTS)
                if is_valid_formula(tup):
                    kept.append((tup, s))
                    if len(kept) == top:
                        break
            out.append(kept)
        return out
