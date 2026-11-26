"""Candidate generation: beam search and sampling, merged into ranked lists.

Candidates are deduplicated by canonical stereo-free SMILES (many SELFIES
strings decode to the same molecule) and ranked by the model's sequence
log-probability. The decoder is re-run on the full prefix each step; with
SELFIES of at most ~30 tokens that is cheap enough not to need a KV cache.
"""

from __future__ import annotations

import torch

from rotml.ml.model import Seq2Seq
from rotml.ml.tokens import Vocab, tokens_to_canonical


@torch.no_grad()
def beam_search(model: Seq2Seq, values: torch.Tensor, present: torch.Tensor, vocab: Vocab,
                beam: int = 10, max_len: int | None = None):
    """Returns token ids (B, beam, L) and log-probabilities (B, beam), best first."""
    model.eval()
    max_len = max_len or model.cfg.max_len
    B = values.shape[0]
    memory, mem_mask = model.encode(values, present)
    memory = memory.repeat_interleave(beam, 0)
    mem_mask = mem_mask.repeat_interleave(beam, 0)
    dev = values.device

    seqs = torch.full((B * beam, 1), vocab.bos, dtype=torch.long, device=dev)
    scores = torch.full((B, beam), float("-inf"), device=dev)
    scores[:, 0] = 0.0
    finished = torch.zeros(B * beam, dtype=torch.bool, device=dev)
    V = model.cfg.vocab_size
    for _ in range(max_len - 1):
        logits = model.decode(seqs, memory, mem_mask)[:, -1].float()
        logits[:, [vocab.pad, vocab.bos]] = float("-inf")  # never generated
        logp = torch.log_softmax(logits, dim=-1)
        # Finished beams only extend with PAD at no cost.
        logp[finished] = float("-inf")
        logp[finished, vocab.pad] = 0.0
        cand = (scores.view(-1, 1) + logp).view(B, beam * V)
        scores, flat = cand.topk(beam, dim=1)
        src = (flat // V) + torch.arange(B, device=dev)[:, None] * beam
        tok = (flat % V).view(-1)
        seqs = torch.cat([seqs[src.view(-1)], tok[:, None]], dim=1)
        finished = finished[src.view(-1)] | (tok == vocab.eos)
        if finished.all():
            break
    return seqs.view(B, beam, -1), scores


@torch.no_grad()
def sample(model: Seq2Seq, values: torch.Tensor, present: torch.Tensor, vocab: Vocab,
           n_samples: int = 100, temperature: float = 1.0, max_len: int | None = None,
           generator: torch.Generator | None = None):
    """Returns token ids (B, n, L) and model log-probabilities (B, n) at temperature 1."""
    model.eval()
    max_len = max_len or model.cfg.max_len
    B = values.shape[0]
    memory, mem_mask = model.encode(values, present)
    memory = memory.repeat_interleave(n_samples, 0)
    mem_mask = mem_mask.repeat_interleave(n_samples, 0)
    dev = values.device
    seqs = torch.full((B * n_samples, 1), vocab.bos, dtype=torch.long, device=dev)
    logprob = torch.zeros(B * n_samples, device=dev)
    finished = torch.zeros(B * n_samples, dtype=torch.bool, device=dev)
    for _ in range(max_len - 1):
        logits = model.decode(seqs, memory, mem_mask)[:, -1].float()
        logits[:, [vocab.pad, vocab.bos]] = float("-inf")  # never generated
        logp = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        tok = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        tok = torch.where(finished, torch.full_like(tok, vocab.pad), tok)
        logprob = logprob + torch.where(finished, torch.zeros_like(logprob),
                                        logp.gather(1, tok[:, None]).squeeze(1))
        seqs = torch.cat([seqs, tok[:, None]], dim=1)
        finished = finished | (tok == vocab.eos)
        if finished.all():
            break
    return seqs.view(B, n_samples, -1), logprob.view(B, n_samples)


def rank_candidates(seq_groups, score_groups, vocab: Vocab) -> list[list[tuple[str, float]]]:
    """Merge token sequences into per-input lists of (canonical SMILES, best log-prob)."""
    ranked = []
    for seqs, scores in zip(seq_groups, score_groups):
        best: dict[str, float] = {}
        for ids, s in zip(seqs, scores):
            s = float(s)
            if s == float("-inf"):
                continue
            smi = tokens_to_canonical(vocab.decode(ids))
            if smi and (smi not in best or s > best[smi]):
                best[smi] = s
        ranked.append(sorted(best.items(), key=lambda kv: -kv[1]))
    return ranked


@torch.no_grad()
def generate_candidates(model: Seq2Seq, values: torch.Tensor, present: torch.Tensor,
                        vocab: Vocab, beam: int = 10, n_samples: int = 0,
                        temperature: float = 1.0, generator: torch.Generator | None = None):
    """Ranked candidate lists from beam search plus (optionally) sampling."""
    b_seqs, b_scores = beam_search(model, values, present, vocab, beam=beam)
    groups = [list(zip(b_seqs[i].cpu(), b_scores[i].cpu())) for i in range(values.shape[0])]
    if n_samples:
        s_seqs, s_scores = sample(model, values, present, vocab, n_samples, temperature,
                                  generator=generator)
        for i in range(values.shape[0]):
            groups[i] += list(zip(s_seqs[i].cpu(), s_scores[i].cpu()))
    return rank_candidates([[g[0] for g in grp] for grp in groups],
                           [[g[1] for g in grp] for grp in groups], vocab)
