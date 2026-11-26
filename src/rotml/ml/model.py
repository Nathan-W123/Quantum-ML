"""Encoder-decoder transformer: observable tokens -> SELFIES tokens.

Encoder: one token per observable slot (rotational constants, derived shape
parameters, dipole, element counts, noise level). A token is
    MLP(Fourier features of its standardized value) + embedding of its slot type.
Unknown slots are masked out of attention, so one model serves every
information condition.

Decoder: standard causal transformer over SELFIES tokens with cross-attention
to the observable tokens; output projection tied to the token embedding.

Auxiliary head: predicts the element counts from the pooled encoder output
(a cheap multi-task signal that helps the encoder learn composition).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    max_len: int
    n_obs: int
    d_model: int = 256
    n_heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1
    n_freq: int = 32
    n_aux: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


class FourierValueEmbedding(nn.Module):
    """Scalar -> d_model via fixed log-spaced Fourier features and an MLP.

    High frequencies let the model resolve small relative differences in the
    rotational constants (0.1% in log B is ~0.003 standard deviations).
    """

    def __init__(self, d_model: int, n_freq: int):
        super().__init__()
        self.register_buffer("freqs", torch.logspace(math.log10(0.25), math.log10(256.0), n_freq))
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq + 1, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))

    def forward(self, v: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, d)
        x = 2 * math.pi * v[..., None] * self.freqs
        return self.mlp(torch.cat([v[..., None], torch.sin(x), torch.cos(x)], dim=-1))


class Seq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        d = cfg.d_model
        self.value_emb = FourierValueEmbedding(d, cfg.n_freq)
        self.slot_emb = nn.Embedding(cfg.n_obs, d)
        enc_layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.ffn_dim, cfg.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.enc_layers, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.tok_emb = nn.Embedding(cfg.vocab_size, d, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, d)
        dec_layer = nn.TransformerDecoderLayer(d, cfg.n_heads, cfg.ffn_dim, cfg.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_layers, norm=nn.LayerNorm(d))
        self.aux_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.n_aux))
        nn.init.normal_(self.tok_emb.weight, std=d**-0.5)

    # --- encoder -----------------------------------------------------------
    def encode(self, values: torch.Tensor, present: torch.Tensor):
        """values, present: (B, n_obs). Returns memory (B, n_obs, d) and its padding mask."""
        slots = torch.arange(values.shape[1], device=values.device)
        x = self.value_emb(values) + self.slot_emb(slots)[None]
        mem_mask = ~present
        memory = self.encoder(x, src_key_padding_mask=mem_mask)
        return memory, mem_mask

    def aux(self, memory: torch.Tensor, mem_mask: torch.Tensor) -> torch.Tensor:
        keep = (~mem_mask).unsqueeze(-1).to(memory.dtype)
        pooled = (memory * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        return self.aux_head(pooled)

    # --- decoder -----------------------------------------------------------
    def decode(self, tokens: torch.Tensor, memory: torch.Tensor, mem_mask: torch.Tensor):
        """tokens: (B, L) starting with BOS. Returns logits (B, L, vocab)."""
        L = tokens.shape[1]
        pos = torch.arange(L, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1)
        h = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tokens == self.pad_id,
                         memory_key_padding_mask=mem_mask, tgt_is_causal=True)
        return h @ self.tok_emb.weight.T

    def forward(self, values, present, tokens_in):
        memory, mem_mask = self.encode(values, present)
        return self.decode(tokens_in, memory, mem_mask), self.aux(memory, mem_mask)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
