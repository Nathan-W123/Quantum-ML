"""Learned correction of cheap geometries toward DFT: the pipeline's GNN verifier.

GFN2-xTB geometries are close to B3LYP ones but not close enough: their
rotational constants are off by ~1% (milestone 2), so wrong candidates often
fit the observed constants as well as the right one (milestone 5 prototype).
This network takes the atoms and an xTB geometry and predicts the geometry DFT
would give for the same conformer. Rotational constants then follow exactly
from the corrected geometry.

Design: SchNet-style continuous-filter message passing on interatomic
distances, then an EGNN-style coordinate update, where each atom moves along
the directions to its neighbours by learned scalar amounts. The update is
rotation- and translation-equivariant, so the constants are invariant by
construction. The last layer starts at zero, so training starts from "no
correction".
"""

from __future__ import annotations

import torch
import torch.nn.functional as nnf
from torch import nn

from rotml.physics.rotor import INERTIA_TO_MHZ, ISOTOPE_MASS

SYMBOL_OF_Z = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
Z_OF_SYMBOL = {s: z for z, s in SYMBOL_OF_Z.items()}
_MASS_BY_Z = torch.zeros(10, dtype=torch.float64)
for _z, _s in SYMBOL_OF_Z.items():
    _MASS_BY_Z[_z] = ISOTOPE_MASS[_s]


def masses(z: torch.Tensor) -> torch.Tensor:
    """Isotopic masses (amu) from atomic numbers; 0 for padding (z = 0)."""
    return _MASS_BY_Z.to(z.device)[z]


def log_constants(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """log of (A, B, C) in GHz, A >= B >= C, for (B, N, 3) coordinates and (B, N) masses.

    Differentiable; padded atoms must have mass 0.
    """
    x = x.double()
    m = m.double()
    com = (m[..., None] * x).sum(1, keepdim=True) / m.sum(1)[:, None, None]
    r = x - com
    r2 = (r * r).sum(-1)
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    inertia = (m[..., None, None] * (r2[..., None, None] * eye - r[..., :, None] * r[..., None, :])).sum(1)
    moments = torch.linalg.eigvalsh(inertia).clamp_min(1e-6)  # ascending: I_a <= I_b <= I_c
    return torch.log(torch.tensor(INERTIA_TO_MHZ / 1000.0, dtype=x.dtype)) - torch.log(moments)


class GeometryCorrector(nn.Module):
    def __init__(self, d: int = 128, n_layers: int = 3, n_rbf: int = 32, cutoff: float = 5.0):
        super().__init__()
        self.config = {"d": d, "n_layers": n_layers, "n_rbf": n_rbf, "cutoff": cutoff}
        self.cutoff = cutoff
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2
        self.emb = nn.Embedding(10, d, padding_idx=0)
        self.filters = nn.ModuleList(
            nn.Sequential(nn.Linear(n_rbf, d), nn.SiLU(), nn.Linear(d, d)) for _ in range(n_layers))
        self.lin_in = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n_layers))
        self.lin_out = nn.ModuleList(
            nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)) for _ in range(n_layers))
        self.coord = nn.Sequential(nn.Linear(2 * d + n_rbf, d), nn.SiLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.coord[-1].weight)
        nn.init.zeros_(self.coord[-1].bias)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Corrected coordinates (B, N, 3) for atomic numbers z (B, N; 0 = padding)."""
        mask = z > 0
        rel = x[:, :, None, :] - x[:, None, :, :]  # r_i - r_j
        dist = rel.norm(dim=-1)
        n = z.shape[1]
        pair = mask[:, :, None] & mask[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=z.device)
        fcut = 0.5 * (torch.cos(torch.pi * dist / self.cutoff) + 1.0) * (dist < self.cutoff) * pair
        rbf = torch.exp(-self.gamma * (dist[..., None] - self.centers) ** 2)
        h = self.emb(z)
        for filt, lin_in, lin_out in zip(self.filters, self.lin_in, self.lin_out):
            w = filt(rbf) * fcut[..., None]
            h = h + lin_out(torch.einsum("bijd,bjd->bid", w, lin_in(h)))
        hi = h[:, :, None, :].expand(-1, -1, n, -1)
        hj = h[:, None, :, :].expand(-1, n, -1, -1)
        step = self.coord(torch.cat([hi, hj, rbf], dim=-1)).squeeze(-1) * fcut  # (B, N, N), Angstrom
        unit = rel / dist.clamp_min(1e-6)[..., None]
        return x + (step[..., None] * unit).sum(2) * mask[..., None]


def corrector_loss(model: GeometryCorrector, z, x_cheap, x_ref, ref_abc, bond_cutoff: float = 3.0):
    """(total, constants MSE in log units, distance MSE in Angstrom^2) for one batch."""
    x_pred = model(z, x_cheap)
    m = masses(z)
    l_abc = nnf.mse_loss(log_constants(x_pred, m), torch.log(ref_abc.double())).float()
    mask = z > 0
    pair = mask[:, :, None] & mask[:, None, :]
    d_ref = torch.cdist(x_ref, x_ref)
    near = pair & (d_ref < bond_cutoff) & (d_ref > 0)
    l_dist = ((torch.cdist(x_pred, x_pred) - d_ref)[near] ** 2).mean()
    return l_abc + l_dist, l_abc, l_dist
