"""Observable features for the inverse model.

Each molecule is described by a fixed set of observable "slots". Every slot
becomes one encoder token; slots that are unknown in a given condition are
masked out of attention, so one model handles "A, B, C only", "+ formula",
"+ dipole" and combinations.

Slots:
    logA, logB, logC   log rotational constants (GHz), sorted A >= B >= C
    kappa              Ray's asymmetry parameter
    defect             asinh(inertial defect / amu A^2)
    mu                 dipole magnitude (Debye)
    nC, nH, nN, nO, nF element counts (the molecular formula)
    log_noise          log10 of the relative uncertainty on A, B, C

Noise is applied to the raw constants (multiplicative, per constant) and the
derived slots are recomputed from the noisy values, so they stay consistent.
"""

from __future__ import annotations

import torch

from rotml.physics.rotor import INERTIA_TO_MHZ

OBS_NAMES = ["logA", "logB", "logC", "kappa", "defect", "mu",
             "nC", "nH", "nN", "nO", "nF", "log_noise"]
ELEMENTS = ["C", "H", "N", "O", "F"]
N_OBS = len(OBS_NAMES)
ROTATIONAL = [0, 1, 2, 3, 4]
DIPOLE = [5]
FORMULA = [6, 7, 8, 9, 10]
NOISE = [11]

DIPOLE_NOISE_DEBYE = 0.1


def raw_features(abc: torch.Tensor, mu: torch.Tensor, counts: torch.Tensor,
                 noise_rel: torch.Tensor) -> torch.Tensor:
    """Unstandardized slot values, shape (B, N_OBS).

    abc: (B, 3) rotational constants in GHz (any order; sorted here).
    mu: (B,) dipole magnitude; counts: (B, 5) element counts; noise_rel: (B,).
    """
    abc = torch.sort(abc, dim=1, descending=True).values
    A, B, C = abc.unbind(dim=1)
    denom = A - C
    kappa = torch.where(denom.abs() > 1e-9 * A, (2 * B - A - C) / denom.clamp_min(1e-30),
                        torch.zeros_like(A)).clamp(-1, 1)
    inertia = INERTIA_TO_MHZ / (abc * 1000.0)  # I_a <= I_b <= I_c
    defect = torch.asinh(inertia[:, 2] - inertia[:, 0] - inertia[:, 1])
    cols = [torch.log(A), torch.log(B), torch.log(C), kappa, defect, mu,
            *counts.unbind(dim=1), torch.log10(noise_rel)]
    return torch.stack(cols, dim=1).float()


def perturb(abc: torch.Tensor, mu: torch.Tensor, noise_rel: torch.Tensor,
            generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Multiplicative log-normal noise on A, B, C; additive Gaussian noise on mu."""
    eps = torch.randn(abc.shape, generator=generator, dtype=abc.dtype)
    noisy_abc = abc * torch.exp(noise_rel[:, None] * eps)
    eps_mu = torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
    noisy_mu = (mu + DIPOLE_NOISE_DEBYE * eps_mu).clamp_min(0.0)
    return noisy_abc, noisy_mu


def condition_mask(batch: int, formula: bool, dipole: bool) -> torch.Tensor:
    """Boolean (B, N_OBS) mask of slots that are known in a fixed condition."""
    present = torch.zeros(batch, N_OBS, dtype=torch.bool)
    present[:, ROTATIONAL + NOISE] = True
    if formula:
        present[:, FORMULA] = True
    if dipole:
        present[:, DIPOLE] = True
    return present


def random_condition_mask(batch: int, p_formula: float, p_dipole: float,
                          generator: torch.Generator | None = None) -> torch.Tensor:
    """Per-sample random mix of conditions, for training one model on all of them."""
    present = condition_mask(batch, False, False)
    f = torch.rand(batch, generator=generator) < p_formula
    d = torch.rand(batch, generator=generator) < p_dipole
    present[:, FORMULA] = f[:, None]
    present[:, DIPOLE] = d[:, None]
    return present


CONDITIONS = {
    "ABC": (False, False),
    "ABC + formula": (True, False),
    "ABC + |mu|": (False, True),
    "ABC + formula + |mu|": (True, True),
}
