# Research plan: structure determination from rotational spectra

## Goal

Infer the structure of an **unknown** molecule from its rotational or
rovibrational spectrum, in the low-information setting: molecular formula
optional, isotopologue data optional, starting from line lists, with every
answer checked by physics and reported with calibrated probabilities.

Flagship application: carriers of unidentified lines in radio-astronomical
surveys. Second application: unknown components of laboratory mixtures
(discharge, pyrolysis, reaction products), where results can be confirmed.

## Pipeline

```
spectrum -> [line grouping] -> constants -> Transformer (propose) -> GNN (screen)
         -> xtb -> DFT -> high-level QM + spectrum simulation -> ranked candidates + probabilities
```

| Stage | Role | Cost per candidate |
|---|---|---|
| Line grouping (astro only) | Cluster unassigned lines by carrier (velocity, width, excitation, pattern) | - |
| Spectrum -> constants | Line assignment and fitting (ML-assisted; SPFIT/PGOPHER) | - |
| Transformer (~10M params, supervised) | Inverse model: observables -> SELFIES; proposes ~1,000-5,000 candidates | ms |
| Equivariant GNN (ensemble of 5) | Forward surrogate: structure -> DFT-quality A, B, C, dipoles, hyperfine | ms |
| GFN2-xTB + conformer search | Cheap physics check | s-min |
| DFT | Accurate physics check | ~0.5-2 h |
| High-level QM + full spectrum simulation | Final ranking | h-day |
| Scoring | P(candidate) ~ prior x exp(-sum (obs - calc)^2 / 2 sigma^2), conformal candidate sets, "none of the above" flag | - |

RL (expert iteration with a physics reward) and 3D diffusion with
moment-of-inertia guidance come after the supervised baseline.

## Closest prior work

| Work | What it does | Gap it leaves |
|---|---|---|
| McCarthy & Lee, J. Phys. Chem. A 2020 (rotconML) | Constants -> formula, SMILES, functional groups; dropout uncertainty | No physics verification or calibrated ranking; closed-shell organics |
| Schwarting et al., J. Chem. Phys. 2024 ("Twins") | Rotational constants do not uniquely identify molecules at standard accuracy | Milestone 1 reproduces this on QM9 |
| Cheng et al., J. Chem. Phys. 2024 (KREED) | Diffusion: formula + moments of inertia + Kraitchman coordinates -> 3D structure | Needs formula and isotopologue data |
| Fried et al., arXiv 2608.18221 (2026) | Automated line assignment in GOTHAM / PILS; candidates from chemical-embedding similarity | Candidates not inferred from the unassigned lines' spectra |
| Automated Mixture Analysis via Structural Evaluation, arXiv 2408.15819 | Lab mixture analysis against database candidates | Known molecules only |
| SECS, NMIRacle, MMST, ... (2025-2026) | Multimodal NMR / IR / MS -> structure | No rotational spectroscopy |

## Milestones

1. **Identifiability on QM9** - how many molecules share A, B, C within a tolerance (done; `results/m1`).
2. **Cheap-method error** - MMFF94 and GFN2-xTB vs DFT, sets funnel tolerances (done; `results/m2`).
   GFN2-xTB needs ~7% tolerance for 95% recall on the right conformer; conformer
   coverage, not method error, dominates for flexible molecules.
3. **Baseline inverse model** - constants -> SELFIES transformer, formula split, recall@K, with/without formula; compare to rotconML.
   A 3.8M-parameter CPU run (`results/m3_cpu_medium`) reaches 43% top-10 recall on unseen formulas when
   given the formula (random isomers: 4.4%), and a decoy-constants control confirms it uses the constants.
   Without the formula: 3% top-10, because it rarely produces an unseen formula. A separate formula
   model (`results/m3_formula`) helps only a little (top-100 recall 2% -> 6%): unseen formulas are hard
   to infer from A, B, C.
   The 9.65M-parameter run needs a GPU (`notebooks/m3_train_colab.ipynb`).
4. **KREED baseline** - reproduce on QM9, retrain with fewer/no isotopologue coordinates. (Removing the formula requires changing the model: that is the research contribution.)
5. **Full pipeline** - transformer proposals -> physics scoring -> calibrated ranking; test on experimental CDMS/JPL constants.
   First prototype (`results/m5_rerank`): GFN2-xTB re-ranking of the top 30 candidates lifts top-1 recall
   from 11% to 26.5% (formula given); exact DFT constants would give 47%, against a 50.5% ceiling.

Then: astro chemical space (radicals, ions, carbon chains), raw line lists,
line grouping, retrospective rediscovery of detected interstellar molecules.
