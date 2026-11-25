# Quantum-ML: molecular structure from rotational spectra

Research code for inferring the structure of **unknown** molecules from their
rotational / rovibrational spectra: an ML model proposes candidate structures,
and a funnel of increasingly accurate physics (force field -> GFN2-xTB -> DFT)
checks and ranks them. Applications: unidentified lines in radio-astronomical
surveys and unknown components of laboratory mixtures.

See [`docs/research_plan.md`](docs/research_plan.md) for the full plan, the
pipeline design and the closest prior work.

## Status

| Milestone | Status |
|---|---|
| 1. Identifiability of QM9 molecules from rotational constants | done |
| 2. Error of cheap geometry methods (MMFF94, GFN2-xTB) vs DFT | done |
| 3. Baseline inverse model (constants -> SELFIES transformer) | next (needs a GPU) |
| 4. KREED baseline and ablations | planned |
| 5. Full propose -> verify -> rank pipeline | planned |

## Milestone 1: how much do rotational constants pin down a molecule?

For each of the 133,885 QM9 molecules (B3LYP/6-31G(2df,p)), count how many
*other* QM9 molecules have A, B and C all within a relative tolerance.

Percentage of molecules **uniquely identified**, with the 90th percentile of
the number of other molecules that also match in brackets:

| Known observables | 0.1% | 0.5% | 1% | 2% | 5% |
|---|---|---|---|---|---|
| A, B, C | 98.6% (0) | 57.1% (2) | 12.0% (11) | 1.4% (66) | 0.1% (829) |
| A, B, C + formula | 99.5% (0) | 93.3% (0) | 75.4% (1) | 37.6% (5) | 7.2% (47) |
| A, B, C + dipole magnitude (±0.2 D) | 99.6% (0) | 93.7% (0) | 69.6% (1) | 23.1% (7) | 2.3% (80) |
| A, B, C + formula + dipole | 99.6% (0) | 98.7% (0) | 94.8% (0) | 79.7% (1) | 37.5% (7) |

![Uniquely identified vs tolerance](results/m1/unique_vs_tolerance.png)

Findings:

- **The accuracy of the theory is the bottleneck.** At 0.1% (best composite
  methods with vibrational corrections), rotational constants alone identify
  98.6% of QM9 molecules. At 1% (typical DFT) they identify only 12%.
- **Knowing the formula buys roughly a factor of 2-4 in tolerance**: with the
  formula, 1% tolerance identifies 75% of molecules; without it you need a
  tolerance between 0.25% (90%) and 0.5% (57%).
- **One extra observable helps nearly as much as the formula.** The dipole
  magnitude alone takes 1%-tolerance identification from 12% to 70%; formula
  and dipole together reach 95%.
- **Ambiguity grows quickly with size** (`results/m1/unique_vs_size.png`): at 1%
  with A, B, C only, 63% of 7-heavy-atom molecules are unique but only 8% of
  9-heavy-atom ones.

This agrees with Schwarting et al. (J. Chem. Phys. 2024, "Twins in rotational
spectroscopy"). Caveats: QM9 is a small, filtered corner of chemical space
with one conformer per molecule, so real unknowns face **more** look-alikes
than these counts suggest; and both "observed" and candidate constants come
from the same method here, so the tolerance stands in for theory-vs-experiment
error.

## Milestone 2: how accurate are the cheap funnel stages?

The first funnel stages use cheap methods, and a stage can only discard
candidates as aggressively as that method's error allows. For 2,000 random
QM9 molecules, A, B, C were recomputed with MMFF94 (RDKit) and GFN2-xTB
(tblite) and compared with the DFT values, in two settings:

- **Same conformer:** relax starting from the DFT geometry. This isolates the
  method's own error.
- **Structure search:** start from the molecular graph only, embed 10 ETKDG
  conformers, relax them, and take the best-matching (or lowest-energy) one.
  This is what a funnel stage actually faces.

The table gives the tolerance needed so that 95% of molecules still match.
The 90% and 99% columns are in `results/m2/error_summary.csv`; the GFN2-xTB
bias has been removed.

| Method | Setting | Median error per constant | Tolerance for 95% recall | Time per molecule |
|---|---|---|---|---|
| MMFF94 | same conformer | 1.8% | 14.1% | 2 ms |
| GFN2-xTB | same conformer | 1.6% (bias +1.1 to +1.7%) | **6.8%** | 0.3 s |
| MMFF94 | structure search, best match | 1.8% | 32% | 0.24 s |
| GFN2-xTB | structure search, best match | 1.7% | 30% | 1.8 s |
| either | structure search, lowest energy | 2.6–3.2% | ~90% | — |

![Recall vs tolerance](results/m2/recall_vs_tolerance.png)

**Run at its 95%-recall tolerance, each stage keeps this many QM9 candidates**
(median over all molecules, from the milestone 1 counts):

| Stage | Tolerance | A, B, C only | + formula |
|---|---|---|---|
| GFN2-xTB, right conformer | 6.8% | 798 (0.6% of QM9) | 23 |
| MMFF94, right conformer | 14.1% | 5,401 (4%) | 117 |
| GFN2-xTB, structure search | 30% | 27,781 (21%) | 461 |
| DFT-level, for comparison (M1) | 1% | 5 | 1 |

**Finding the right conformer is the main source of error, more than the
method's accuracy** (`results/m2/by_rotatable_bonds.csv`):

| Rotatable bonds | Molecules | GFN2-xTB, right conformer | GFN2-xTB, structure search |
|---|---|---|---|
| 0 | 870 | 6.0% | 6.5% |
| 1 | 579 | 7.4% | 15.7% |
| 2 | 328 | 9.4% | 22.1% |
| 3+ | 223 | 9.1% | 31.9% |

Findings:

- **GFN2-xTB beats MMFF94 by about 2x at 95% recall.** MMFF94's heavy tail
  comes from torsions relaxing into a different conformer, often one where
  the DFT structure was held by an intramolecular hydrogen bond. MMFF94 is
  about 100x faster, so it works as a loose pre-filter.
- **For rigid molecules, conformer search costs almost nothing.** With 2 or
  more rotatable bonds, 10 ETKDG conformers usually miss the right one, and
  the tolerance triples. A real funnel needs a proper conformer search (CREST)
  or a model that also predicts the conformer.
- **The lowest-energy conformer is a poor guess.** Part of this is a QM9
  artifact: each QM9 molecule is stored as one arbitrary conformer, not its
  global minimum. Cold molecular-beam and interstellar spectra come from
  low-energy conformers, so this needs re-testing against conformer-searched
  DFT references.
- **GFN2-xTB constants are about 1.1–1.7% too large compared with B3LYP.**
  This bias is easy to correct, but the spread, not the bias, sets the
  tolerance.
- **ETKDG cannot embed 3.6% of molecules** (strained polycyclic cages, 6.7%
  of the rigid set), so the funnel needs a fallback embedding route.
- **Even the best cheap stage leaves hundreds of candidates without a
  formula.** That is the room for an ML surrogate (GNN) that predicts
  DFT-quality constants from a cheap conformer, and for proposal models that
  start with fewer, better candidates.

Caveats: the reference is B3LYP, not experiment, and "right conformer" means
QM9's stored conformer.

## Reproduce

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]" tblite ase
```

```bash
mkdir -p data/raw/qm9_molnet
curl -L -o data/raw/qm9_molnet/qm9.zip https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
(cd data/raw/qm9_molnet && unzip qm9.zip)
```

```bash
python scripts/01_prepare_qm9.py              # ~1 min: data/processed/qm9.csv
python scripts/02_identifiability.py          # ~30 s: milestone 1 tables
python scripts/02b_plot_identifiability.py    # milestone 1 figures
OMP_NUM_THREADS=1 python scripts/03_cheap_method_error.py --n 2000 --workers 4   # ~18 min on 4 cores
python scripts/03b_analyze_cheap_methods.py   # milestone 2 tables, figure, link to milestone 1
pytest
```

QM9 comes from the MoleculeNet mirror because figshare (the original host)
blocks automated downloads.

**Data caveat:** in the MoleculeNet `gdb9.sdf`, bond orders and formal charges
are wrong for about 19% of molecules (25,173 have a net formal charge, although
every QM9 molecule is neutral). Read naively with RDKit, those records gain
implicit hydrogens and become different molecules. This code uses the SDF only
for atoms and coordinates, re-perceives bonds from the geometry with total
charge 0, and computes formulas from the explicit atoms. 6,837 molecules (5.1%)
have no valid neutral Lewis structure from their geometry; they keep their
constants and formula (milestone 1) but are excluded where a graph is needed
(milestone 2).

## Layout

```
src/rotml/
  physics/rotor.py               inertia tensor -> A, B, C, asymmetry, inertial defect
  data/qm9.py                    QM9 SDF + CSV loader
  analysis/identifiability.py    k-d tree match counting (milestone 1)
  analysis/cheap_methods.py      MMFF94 / GFN2-xTB / conformer search (milestone 2)
  plotting.py                    figure style
scripts/                         numbered pipeline steps
results/m1, results/m2           summary tables and figures (large per-molecule files are gitignored)
tests/                           unit tests
docs/research_plan.md            plan, pipeline, prior work
```
