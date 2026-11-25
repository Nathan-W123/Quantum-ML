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
| 2. Error of cheap geometry methods (MMFF94, GFN2-xTB) vs DFT | in progress |
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
| A, B, C + formula | 99.5% (0) | 94.5% (0) | 79.8% (1) | 46.3% (4) | 11.8% (36) |
| A, B, C + dipole magnitude (±0.2 D) | 99.6% (0) | 93.7% (0) | 69.6% (1) | 23.1% (7) | 2.3% (80) |
| A, B, C + formula + dipole | 99.6% (0) | 98.8% (0) | 95.4% (0) | 82.8% (1) | 45.2% (6) |

![Uniquely identified vs tolerance](results/m1/unique_vs_tolerance.png)

Findings:

- **The accuracy of the theory is the bottleneck.** At 0.1% (best composite
  methods with vibrational corrections), rotational constants alone identify
  98.6% of QM9 molecules. At 1% (typical DFT) they identify only 12%.
- **Knowing the formula buys roughly a factor of 2-4 in tolerance**: with the
  formula, 1% tolerance identifies 80% of molecules; without it you need a
  tolerance between 0.25% (90%) and 0.5% (57%).
- **One extra observable helps nearly as much as the formula.** The dipole
  magnitude alone takes 1%-tolerance identification from 12% to 70%.
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

_Results are filled in from `results/m2/` once the benchmark finishes._

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
OMP_NUM_THREADS=1 python scripts/03_cheap_method_error.py --n 2000 --workers 4   # ~25 min on 4 cores
python scripts/03b_analyze_cheap_methods.py   # milestone 2 tables, figure, link to milestone 1
pytest
```

QM9 comes from the MoleculeNet mirror because figshare (the original host)
blocks automated downloads.

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
