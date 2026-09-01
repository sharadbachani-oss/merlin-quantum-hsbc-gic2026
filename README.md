# Merlin Digital — GIC 2026 HSBC Enterprise Track

| | |
|---|---|
| **Team** | Merlin Digital |
| **Project** | Quantum Generative Augmentation for Novel-Pattern Fraud Response |
| **Write-up** | `HSBC_SUBMISSION_BODY.md` / `report.pdf` |
| **Prior result** | GIC 2026 — **dual-track finalist** (Mitsubishi/AIST materials track) |
| **Contact** | sharad.bachani@merlin-me.com · ORCID 0009-0008-4679-6717 |

## The claim in one line

When a new fraud pattern is first detected and only **five confirmed
cases** exist, augmenting the training set with samples drawn from a
quantum device catches **+5.8 percentage points more fraud at the same
alert budget** than SMOTE (66.4% vs 60.6%; 40 seeds; paired p = 0.001) —
and **halves the variance** of that outcome (±4.3 vs ±8.2 frauds).

The advantage is confined to the few-label regime: it is decisive at five
labels, absent by ten, and reversed beyond twenty. That is the behaviour
the *power of data* result predicts for quantum models on classical data,
and we report the regimes where classical wins with the same receipts.

## Verify it without credentials

```bash
pip install numpy scikit-learn xgboost
python verify.py                      # replays every headline number
```

Add the open ULB dataset for live end-to-end replication:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip
python verify.py creditcard.csv
```

`results/ulb_raw400.json` holds the raw bitstrings returned by QCi
Dirac-3 across 354 jobs. Everything downstream is deterministic given the
published seeds.

## Contents

```
HSBC_SUBMISSION_BODY.md   the write-up (also as report.pdf)
verify.py                 credential-free replication of every number
qgen_ulb.py               the scored pipeline: landscape fit, device fly, scoring
qgen_card.py              128-qubit generative card (IBM Heron)
qbm_final.py              data-encoded Boltzmann machine (reported negative)
hsbc_ibm_band.py          128-qubit feature-map card (reported null)
results/                  archived device output + graded JSON for every claim
```

## Hardware receipts

| platform | role | scale | job |
|---|---|---|---|
| QCi Dirac-3 | degree-3 generative sampling (scored) | 354 jobs, 4,803 samples | `results/ulb_raw400.json` |
| IBM Heron `ibm_fez` | 128-qubit generative sampling | 65 × 8,192 shots, 2,656 2q gates, 524,288 samples in one job | `dab6therrl7c7386flo0` |
| IBM Heron `ibm_fez` | 128-qubit feature extraction | 206 × 8,192 shots, 2,720 2q gates | `daaucejvpcac73dd232g` |
| IBM Heron `ibm_fez` | data-encoded Boltzmann machine | 87 qubits, 1,152 2q gates, depth 196 | `daba21jvpcac73ddh9h0` |
| QuEra Aquila (Braket) | 256-atom analog anomaly encoding | 8 jobs, 99.3% mean atom loading | Braket console |

## Discipline

Every result carries a pre-registration written **before** execution,
≥3 seeds (40 for the headline), in-job controls, and cloud job IDs.
Negative results ship at the same receipt class as positive ones — the
data-rich regime where jitter beats us, the saturated-baseline null, the
feature-map null, and the failed 87-qubit Boltzmann machine are all in
the write-up and in `results/`.

No credentials are embedded anywhere in this repository. Flight scripts
read `IBM_QUANTUM_CRN` and `QCI_TOKEN` from the environment.
