# Merlin Quantum — GIC 2026 · HSBC — Credit-Card Fraud Detection

Phase-1 concept proposal, 2026 Global Quantum + AI Challenge.

| | |
|---|---|
| **Submitted proposal** | [`HSBC_PROPOSAL_v9.md`](HSBC_PROPOSAL_v9.md) — rendered as `report.pdf` |
| **Team** | Merlin Quantum, the quantum applications division of Merlin Digital (Dubai) |
| **Independent validation** | GIC 2026 **dual-track finalist** — Mitsubishi/AIST materials and QCi tracks; same framework, same instrument |
| **Repository** | https://github.com/sharadbachani-oss/merlin-quantum-hsbc-gic2026 |
| **Superseded material** | `archive/` — earlier drafts and planning notes, kept for provenance; not part of the submission |

## Claim → receipt

Every measurement the proposal reports, with the file in this repository that backs it. Each entry resolves inside this repo.

| # | Measurement | Receipt |
|---:|---|---|
| 1 | In a 14-qubit exact test of the entity-graph response oracle (24 neighbourhoods of six nodes, neighbour–neighbour couplings switched on from none to all fifteen pairs | [`mechanism_collective.json`](results/mechanism_collective.json) |
| 2 | Our tuned temporal baseline on IEEE-CIS (LightGBM, identity-merged; 472,432 train / 118,108 test) reports the statement's §4.1 metric set in full — AUC-ROC 0.9118, AUPRC 0.5532, and at the F1-optimal threshold 0.7244: F1… | [`g1_baseline_merged.json`](results/g1_baseline_merged.json) |
| 3 | On our own 64-rung lattice our exact sparse-Pauli referee reproduces every flown cell out to k = 6 (five cells within 1.4σ on one damping constant, four within 0.6σ), and past k = 8 that same attack leaves the machine: 31 of… | [`A1_FINAL_VERDICT.md`](results/cited/A1_FINAL_VERDICT.md) · [`k6_census.json`](results/k6_census.json) · [`GATEB_PART2_TN_ESTIMATE.md`](results/cited/GATEB_PART2_TN_ESTIMATE.md) |
| 4 | fewshot_ladder.py, laptop, ≈2 h: 4 label counts × 9 arms × 40 seeds, exact 2²⁴ enumeration | [`fewshot_ladder.py`](fewshot_ladder.py) |
| 5 | Quantum execution sample counts (statement §4.2). Every transaction subsample is drawn by stratified sampling preserving its source population's fraud/non-fraud ratio. ibm_fez gray-zone feature-extraction card: 96… | [`hsbc_ibm_band.py`](hsbc_ibm_band.py) · [`g4_band_analysis.json`](results/g4_band_analysis.json) · [`qgen_result.json`](results/qgen_result.json) · [`ulb_raw.json`](results/cited/ulb_raw.json) · [`hsbc_aquila_class.json`](results/hsbc_aquila_class.json) · [`hsbc_qkernel.json`](results/hsbc_qkernel.json) · [`hsbc_field_response_metrics.json`](results/hsbc_field_response_metrics.json) |
| 6 | Label-scarce augmentation, reproduced against its exact floor (ULB, time-sorted 80/20 split, 40 seeds per cell, frauds caught at a fixed 75-alert budget on the 56,962-transaction test window | [`fewshot_ladder.json`](results/fewshot_ladder.json) |
| 7 | AUPRC mean ± sd over 40 seeds; frauds caught at the 75-alert budget in parentheses; the full-label rung is the archived receipt | [`ulb_result.json`](results/ulb_result.json) |
| 8 | ULB degree-3 landscape sampling, 40 jobs, 82 distinct patterns | [`ulb_model.json`](results/ulb_model.json) |
| 9 | Collective-mechanism test: 24 neighbourhoods × coupling ladder 0–15 pairs, pairwise-additive prediction, burst vs spread (exploratory) | [`mechanism_collective.py`](mechanism_collective.py) |
| 10 | 128-qubit feature-extraction band card: 206 circuits × 8,192 shots, 2,720 two-qubit gates; four-arm analysis | [`ibm_band_state.json`](results/cited/ibm_band_state.json) |
| 11 | 128-qubit training-free generative card: 65 circuits × 8,192 shots, 524,288 samples | [`qgen_state.json`](results/cited/qgen_state.json) |
| 12 | Data-encoded quantum Boltzmann machine: data, null control and replica | [`qbm_model.json`](results/qbm_model.json) · [`qbm_result.json`](results/qbm_result.json) |
| 13 | Quantum kernel race, bar locked before any quantum Gram | [`hsbc_classical.json`](results/hsbc_classical.json) |
| 14 | Alert-policy optimisation gate, greedy = exact at equal catch | [`policy_gate.json`](results/policy_gate.json) |
| 15 | Graph-shape response oracle vs labels: 10,362 neighbourhoods, five arms, five temporal folds | [`hsbc_response_test_workstation.json`](hsbc_response_test_workstation.json) · [`floor_response_feature.json`](floor_response_feature.json) |
| 16 | Field-carrying response oracle: floor (600 synthetic neighbourhoods) and label test (10,362 neighbourhoods, six arms, five folds) | [`floor_field_response.json`](floor_field_response.json) · [`hsbc_field_response.json`](hsbc_field_response.json) · [`hsbc_field_response.py`](hsbc_field_response.py) |

## Verifying

```
python verify.py
```

Replays the headline numbers from archived counts — no credentials, no network, numpy only.
