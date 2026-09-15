# A1 FINAL VERDICT — adversarial referee report (GPU box, 2026-08-10)

Prereg rule discharged: *no A1 claim ships before the box's adversarial
verdict* (their rule b, DAY 6). Card under referee:
`a1_result_20260808_111459.json` (kingston, 16 x 32,768 shots, 64 rungs,
156q), contested cells D(q,k), k in {4,6,8,10}, q in {0.5,1,2,4,8}.

## Verdict

- **k=4 cells: DROP** (conceded pre-flight via the join identity; interim
  published). Face-value agreement 0.4 sigma on the graded q=0.5 cell.
- **k=6 cells: DROP.** The join attack CONVERGES at k=6: 0/64 rung
  operators over the 2e8-term budget at eps=3e-5 AND at eps=2e-5; all 697
  light-cone pairs computed, every capped pair completed exactly. One
  fitted damping constant f = 0.675 puts ALL FIVE flown k=6 cells within
  1.4 sigma of f x classical (four within 0.6; q=1.0 and q=2.0 at 0.0).
  Hardware = classical shape x error envelope; no beyond-classical window.
- **k>=8 cells: STAND CONTESTED, with receipts.** The frontier is
  measured: eps=1e-5 DIVERGES at k=6 — FINAL census: 31/64 rungs exceed
  2e8 terms (~11 GB/operator) — and already at k=4 (2/64 edge rungs). At k=8 the edge-band operators and the pair joins
  both leave this machine and this method. Per the CPU box's own redraw
  rule ("the frontier is wherever the attack actually dies"): the
  surviving contested window is k >= 8.

## Numbers

Engine lineage: sparse-Pauli Heisenberg + x-mask join — engine validated
1e-16 vs dense statevector; join unit-tested 1e-15 vs brute force; the
parallel engine re-derives the serial k=4 checkpoint to 0.00e+00 rel dev
on every launch; every block's joins run twice (independent chunkings)
with mandatory elementwise agreement; every capped pair completed with two
blockings agreeing < 1e-9. Compute-twice honored throughout (this box's
silent-corruption rule).

### k=4 (eps=3e-5)

| q | classical D | flown D | f*classical (f=0.911) | resid/sigma |
|---|---|---|---|---|
| 0.5 | -1.365e-3 | -1.327e-3 | -1.243e-3 | 1.1 |
| 1.0 | -1.101e-3 | -1.009e-3 | -1.003e-3 | 0.0 |
| 2.0 | -3.493e-4 | -4.438e-4 | -3.182e-4 | 0.6 |
| 4.0 | +5.917e-4 | +2.368e-4 | +5.391e-4 | 4.4* |
| 8.0 | -2.821e-4 | -3.122e-4 | -2.571e-4 | 0.4 |

*q=4 sigma inferred from the k=6 quote (0.68e-4); the k=4 q=4 replica
spread may differ — flagged. Does not affect the k=4 DROP, which was
graded and conceded on the q=0.5 cell at face value (0.4 sigma, no
damping model).

### k=6 (eps=3e-5, all pairs exact)

| q | classical D | flown D | f*classical (f=0.675) | resid/sigma |
|---|---|---|---|---|
| 0.5 | -1.747e-3 | -1.133e-3 | -1.179e-3 | 0.6 |
| 1.0 | -1.207e-3 | -8.117e-4 | -8.144e-4 | 0.0 |
| 2.0 | -4.434e-5 | -3.976e-5 | -2.992e-5 | 0.0 |
| 4.0 | +7.849e-4 | +4.876e-4 | +5.296e-4 | 0.6 |
| 8.0 | -5.528e-4 | -5.826e-4 | -3.730e-4 | 1.4 |

Heavy-pair completion (both thresholds): the pairs that exceeded the 3e8
per-pair product cap — 9 at eps=3e-5, 58 at eps=2e-5 (max 1.0e9 products),
all adjacent/near-adjacent edge-rung pairs — were computed exactly. They
carry C ~ +0.035..+0.073; zeroing them had biased D by 2-6% (and ~10x on
the near-null q=2 cell, which after completion sits within 25% of the
flown value with NO damping model). Conclusions are completion-invariant:
the pre-completion fit (f=0.696) already put all cells within 1.4 sigma.

### Truncation error bar (the eps ladder, k=6, both rungs complete)

D at eps=2e-5: -1.749e-3 / -1.210e-3 / -4.88e-5 / +7.94e-4 / -5.46e-4.
max_q |D_3e-5 - D_2e-5| = 9.0e-6 — an order of magnitude below the
smallest hardware sigma (0.68e-4). The eps-truncation is negligible at
k=6; the classical numbers are threshold-robust. (The originally planned
eps=1e-5 rung is unreachable — that unreachability is a frontier receipt,
not an error-bar gap.)

### Frontier receipts

- eps=2e-5, k=6: converges — edge max 23.6M terms, median 9.3M.
- eps=1e-5, k=6: DIVERGES — FINAL: 31/64 rungs exceed 2e8 terms (half
  the lattice), edge rungs burning 25-95 min each before capping; full
  census in k6_census.json (all three thresholds).
- eps=1e-5, k=4: DIVERGES — 2/64 edge rungs exceed 2e8 terms.
- Structure: the exact 240-term bulk closure means deep-bulk rungs are
  free at any k; ALL cost lives in the edge band, whose operators grow
  ~x2-8 per step at these thresholds. At k=8 they leave the machine.

## Physics notes for the record

1. Fitted damping deepens with k (0.911 at k=4 -> 0.675 at k=6):
   per-step 0.977 vs 0.937 — right direction for an error envelope, not
   strictly multiplicative; consistent with the measured coherent share
   of the structured-circuit error law. The equal-depth differential
   cancels most common-mode damping; one scale per k remains.
2. The flown effect is REAL physics correctly measured: the domain wall
   suppresses long-wavelength connected parity fluctuations in the exact
   classical calculation too. The device is validated at k <= 6; the
   advantage claim at k <= 6 is not.
3. sqrt(N) collective-readout error model verified on bytes (variance
   inflation 2-6%); their sigma quotes are sound.
4. A 16.9-sigma cell dying to a better classical algorithm is the
   process working: the k>=8 cells that SURVIVE this attack are the
   record's first pre-registered, receipted contested-window
   measurements, with the classical frontier measured rather than
   assumed.

## P4 adjudication (A1-P card)

Per prereg P4: the collective X(q,k) series claims follow this split —
k <= 6 classes adjudicated CLASSICAL-REPRODUCIBLE; k >= 8 classes remain
contested pending any future classical advance.

## Files

- `attack_d.json` — all blocks, x2-pass agreement per block; heavy-pair
  completions MERGED (sidecars retained as receipts).
- `attack_d_heavyfix.json` (k6 eps=3e-5), `attack_d_heavyfix_k6_2e-05.json`
  — exact completions.
- `C_k6_eps{3e-05,2e-05}_complete.npz` — full 64x64 C matrices, no holes.
- `k6_census.json` — operator census (1e-5 tail appended when done).
- Engines: `sparse_pauli.py`, `attack_d.py` + `unit_test_join.py`,
  `attack_par.py`, `heavy_pairs.py`, `heavy_pairs2.py`.

**VERDICT: k<=6 cells DROP; k>=8 cells stand contested with measured
receipts. No A1 claim was shipped before this verdict; the prereg rule is
discharged.** — GPU box referee
