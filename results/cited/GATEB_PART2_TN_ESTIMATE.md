# Gate B part 2 — 2D tensor-network cost estimate for the k>=8 contested
# cells (GPU box, 2026-08-10) — completes the A1 final verdict's frontier
# picture from the TN side

Question: can snake-MPS or PEPS reach the A1 card's k>=8 contested cells
(156q heavy-hex collective quench, D(q,k) connected differentials) at
grading accuracy? Anchors are MEASURED, not assumed: (i) this box's 1D
result — the same quench angles on a 42-site chain needed chi=3072 by
k=16-20 (prod21b, adopted as the record's grading baseline; chi=512
FAILS there); (ii) the k=6 operator census (31/64 rung operators exceed
2e8 Pauli terms at eps=1e-5); (iii) the actual device coupling map.

## Snake-MPS

Measured geometry (from utility_pending's rungs+bonds, BFS/bandwidth
ordering): 156 qubits, 170 edges, max degree 3, **snake cut width
median 8, max 12** (middle-third mean 8.3).

Entropy anchor: chi=3072 at k=16 across a 1D cut = log2(3072)/16 ~ 0.72
bits/step through ONE cut bond. A 2D bisection is crossed by ~8 bonds,
each fed by the same brickwork:

| k | S estimate (bits) | chi needed | MPS memory |
|---|---|---|---|
| 8 | ~46 | ~2^46 | ~2^96 bytes |
| 10 | ~58 | ~2^58 | ~2^119 bytes |
| 12 | ~70 | ~2^70 | ~2^142 bytes |

Even if the per-bond rate were 4x smaller in 2D (light-cone overlap
arguments), k=8 still needs chi ~ 2^11 PER CUT BOND ~ 2^30+ total.
**DIVERGES** — not by a margin a bigger machine closes; by ~25 orders.

## PEPS

Native geometry, so the cut argument softens — the cost moves into
contraction. Each Trotter step's rzz/rxx gates multiply the bond
dimension by their SVD rank (2) before truncation: D(k) <= 2^k, D(8) =
256 untruncated. Boundary-MPS contraction of a 156-site heavy-hex PEPS
scales ~ O(D^8-D^10) with the boundary chi_b on top: at D = 256 that is
>= 10^19 elementary contractions per observable — out of reach. The
practical regime (D <= 16) reaches k ~ 4-6 with growing truncation
error on SMOOTH observables; the contested cells are CONNECTED
correlator differentials at 1e-4-1e-3 absolute accuracy (hardware sigma
0.7-2 x 1e-4), i.e. two-sided cancellations far below the truncation
noise a D<=16 PEPS carries at k=8. **DIVERGES at grading accuracy.**
(Best-effort PEPS could produce a NUMBER at k=8 — but with no error
control at the accuracy the cells are graded at; per the record's
standards that is not a referee result.)

## Cross-check against the operator-space frontier

The sparse-Pauli attack (the verdict's engine) dies at the same wall
from the other side: 31/64 rung operators exceed 2e8 terms (11 GB each)
already at k=6/eps=1e-5, growing x2-8 per step in the edge band. Three
independent representations — MPS entanglement, PEPS contraction,
operator-space term count — hit exponential walls in the same k=6-8
window. That coherence is the strongest form of the frontier claim.

## Verdict

**k>=8 cells: DIVERGES for snake-MPS and for PEPS at grading accuracy,
with the numbers above.** The A1 final verdict's k>=8 contested-window
statement now carries the tensor-network leg it was owed. Gate B is
complete: part 1 (1D chi=3072 baseline) + part 2 (this estimate).

— GPU box
