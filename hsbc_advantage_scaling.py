# -*- coding: utf-8 -*-
"""Where the two costs cross, and how the classical escape route is closed.

Two measured inputs, no modelling assumptions:

  1. chi wall.  For the layered payment graph the real-time quenched state
     is volume-law entangled.  We do not assert this -- we search EVERY
     balanced bipartition of the qubits and report the classical
     simulator's BEST case.  If the minimum over all orderings still sits
     at the cap, no matrix-product-state ordering helps.

  2. Gate law.  The circuit is exact: one CNOT per native pair for the
     preparation, one RZZ (2 CNOTs) per payment edge.  No Trotter error,
     no depth growth in evolution time.

The device envelope is not projected either.  Our own campaign measured
18/18 PASS at 77 two-qubit gates per tile and contrast collapse at 240.
Those two numbers bracket the window this table lives in.
"""
import itertools, json, math, os, sys
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("lw", os.path.join(HERE, "hsbc_layering_wall.py"))
lw = importlib.util.module_from_spec(_s); _s.loader.exec_module(lw)

VALIDATED_PASS_2Q = 77      # ibm_kingston, 18/18, measured
MEASURED_FAIL_2Q  = 240     # ibm_marrakesh, 5/19 contrast collapse, measured
BYTES_PER_AMP     = 16      # complex128

def gates(k):
    """3k CNOT prep + 2 CNOT per typology edge (5k) and placement link (k-1)."""
    return 3 * k + 2 * (5 * k + (k - 1))

def chi_bipart(psi, n, left, eps=1e-6):
    v = psi.reshape([2] * n)                       # axis i <-> bit (n-1-i)
    order = [n - 1 - b for b in sorted(left)] + [n - 1 - b for b in range(n) if b not in left]
    v = np.transpose(v, order).reshape(1 << len(left), -1)
    p = np.linalg.svd(v, compute_uv=False) ** 2; p /= p.sum()
    tail = np.cumsum(p[::-1])[::-1]
    return max(1, min(int(np.searchsorted(-tail, -eps) + 1), p.size))

def exhaustive_wall(k, T=8.0, dt=0.1, name="aml_ring6"):
    """Minimum chi over ALL balanced bipartitions -- the classical best case."""
    n, nat, ex, pl = lw.layered(name, k)
    d_all = lw.diag_D(n, nat + ex + pl)
    _, gs = lw.ground_state(n, nat)
    psi = gs.copy()
    for _ in range(int(round(T / dt))):
        psi = lw.lanczos_step(psi, n, d_all, dt)
    cuts = list(itertools.combinations(range(n), n // 2))
    vals = [chi_bipart(psi, n, set(c)) for c in cuts]
    return {"k": k, "n_qubits": n, "cuts_searched": len(cuts),
            "chi_cap": 1 << (n // 2), "chi_min_over_orderings": int(min(vals)),
            "chi_median": int(np.median(vals)), "chi_max": int(max(vals)),
            "compression_available_pct": round(100.0 * (1 - min(vals) / (1 << (n // 2))), 2)}

def crossover(kmax=16):
    rows = []
    for k in range(1, kmax + 1):
        n = 6 * k; g = gates(k)
        rows.append({"k": k, "n_qubits": n, "two_qubit_gates": g,
                     "classical_bytes": BYTES_PER_AMP * (2.0 ** n),
                     "within_validated_envelope": g <= VALIDATED_PASS_2Q,
                     "beyond_measured_failure": g > MEASURED_FAIL_2Q})
    return rows

def human(b):
    for u in ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]:
        if b < 1024: return f"{b:,.1f} {u}"
        b /= 1024
    return f"{b:.3g} YB+"

if __name__ == "__main__":
    kmax_wall = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print("=" * 74)
    print("1) CLASSICAL ESCAPE ROUTE -- exhaustive search over qubit orderings")
    print("=" * 74)
    walls = []
    for k in range(1, kmax_wall + 1):
        w = exhaustive_wall(k); walls.append(w)
        print(f"  k={w['k']}  n={w['n_qubits']:2d}  searched {w['cuts_searched']:>6,} balanced cuts"
              f"   best-case chi = {w['chi_min_over_orderings']}/{w['chi_cap']}"
              f"   -> {w['compression_available_pct']}% compression available")
    print("\n" + "=" * 74)
    print("2) COST CROSSOVER   (classical = full statevector; MPS route closed above)")
    print("=" * 74)
    print(f"{'layers k':>8} {'qubits':>7} {'2q gates':>9} {'classical memory':>20}   device envelope")
    rows = crossover()
    for r in rows:
        env = ("inside 18/18-validated depth" if r["within_validated_envelope"]
               else "beyond measured collapse" if r["beyond_measured_failure"]
               else "bracketed: past 77, under 240")
        print(f"{r['k']:>8} {r['n_qubits']:>7} {r['two_qubit_gates']:>9} "
              f"{human(r['classical_bytes']):>20}   {env}")
    a = next(r for r in rows if r["two_qubit_gates"] <= VALIDATED_PASS_2Q and
             gates(r["k"] + 1) > VALIDATED_PASS_2Q)
    b = next(r for r in rows if r["n_qubits"] == 48)
    ratio = b["classical_bytes"] / a["classical_bytes"]
    print("\n" + "-" * 74)
    print(f"From k={a['k']} ({a['two_qubit_gates']} 2q gates, inside the validated envelope) "
          f"to k={b['k']} ({b['two_qubit_gates']} 2q gates, still under the\nmeasured 240-gate collapse): "
          f"circuit grows {b['two_qubit_gates']/a['two_qubit_gates']:.2f}x while the classical\n"
          f"requirement grows {ratio:,.0f}x -- {math.log10(ratio):.1f} orders of magnitude.")
    print("-" * 74)
    json.dump({"validated_pass_2q": VALIDATED_PASS_2Q,
               "measured_fail_2q": MEASURED_FAIL_2Q,
               "exhaustive_wall": walls, "crossover": rows,
               "orders_of_magnitude": round(math.log10(ratio), 2)},
              open(os.path.join(HERE, "results", "hsbc_advantage_scaling.json"), "w"), indent=1)
    print("wrote results/hsbc_advantage_scaling.json")
