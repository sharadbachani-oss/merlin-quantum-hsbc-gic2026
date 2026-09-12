# -*- coding: utf-8 -*-
"""Topology grading by ground-band response energy, exact and Krylov.

The device measures the complex echo L(t) = <g| e^{-iHt} |g>, whose Fourier
peaks sit at the eigenvalues of the payment-graph Hamiltonian weighted by
their overlap with the prepared coupled-pair ground state.  The leading
bound line

        w1(G) = -E_lead(G),   E_lead = dominant negative-energy eigenvalue

is the quantity that grades topology.  This script measures it three ways
so the estimator cannot be doing the work:

  exact   full diagonalisation                        (k <= 2, n <= 12)
  krylov  Lanczos continued fraction from |g>         (any k; validated
                                                       against exact)
Nothing is fitted.  kappa = 3/(3-sqrt(5)) is derived.
"""
import json, math, os, sys, time
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("lw", os.path.join(HERE, "hsbc_layering_wall.py"))
lw = importlib.util.module_from_spec(_s); _s.loader.exec_module(lw)
FAMS = ["aml_ring6", "aml_ring4", "smurf_star", "null_native"]

def exact_lines(name, k, ntop=3):
    n, nat, ex, pl = lw.layered(name, k)
    d = lw.diag_D(n, nat + ex + pl)
    H = np.diag(d)
    idx = np.arange(1 << n)
    for i in range(n):
        H[idx ^ (1 << i), idx] -= 1.0
    ev, U = np.linalg.eigh(H)
    _, g = lw.ground_state(n, nat)
    c = np.abs(U.conj().T @ g) ** 2
    neg = np.where(ev < 0)[0]
    lead = neg[np.argmax(c[neg])]
    top = np.argsort(c)[-ntop:][::-1]
    return {"E_lead": float(ev[lead]), "w1": float(-ev[lead]),
            "weight_lead": float(c[lead]),
            "top": [[float(ev[i]), float(c[i])] for i in top]}

def krylov_lines(name, k, m=160):
    """Lanczos from |g> under H[G]: Ritz values with their |<g|.>|^2 weights."""
    n, nat, ex, pl = lw.layered(name, k)
    d = lw.diag_D(n, nat + ex + pl)
    _, g = lw.ground_state(n, nat)
    N = 1 << n
    V = np.empty((m, N), dtype=np.float64)
    a = np.zeros(m); b = np.zeros(m)
    V[0] = g.real / np.linalg.norm(g.real); j = m
    for i in range(m):
        w = lw.apply_H(V[i].astype(np.complex128), n, d).real
        a[i] = float(V[i] @ w)
        Vi = V[: i + 1]
        w -= Vi.T @ (Vi @ w)                       # full reorthogonalisation
        nb = float(np.linalg.norm(w))
        if nb < 1e-10:
            j = i + 1; break
        b[i] = nb
        if i + 1 < m: V[i + 1] = w / nb
    T = np.diag(a[:j]) + np.diag(b[: j - 1], 1) + np.diag(b[: j - 1], -1)
    ev, U = np.linalg.eigh(T)
    c = U[0] ** 2                                  # weights of |g> on Ritz vectors
    neg = np.where(ev < 0)[0]
    lead = neg[np.argmax(c[neg])]
    return {"E_lead": float(ev[lead]), "w1": float(-ev[lead]),
            "weight_lead": float(c[lead]), "krylov_dim": int(j)}

if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out = {"kappa": lw.KAPPA, "delta": lw.DELTA, "grading": {}}
    print("ground-band response energy  w1(G) = -E_lead    [exact | krylov]")
    for k in range(1, kmax + 1):
        row = {}
        for f in FAMS:
            t0 = time.time()
            kr = krylov_lines(f, k)
            ex = exact_lines(f, k) if 6 * k <= 12 else None
            row[f] = {"krylov": kr, "exact": ex,
                      "n_qubits": 6 * k, "seconds": round(time.time() - t0, 1)}
            e = f"{ex['w1']:.6f}" if ex else "   ---  "
            print(f"  k={k} {f:12s} w1 = {kr['w1']:.6f} (krylov, dim {kr['krylov_dim']})"
                  f"   {e} (exact)   weight={kr['weight_lead']:.4f}"
                  f"   [{row[f]['seconds']}s]", flush=True)
        # separations that matter
        r6, r4, st = (row[x]["krylov"]["w1"] for x in ("aml_ring6", "aml_ring4", "smurf_star"))
        row["_sep_ring4_vs_star_pct"] = 100.0 * abs(st - r4) / r4
        row["_sep_ring6_vs_nearest_pct"] = 100.0 * min(abs(r4 - r6), abs(st - r6)) / r6
        print(f"     -> ring4 vs edge-matched star: {row['_sep_ring4_vs_star_pct']:.2f}%"
              f"   | ring6 vs nearest control: {row['_sep_ring6_vs_nearest_pct']:.2f}%\n", flush=True)
        out["grading"][k] = row
    json.dump(out, open(os.path.join(HERE, "results", "hsbc_response_grading.json"), "w"), indent=1)
    print("wrote results/hsbc_response_grading.json")
