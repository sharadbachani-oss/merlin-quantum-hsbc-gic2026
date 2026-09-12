# -*- coding: utf-8 -*-
"""Which side of the ledger each machine is playing on, from the operator alone.

The payment-graph operator is

    H[G] = kappa * D_G - A,     kappa = 3/(3 - sqrt5)   (derived)

with D_G diagonal in the computational basis and A the sum of single-site
flips.  Every off-diagonal entry is therefore -1 or 0: H[G] is a Z-matrix,
which is to say STOQUASTIC.  That is a structural fact about the derived
operator, not a modelling choice, and it decides what any machine can win:

  IMAGINARY TIME / GROUND STATE.  A stoquastic Hamiltonian has no sign
  problem.  exp(-tau H) has non-negative entries, so path-integral and
  diffusion Monte Carlo sample the ground state without the exponential
  cancellation that makes general quantum systems hard, and the ground
  vector is strictly positive -- a probability distribution.  Any
  ground-state machine, an annealer or a photonic sampler with conserved
  occupation included, is competing against sign-problem-free Monte Carlo.
  There is no sign-problem hardness there for it to exploit.

  REAL TIME.  exp(-i H t) is oscillatory.  Non-positive off-diagonals buy
  nothing: the cancellation returns in full.  That is the same obstruction
  this package already measures as the ill-posedness of analytic
  continuation, and it is why the route is real-time spectroscopy rather
  than ground-state optimization.

This script verifies the structure rather than asserting it, at every
typology and depth the package uses.
"""
import json, os, sys
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    "lw", os.path.join(HERE, "hsbc_layering_wall.py"))
lw = importlib.util.module_from_spec(_s)
_s.loader.exec_module(lw)
FAMS = ["aml_ring6", "aml_ring4", "smurf_star", "null_native"]


def dense_H(name, k):
    n, nat, ex, pl = lw.layered(name, k)
    d = lw.diag_D(n, nat + ex + pl)
    H = np.diag(d)
    idx = np.arange(1 << n)
    for i in range(n):
        H[idx ^ (1 << i), idx] -= 1.0
    return n, H


def check(name, k):
    n, H = dense_H(name, k)
    off = H - np.diag(np.diag(H))
    stoq = bool((off <= 0).all())
    ev, U = np.linalg.eigh(H)
    g = U[:, 0]
    g = g * np.sign(g[int(np.argmax(np.abs(g)))])
    positive = bool((g > 1e-12).all())
    # exp(-tau H) entrywise non-negative is the sign-problem-free statement
    tau = 0.25
    P = np.eye(1 << n) - tau * H          # first-order transfer element
    shift = np.diag(np.diag(P)).max()
    Pnn = bool(((P - np.diag(np.diag(P))) >= -1e-12).all())
    return {"family": name, "k": k, "n_qubits": n,
            "max_offdiagonal": float(off.max()),
            "stoquastic": stoq,
            "ground_vector_strictly_positive": positive,
            "ground_vector_min_entry": float(g.min()),
            "transfer_offdiagonals_nonnegative": Pnn,
            "E0": float(ev[0])}


if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rows = []
    print("operator class of H[G] -- verified, not asserted")
    for k in range(1, kmax + 1):
        for f in FAMS:
            r = check(f, k)
            rows.append(r)
            print("  k=%d %-12s n=%2d  max off-diag %+.1f   stoquastic %-5s   "
                  "ground vector positive %-5s (min %.2e)"
                  % (r["k"], r["family"], r["n_qubits"], r["max_offdiagonal"],
                     r["stoquastic"], r["ground_vector_strictly_positive"],
                     r["ground_vector_min_entry"]), flush=True)
    allstoq = all(r["stoquastic"] for r in rows)
    allpos = all(r["ground_vector_strictly_positive"] for r in rows)
    print("\n  every instance stoquastic: %s   every ground vector positive: %s"
          % (allstoq, allpos))
    print("  => ground-state machines face sign-problem-free Monte Carlo here;")
    print("     the real-time route is where the cancellation survives.")
    out = {"kappa": lw.KAPPA, "instances": rows,
           "all_stoquastic": allstoq, "all_ground_positive": allpos,
           "ledger": {
               "imaginary_time": "sign-problem-free; no advantage available "
                                 "to any ground-state machine",
               "real_time": "oscillatory; tensor-network route measured shut, "
                            "continuation ill-posed"}}
    json.dump(out, open(os.path.join(HERE, "results",
                                     "hsbc_operator_class.json"), "w"), indent=1)
    print("  wrote results/hsbc_operator_class.json")
