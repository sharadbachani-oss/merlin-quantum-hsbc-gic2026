# -*- coding: utf-8 -*-
"""The three barriers to classical simulation, measured on the derived tile.

A quantum machine only wins where a classical one cannot follow, and there
are exactly three known reasons it cannot:

  1. ENTANGLEMENT beyond tensor-network compression.  Area-law states are
     matrix-product states and give nothing away.  The bond dimension is
     the measurable.
  2. SIGN STRUCTURE that defeats Monte Carlo.  Non-negative amplitudes
     mean no cancellation and no advantage; the measurable is how much
     cancellation the propagator actually carries.
  3. NON-STABILIZERNESS, or magic.  By Gottesman-Knill a Clifford circuit
     is classically simulable however entangled it is.  The measurable is
     the stabilizer Renyi entropy.

This script asks what the DERIVED physics supplies against each, with no
fitting.  All three are computed on the same 6-qubit tile the device runs.

  kappa = 3/(3-sqrt5),  delta = sqrt((kappa/2)^2+4) - kappa/2
  chi   = atan(delta/2),  theta* = pi/2 - 2*chi
  preparation = RY(theta*) . CX . H(x)H  per coupled pair
"""
import json, math, os, sys
import itertools
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    "lw", os.path.join(HERE, "hsbc_layering_wall.py"))
lw = importlib.util.module_from_spec(_s)
_s.loader.exec_module(lw)
K, DELTA = lw.KAPPA, lw.DELTA
CHI = math.atan(DELTA / 2.0)
THETA = math.pi / 2.0 - 2.0 * CHI
FAMS = ["null_native", "aml_ring4", "aml_ring6", "smurf_star"]

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [I2, X, Y, Z]


def dense_H(name, k=1):
    n, nat, ex, pl = lw.layered(name, k)
    E = sorted({(min(a, b), max(a, b)) for a, b in nat + ex + pl})
    H = np.diag(lw.diag_D(n, E))
    idx = np.arange(1 << n)
    for i in range(n):
        H[idx ^ (1 << i), idx] -= 1.0
    return n, H, E


# ------------------------------------------------------ barrier 3: magic
def apply_1q(psi, n, i, M):
    """Apply a single-qubit matrix on bit i (axis n-1-i in the tensor view)."""
    v = psi.reshape([2] * n)
    v = np.moveaxis(v, n - 1 - i, 0)
    v = np.tensordot(M, v, axes=([1], [0]))
    return np.moveaxis(v, 0, n - 1 - i).reshape(-1)


def stabilizer_renyi2(psi, n):
    """M_2 = -log2( d * sum_P Xi_P^2 ),  Xi_P = <psi|P|psi>^2 / d.
    Zero exactly on stabilizer states, positive otherwise."""
    d = 1 << n
    tot = 0.0
    for combo in itertools.product(range(4), repeat=n):
        phi = psi
        for i, p in enumerate(combo):
            if p:
                phi = apply_1q(phi, n, i, PAULI[p])
        e = float(np.vdot(psi, phi).real)
        tot += (e * e / d) ** 2
    return float(-math.log2(d * tot))


# ------------------------------------------------- barrier 2: sign weight
def sign_survival(M):
    """|sum of entries| / sum of |entries|.  1.0 means no cancellation at
    all, so Monte Carlo samples it without a sign problem; small means the
    propagator lives on cancellation."""
    return float(abs(M.sum()) / np.abs(M).sum())


# --------------------------------------------- barrier 1: entanglement
def entanglement(psi, n, cut=None):
    c = cut or n // 2
    s = np.linalg.svd(psi.reshape(1 << (n - c), 1 << c), compute_uv=False)
    p = s ** 2
    p = p / p.sum()
    nz = p[p > 1e-15]
    tail = np.cumsum(p[::-1])[::-1]
    chi = max(1, min(int(np.searchsorted(-tail, -1e-6) + 1), p.size))
    return float(-(nz * np.log(nz)).sum()), int(chi), 1 << c


def main(times=(0.0, 2.0, 4.0, 8.0, 16.0)):
    out = {"kappa": K, "delta": DELTA, "chi_angle": CHI, "theta_star": THETA,
           "theta_star_over_pi": THETA / math.pi, "families": {}}
    print("derived angles:  chi = atan(delta/2) = %.9f   theta* = pi/2 - 2chi"
          " = %.9f rad = %.6f pi" % (CHI, THETA, THETA / math.pi))
    # is the preparation angle Clifford?  Clifford single-qubit rotations are
    # multiples of pi/2; the T gate sits at pi/4.
    near = min((abs(THETA - m * math.pi / 4), m) for m in range(9))
    print("  nearest multiple of pi/4 is %d*pi/4, off by %.6f rad (%.3f%%)"
          % (near[1], near[0], 100 * near[0] / (math.pi / 4)))
    print("  Clifford (multiple of pi/2): %s   T-angle (pi/4): %s"
          % (abs(THETA % (math.pi / 2)) < 1e-12, abs(THETA - math.pi / 4) < 1e-12))
    out["theta_is_clifford"] = bool(abs(THETA % (math.pi / 2)) < 1e-12)
    out["theta_is_T_angle"] = bool(abs(THETA - math.pi / 4) < 1e-12)
    out["theta_offset_from_T_rad"] = float(abs(THETA - math.pi / 4))

    print("\nbarrier 2 -- SIGN: cancellation carried by the propagator")
    n, H, E = dense_H("aml_ring6")
    from scipy.linalg import expm
    rows = []
    for t in (1.0, 4.0, 8.0):
        im = sign_survival(expm(-t * H))          # imaginary time
        re = sign_survival(expm(-1j * t * H))     # real time
        rows.append({"t": t, "imaginary_time_sign": im, "real_time_sign": re})
        print("   t=%4.1f   exp(-tH) sign survival %.6f      "
              "exp(-itH) sign survival %.6f   ratio %.1fx"
              % (t, im, re, im / max(re, 1e-18)))
    out["sign"] = rows

    print("\nbarriers 1 and 3 -- ENTANGLEMENT and MAGIC along the quench")
    print("   family        t     S(nats)  chi/cap    magic M2   M2/qubit")
    for f in FAMS:
        n, H, E = dense_H(f)
        _, g = lw.ground_state(n, [(i, i + 3) for i in range(3)])
        ev, U = np.linalg.eigh(H)
        c0 = U.conj().T @ g
        recs = []
        for t in times:
            psi = U @ (np.exp(-1j * ev * t) * c0)
            S, chi, cap = entanglement(psi, n)
            M2 = stabilizer_renyi2(psi, n)
            recs.append({"t": t, "S_nats": S, "chi": chi, "chi_cap": cap,
                         "magic_M2": M2, "magic_per_qubit": M2 / n})
            print("   %-12s %5.1f   %6.3f   %2d/%-2d    %8.4f   %8.4f"
                  % (f, t, S, chi, cap, M2, M2 / n), flush=True)
        out["families"][f] = recs
    json.dump(out, open(os.path.join(HERE, "results",
                                     "hsbc_three_barriers.json"), "w"), indent=1)
    print("\n  wrote results/hsbc_three_barriers.json")


if __name__ == "__main__":
    main()
