# -*- coding: utf-8 -*-
"""Layering depth: the wall the launderer chooses, and what it costs each side.

Industrial object.  Real mule networks are LAYERED: k laundering rings
routed through a shared placement tier, so value hops between rings.  k is
the launderer's cheapest control -- adding a layer is one more account --
and it is exactly the axis on which classical pattern mining degrades.

This script scales the ALREADY-VALIDATED single-tile protocol to k layers
and measures, with no fitting and no free parameters:

  (a) does the typology discriminator survive layering?  Weight of the
      response S(omega) on the derived gap line, ring-6 layers vs ring-4,
      smurf-star and null controls, at k = 1, 2, 3;
  (b) what does the SAME object cost a classical simulator?  The Schmidt
      rank chi that a matrix-product state needs to carry the real-time
      quenched state at truncation error 1e-6;
  (c) what does it cost the device?  Two-qubit gates: one exact pair
      preparation plus one RZZ per payment edge.

Protocol is inherited verbatim from hsbc_spectral_route.py:
  operator      H[G] = kappa * D_G - A,  kappa = 3/(3-sqrt(5))   (derived)
  state         native ground state, exact preparation
  observable    complex L(t) = <gs| e^{-iHt} |gs>  -> FFT peaks at EIGENVALUES
  band          derived gap line  Delta_s +/- 0.12
"""
import json, math, os, sys, time
import numpy as np

SQRT5  = math.sqrt(5.0)
KAPPA  = 3.0 / (3.0 - SQRT5)
DELTA  = math.sqrt((KAPPA / 2.0) ** 2 + 4.0) - KAPPA / 2.0
DELTA_S = 0.4862544125
BAND   = 0.12
EPS    = 1e-6
HERE   = os.path.dirname(os.path.abspath(__file__))

# --- single-tile typologies, verbatim from the validated model gate -------
TILE = {
    "null_native": [],
    "aml_ring4":   [(0, 1), (1, 2), (2, 3)],
    "aml_ring6":   [(0, 1), (1, 2), (2, 4), (4, 5), (5, 3)],
    "smurf_star":  [(0, 1), (0, 2), (0, 4)],
}

def layered(name, k):
    """k interleaved copies of a tile, coupled through a shared placement tier.

    Layer l occupies qubits {l, l+k, ..., l+5k}, so the global native pairing
    (i, i+3k) reproduces the tile's own pairs (0,3),(1,4),(2,5) in every
    layer.  Placement accounts of adjacent layers are linked (l, l+1):
    that link is what makes the layers one network rather than k tiles."""
    n = 6 * k
    native = [(i, i + 3 * k) for i in range(3 * k)]
    extra, place = [], []
    for l in range(k):
        v = [l + j * k for j in range(6)]
        extra += [(v[a], v[b]) for a, b in TILE[name]]
    for l in range(k - 1):
        place.append((l, l + 1))
    return n, native, extra, place

# ---------------------------------------------------- matrix-free operator
def diag_D(n, edges):
    idx = np.arange(1 << n, dtype=np.int64)
    d = np.zeros(1 << n)
    for a, b in sorted({(min(x, y), max(x, y)) for x, y in edges if x != y}):
        d += 0.5 * (1.0 - (1 - 2 * ((idx >> a) & 1)) * (1 - 2 * ((idx >> b) & 1)))
    return KAPPA * d

def apply_H(psi, n, d):
    out = d * psi
    v = psi.reshape([2] * n)
    for i in range(n):
        out -= np.flip(v, axis=i).reshape(-1)
    return out

def pair_ground():
    """Exact 2-qubit ground state of h = kappa/2 (I - Z Z) - X - X."""
    h = np.zeros((4, 4))
    for s in range(4):
        za, zb = 1 - 2 * (s & 1), 1 - 2 * ((s >> 1) & 1)
        h[s, s] = (KAPPA / 2.0) * (1.0 - za * zb)
        h[s ^ 1, s] -= 1.0
        h[s ^ 2, s] -= 1.0
    w, v = np.linalg.eigh(h)
    return float(w[0]), v[:, 0]

def ground_state(n, native):
    """Native ground state in closed form: the pairs commute and cover every
    qubit, so the state is an exact tensor product -- 1 CNOT per pair on the
    device, and no iterative solve here."""
    e1, g = pair_ground()
    idx = np.arange(1 << n, dtype=np.int64)
    psi = np.ones(1 << n)
    for a, b in native:
        psi *= g[(((idx >> a) & 1) | (((idx >> b) & 1) << 1))]
    psi /= np.linalg.norm(psi)
    return e1 * len(native), psi.astype(np.complex128)

def lanczos_step(psi, n, d, dt, m=20):
    V = np.empty((m, psi.size), dtype=np.complex128)
    a = np.zeros(m); b = np.zeros(m)
    beta = np.linalg.norm(psi); V[0] = psi / beta; j = m
    for i in range(m):
        w = apply_H(V[i], n, d)
        a[i] = np.vdot(V[i], w).real
        Vi = V[: i + 1]
        w -= Vi.T @ (Vi.conj() @ w)              # full reorthogonalisation
        nb = np.linalg.norm(w)
        if nb < 1e-13:
            j = i + 1; break
        b[i] = nb
        if i + 1 < m:
            V[i + 1] = w / nb
    T = np.diag(a[:j]) + np.diag(b[: j - 1], 1) + np.diag(b[: j - 1], -1)
    ev, U = np.linalg.eigh(T)
    return beta * ((U @ (np.exp(-1j * ev * dt) * U[0].conj())) @ V[:j])

# ------------------------------------------------------------- observables
def spectral_weight(times, L, lo, hi, n_pad=8):
    x = np.asarray(L, dtype=np.complex128)        # COMPLEX -> eigenvalues
    y = np.fft.fftshift(np.fft.fft(x * np.hanning(len(x)), n=n_pad * len(x)))
    om = np.fft.fftshift(np.fft.fftfreq(len(y), d=float(times[1] - times[0]))) * 2 * np.pi
    S = np.abs(y) ** 2; S = S / S.sum()
    m = (om >= lo) & (om <= hi)
    return float(S[m].sum())

def chi_at(psi, n, eps=EPS):
    c = n // 2
    s = np.linalg.svd(psi.reshape(1 << (n - c), 1 << c), compute_uv=False)
    p = s ** 2; p = p / p.sum()
    tail = np.cumsum(p[::-1])[::-1]
    chi = int(np.searchsorted(-tail, -eps) + 1)
    nz = p[p > 1e-15]
    return max(1, min(chi, p.size)), float(-(nz * np.log(nz)).sum())

# -------------------------------------------------------------------- main
def measure(name, k, T=32.0, npts=321, T_chi=8.0):
    n, native, extra, place = layered(name, k)
    edges = native + extra + place
    d_nat = diag_D(n, native)
    d_all = diag_D(n, edges)
    E0, gs = ground_state(n, native)            # exact native prep (closed form)
    times = np.linspace(0.0, T, npts)
    dt = times[1] - times[0]
    psi = gs.copy(); L = [np.vdot(gs, psi)]
    chi_max, S_max = chi_at(gs, n)[0], chi_at(gs, n)[1]
    for t in times[1:]:
        psi = lanczos_step(psi, n, d_all, dt)
        L.append(np.vdot(gs, psi))
        if t <= T_chi and (len(L) % 8 == 1):      # chi wall on a bounded window
            c, s = chi_at(psi, n)
            chi_max = max(chi_max, c); S_max = max(S_max, s)
    L = np.array(L)
    # leading response line: the lowest-frequency peak the device actually sees
    y = np.fft.fftshift(np.fft.fft(L * np.hanning(len(L)), n=8 * len(L)))
    om = np.fft.fftshift(np.fft.fftfreq(len(y), d=dt)) * 2 * np.pi
    S = np.abs(y) ** 2; S = S / S.sum()
    m = om > 0.05
    o1, s1 = om[m], S[m]
    lead = int(np.argmax(s1))
    return {
        "k": k, "n_qubits": n, "edges": len(edges),
        "native_pairs": len(native), "typology_edges": len(extra),
        "placement_links": len(place),
        "E0_native": E0,
        "two_qubit_gates": 3 * k + 2 * (len(extra) + len(place)),
        "chi_max": int(chi_max), "chi_cap": 1 << (n // 2),
        "S_max_nats": S_max,
        "weight_on_gap_line": spectral_weight(times, L, DELTA_S - BAND, DELTA_S + BAND),
        "echo_mean_abs": float(np.mean(np.abs(L))),
        "lead_omega": float(o1[lead]),
        "lead_height": float(s1[lead]),
        "spectrum_omega": o1[::16].tolist(),
        "spectrum_S": s1[::16].tolist(),
    }

if __name__ == "__main__":
    if len(sys.argv) > 2:                       # single family, for parallel runs
        fam, kmax = sys.argv[1], int(sys.argv[2])
        rows = []
        for k in range(1, kmax + 1):
            t0 = time.time(); r = measure(fam, k); r["seconds"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"  {fam:12s} k={k} n={r['n_qubits']:2d} E={r['edges']:2d} "
                  f"2q={r['two_qubit_gates']:3d} chi={r['chi_max']:4d}/{r['chi_cap']:<5d} "
                  f"S={r['S_max_nats']:.3f} w1={r['lead_omega']:.4f}"
                  f" wgap={r['weight_on_gap_line']:.5f} ({r['seconds']}s)", flush=True)
        json.dump(rows, open(os.path.join(HERE, "results", f"layer_{fam}.json"), "w"), indent=1)
        sys.exit(0)
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"layering wall  kappa={KAPPA:.9f}  Delta_s={DELTA_S}  eps={EPS}")
    out = {"kappa": KAPPA, "delta": DELTA, "delta_s": DELTA_S,
           "truncation_eps": EPS, "band": BAND, "families": {}}
    for name in TILE:
        rows = []
        for k in range(1, kmax + 1):
            t0 = time.time(); r = measure(name, k); r["seconds"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"  {name:12s} k={k} n={r['n_qubits']:2d} E={r['edges']:2d} "
                  f"2q={r['two_qubit_gates']:3d} chi={r['chi_max']:4d}/{r['chi_cap']:<5d} "
                  f"S={r['S_max_nats']:.3f} w1={r['lead_omega']:.4f} "
                  f"({r['seconds']}s)", flush=True)
        out["families"][name] = rows
    p = os.path.join(HERE, "results", "hsbc_layering_wall.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)
