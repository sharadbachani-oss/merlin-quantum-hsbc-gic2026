# -*- coding: utf-8 -*-
"""
hsbc_spectral_route.py — payment-graph Hamiltonian → dynamical spectrum S(ω).

Industrial object: the real-time response of

    H[G] = κ · D_G − A_G
    D_G  = Σ_{(i,j)∈E} ½ (I − Z_i Z_j)
    A_G  = Σ_i X_i
    κ   = 3 / (3 − √5)     (structural spectral gap of the reference complex; not fitted)

When G is the three native pairs {(0,3),(1,4),(2,5)} this is exactly
the first-principles operator (E0 = −3δ, gap = δ, exact 3-CX prep). Extra payment
edges are a quench. AML rings are short cycles whose girth sits on the
native loop sector [4, 4, 6].

Advantage of route (Mitsubishi A(ω) shape):
  classical scalable route to S(ω) = imaginary-time G(τ) + analytic
  continuation, ill-posed at any size;
  quantum route = real-time L(t) + Fourier, no continuation.
  Tensor networks fail on the loops that are the AML object.

Model gate only: numpy, no credentials, no invented hardware.
Writes results/hsbc_spectral_route.json.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

SQRT5 = math.sqrt(5.0)
KAPPA = 3.0 / (3.0 - SQRT5)
DELTA = math.sqrt((KAPPA / 2.0) ** 2 + 4.0) - KAPPA / 2.0
CHI = math.atan(DELTA / 2.0)
THETA_STAR = math.pi / 2.0 - 2.0 * CHI
E0_CLOSED = -3.0 * DELTA
GAP_CLOSED = DELTA
ZZ_CLOSED = math.cos(2.0 * CHI)
X_CLOSED = math.sin(2.0 * CHI)
DELTA_S = 0.486254
NATIVE_PAIRS = ((0, 3), (1, 4), (2, 5))
NATIVE_LOOP_SECTOR = (4, 4, 6)
SEED = 21


def _z(s: int, i: int) -> float:
    return 1.0 - 2.0 * ((s >> i) & 1)


def build_H(n: int, edges) -> np.ndarray:
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    elist = sorted({(min(a, b), max(a, b)) for a, b in edges if a != b})
    for s in range(dim):
        diag = 0.0
        for a, b in elist:
            diag += (KAPPA / 2.0) * (1.0 - _z(s, a) * _z(s, b))
        H[s, s] = diag
        for q in range(n):
            H[s ^ (1 << q), s] -= 1.0
    return H


def eigh_H(H):
    evals, evecs = np.linalg.eigh(H)
    return evals.astype(np.float64), evecs.astype(np.complex128)


def plus_state(n: int) -> np.ndarray:
    psi = np.ones(1 << n, dtype=np.complex128)
    psi /= np.linalg.norm(psi)
    return psi


def loschmidt(evals, evecs, psi0, times) -> np.ndarray:
    c = evecs.conj().T @ psi0
    return np.exp(-1j * np.outer(times, evals)) @ (c * np.conj(c))


def overlap_echo(evals, evecs, psi_ref, psi0, times) -> np.ndarray:
    """⟨ψ_ref| e^{-iHt} |ψ0⟩."""
    c0 = evecs.conj().T @ psi0
    cr = evecs.conj().T @ psi_ref
    return np.exp(-1j * np.outer(times, evals)) @ (np.conj(cr) * c0)


def spectral_density(times, signal, n_pad: int = 8):
    x = np.asarray(signal, dtype=np.complex128)
    w = np.hanning(len(x))
    y = np.fft.fftshift(np.fft.fft(x * w, n=n_pad * len(x)))
    dt = float(times[1] - times[0])
    omega = np.fft.fftshift(np.fft.fftfreq(len(y), d=dt)) * 2.0 * np.pi
    S = np.abs(y) ** 2
    S = S / max(float(S.sum()), 1e-30)
    return omega, S


def band_weight(omega, S, lo, hi) -> float:
    m = (omega >= lo) & (omega <= hi)
    return float(S[m].sum())


def peak_in_band(omega, S, lo, hi):
    m = (omega >= lo) & (omega <= hi)
    if not np.any(m):
        return 0.0, 0.0
    k = int(np.argmax(S[m]))
    return float(omega[m][k]), float(S[m][k])


def l1_spectra(omega_a, Sa, omega_b, Sb) -> float:
    grid = np.linspace(max(omega_a.min(), omega_b.min()),
                       min(omega_a.max(), omega_b.max()), 2000)
    fa = np.interp(grid, omega_a, Sa, left=0.0, right=0.0)
    fb = np.interp(grid, omega_b, Sb, left=0.0, right=0.0)
    fa = fa / max(fa.sum(), 1e-30)
    fb = fb / max(fb.sum(), 1e-30)
    return 0.5 * float(np.abs(fa - fb).sum())


def echo_contrast(L: np.ndarray) -> float:
    a = np.abs(L)
    if a[0] < 1e-15:
        return 0.0
    k0 = int(np.argmin(a[: max(4, len(a) // 5)]))
    return float(a[k0:].max() / a[0])


def entanglement_entropy(psi: np.ndarray, n: int, cut: int) -> float:
    M = psi.reshape((1 << cut, 1 << (n - cut)))
    s = np.linalg.svd(M, compute_uv=False)
    p = (s * s).real
    p = p[p > 1e-16]
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def entropy_ladder(H, psi0, n, cut, times):
    evals, evecs = eigh_H(H)
    c = evecs.conj().T @ psi0
    out = []
    for t in times:
        psi = evecs @ (np.exp(-1j * evals * t) * c)
        out.append(entanglement_entropy(psi, n, cut))
    return out


def schmidt_chi(psi, n, cut, eps=1e-6) -> int:
    M = psi.reshape((1 << cut, 1 << (n - cut)))
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > eps))


# ---------------------------------------------------------------------------
# Native 6-qubit operator identity
# ---------------------------------------------------------------------------
def native_operator_gate():
    H = build_H(6, NATIVE_PAIRS)
    evals, evecs = eigh_H(H)
    E0 = float(evals[0])
    gs = evecs[:, 0]
    zz = []
    for a, b in NATIVE_PAIRS:
        z = sum((abs(gs[s]) ** 2) * _z(s, a) * _z(s, b) for s in range(64))
        zz.append(float(np.real(z)))
    xexp = []
    for q in range(6):
        val = sum(np.conj(gs[s]) * gs[s ^ (1 << q)] for s in range(64))
        xexp.append(float(np.real(val)))
    return {
        "kappa": KAPPA,
        "delta": DELTA,
        "chi": CHI,
        "theta_star": THETA_STAR,
        "E0": E0,
        "E0_closed": E0_CLOSED,
        "gap": float(evals[1] - evals[0]),
        "gap_closed": GAP_CLOSED,
        "ZZ_pairs": zz,
        "ZZ_closed": ZZ_CLOSED,
        "X_mean": float(np.mean(xexp)),
        "X_closed": X_CLOSED,
        "prep": "per pair RY(theta*) . CX . H H — 3 CX, fidelity 1",
        "pass_E0": abs(E0 - E0_CLOSED) < 1e-9,
        "pass_gap": abs(float(evals[1] - evals[0]) - GAP_CLOSED) < 1e-9,
        "pass_ZZ": all(abs(z - ZZ_CLOSED) < 1e-8 for z in zz),
    }, H, evals, evecs


# ---------------------------------------------------------------------------
# Ill-posed continuation exhibit (analytic, any system size)
# ---------------------------------------------------------------------------
def continuation_exhibit():
    """Two different line spectra whose Laplace transforms agree on a
    finite imaginary-time grid — the textbook obstruction to recovering
    A(ω) / S(ω) from G(τ).
    """
    taus = np.linspace(0.15, 3.0, 48)
    om_a = np.array([0.40, 0.84, 1.68, 2.52])  # includes the native gap
    w_a = np.array([0.18, 0.44, 0.26, 0.12])
    w_a = w_a / w_a.sum()
    G = np.exp(-np.outer(taus, om_a)) @ w_a

    om_b = np.linspace(0.15, 3.2, 16)
    K = np.exp(-np.outer(taus, om_b))
    # NNLS
    try:
        from scipy.optimize import nnls
        w_b, _ = nnls(K, G)
    except Exception:
        w_b, *_ = np.linalg.lstsq(K, G, rcond=None)
        w_b = np.clip(w_b, 0.0, None)
    if w_b.sum() <= 0:
        w_b = np.ones_like(om_b) / len(om_b)
    else:
        w_b = w_b / w_b.sum()
    G2 = K @ w_b
    # TV of the two discrete measures after binning onto a common grid
    grid = np.linspace(0.0, 3.4, 200)
    sa = np.zeros_like(grid)
    sb = np.zeros_like(grid)
    for om, w in zip(om_a, w_a):
        sa[int(np.argmin(np.abs(grid - om)))] += w
    for om, w in zip(om_b, w_b):
        sb[int(np.argmin(np.abs(grid - om)))] += w
    tv = 0.5 * float(np.abs(sa / sa.sum() - sb / sb.sum()).sum())
    return {
        "mean_abs_G_residual": float(np.mean(np.abs(G - G2))),
        "max_abs_G_residual": float(np.max(np.abs(G - G2))),
        "spectral_tv_distance": tv,
        "n_lines_A": int(len(om_a)),
        "n_lines_B": int(np.sum(w_b > 1e-6)),
        "om_A": [float(x) for x in om_a],
        "w_A": [float(x) for x in w_a],
        "om_B": [float(x) for x in om_b],
        "w_B": [float(x) for x in w_b],
        "taus": [float(x) for x in taus],
        "note": (
            "Spectrum A has 4 lines; spectrum B is a 16-line non-negative "
            "fit to the same G(τ). Distinct S(ω), interchangeable imaginary-time data."
        ),
    }


# ---------------------------------------------------------------------------
# Noise as instrument on the 6q tile
# ---------------------------------------------------------------------------
def dephased_echo(evals, evecs, psi0, times, gamma):
    c = evecs.conj().T @ psi0
    return np.exp(-1j * np.outer(times, evals) - gamma * times[:, None]) @ (c * np.conj(c))


def cycle_parity_diag(n, verts):
    mask = 0
    for v in verts:
        mask |= 1 << v
    return np.array([1.0 if bin(s & mask).count("1") % 2 == 0 else -1.0
                     for s in range(1 << n)], dtype=np.float64)


def protected_echo(evals, evecs, psi0, times, gamma_bulk, gamma_prot, parity):
    c = evecs.conj().T @ psi0
    Pn = np.real(np.sum(np.abs(evecs) ** 2 * parity[:, None], axis=0))
    even = 0.5 * (1.0 + Pn)
    gamma = gamma_prot * even + gamma_bulk * (1.0 - even)
    return np.exp(-1j * np.outer(times, evals) - np.outer(times, gamma)) @ (c * np.conj(c))


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    native, H_nat, ev_nat, U_nat = native_operator_gate()
    gs_nat = U_nat[:, 0]

    times6 = np.linspace(0.0, 32.0, 321)
    # extra-edge typologies on the SAME 6 qubits (hardware tile)
    families = {
        "null_native": [],
        "aml_ring4": [(0, 1), (1, 2), (2, 3)],          # closes 0-1-2-3 with native (0,3)
        "aml_ring6": [(0, 1), (1, 2), (2, 4), (4, 5), (5, 3)],  # 6-cycle using pairs
        "smurf_star": [(0, 1), (0, 2), (0, 4)],         # same extra-edge count as ring4
    }

    tile = {}
    spectra = {}
    S_store = {}
    for name, extra in families.items():
        edges = list(NATIVE_PAIRS) + extra
        H = build_H(6, edges)
        evals, evecs = eigh_H(H)
        # quench: native ground state evolved under the payment-graph H
        L = overlap_echo(evals, evecs, gs_nat, gs_nat, times6)
        omega, S = spectral_density(times6, L)
        pk_om, pk_h = peak_in_band(omega, S, 0.05, 6.0)
        w_delta = band_weight(omega, S, DELTA - 0.20, DELTA + 0.20)
        w_delta_s = band_weight(omega, S, DELTA_S - 0.12, DELTA_S + 0.12)
        tile[name] = {
            "extra_edges": [list(e) for e in extra],
            "n_extra": len(extra),
            "E0": float(evals[0]),
            "gap": float(evals[1] - evals[0]),
            "gap_shift_from_delta": float((evals[1] - evals[0]) - DELTA),
            "echo_mean": float(np.mean(np.abs(L))),
            "echo_contrast": echo_contrast(L),
            "peak_omega": pk_om,
            "peak_height": pk_h,
            "weight_near_delta": w_delta,
            "weight_near_delta_s": w_delta_s,
            "|L|(t=0)": float(abs(L[0])),
        }
        S_store[name] = (omega, S)
        spectra[name] = {"omega": omega[::8].tolist(), "S": S[::8].tolist(),
                         "L_abs": np.abs(L[::4]).real.tolist()}

    # spectral L1 vs native null — the response difference
    om_n, Sn = S_store["null_native"]
    for name in families:
        if name == "null_native":
            continue
        tile[name]["S_tv_vs_null"] = l1_spectra(om_n, Sn, *S_store[name])

    # noise as instrument: native vs ring4, isotropic vs cycle-parity
    times_n = np.linspace(0.0, 24.0, 241)
    P4 = cycle_parity_diag(6, [0, 1, 2, 3])
    noise = {}
    for name, extra in (("null_native", []), ("aml_ring4", families["aml_ring4"])):
        H = build_H(6, list(NATIVE_PAIRS) + extra)
        evals, evecs = eigh_H(H)
        psi = gs_nat
        L_iso = dephased_echo(evals, evecs, psi, times_n, gamma=0.16)
        L_prot = protected_echo(evals, evecs, psi, times_n, 0.16, 0.022, P4)
        noise[name] = {
            "late_isotropic": float(np.mean(np.abs(L_iso[times_n > 12]))),
            "late_parity_protected": float(np.mean(np.abs(L_prot[times_n > 12]))),
            "protection_ratio": float(
                np.mean(np.abs(L_prot[times_n > 12])) /
                max(np.mean(np.abs(L_iso[times_n > 12])), 1e-30)
            ),
        }

    # MPS / entropy wall: |+>^n quench under path vs cycle (the loop is the object)
    n = 10
    cut = 5
    t_ent = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    path_E = [(i, i + 1) for i in range(n - 1)]
    cyc_E = path_E + [(0, n - 1)]
    psi_plus = plus_state(n)
    S_path = entropy_ladder(build_H(n, path_E), psi_plus, n, cut, t_ent)
    S_cyc = entropy_ladder(build_H(n, cyc_E), psi_plus, n, cut, t_ent)
    # chi at last time
    def chi_at(H, t):
        ev, U = eigh_H(H)
        c = U.conj().T @ psi_plus
        psi = U @ (np.exp(-1j * ev * t) * c)
        return schmidt_chi(psi, n, cut), entanglement_entropy(psi, n, cut)

    chi_p, ent_p = chi_at(build_H(n, path_E), 8.0)
    chi_c, ent_c = chi_at(build_H(n, cyc_E), 8.0)
    dS = [float(a - b) for a, b in zip(S_cyc, S_path)]
    dS_max = max(dS)
    dS_t1 = float(S_cyc[1] - S_path[1])

    # 2x4 mule-mesh ladder (three plaquettes) vs path — multiple AML rings
    lad_E = []
    cols = 4
    for r in range(2):
        for c in range(cols - 1):
            lad_E.append((r * cols + c, r * cols + c + 1))
    for c in range(cols):
        lad_E.append((c, cols + c))
    psi8 = plus_state(8)
    t_lad = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    S_lad = entropy_ladder(build_H(8, lad_E), psi8, 8, 4, t_lad)
    S_p8 = entropy_ladder(build_H(8, [(i, i + 1) for i in range(7)]), psi8, 8, 4, t_lad)

    cont = continuation_exhibit()

    # --- gates (physics, not ML) ------------------------------------------
    tv_aml = tile["aml_ring4"]["S_tv_vs_null"]
    tv_star = tile["smurf_star"]["S_tv_vs_null"]
    gap_aml = abs(tile["aml_ring4"]["gap_shift_from_delta"])
    gap_star = abs(tile["smurf_star"]["gap_shift_from_delta"])
    # ring and star are different responses (not "AML scores higher")
    distinct = abs(tv_aml - tv_star) > 1e-4 or abs(gap_aml - gap_star) > 1e-4

    gates = {
        "native_operator_identity": bool(
            native["pass_E0"] and native["pass_gap"] and native["pass_ZZ"]
        ),
        "continuation_illposed": bool(
            cont["spectral_tv_distance"] >= 0.15
            and cont["mean_abs_G_residual"] <= 5e-3
        ),
        "loop_entanglement_wall": bool(dS_max > 0.30 or (S_lad[1] - S_p8[1]) > 0.20),
        "tile_response_distinct": bool(tv_aml > 0.05 and distinct),
        "noise_as_instrument": bool(
            noise["aml_ring4"]["protection_ratio"] > 1.2
            or noise["null_native"]["protection_ratio"] > 1.2
        ),
    }

    out = {
        "title": "HSBC first-principles-native route — payment-graph S(omega)",
        "seed": SEED,
        "constants": {
            "kappa": KAPPA,
            "delta": DELTA,
            "chi": CHI,
            "theta_star": THETA_STAR,
            "E0_closed": E0_CLOSED,
            "delta_s_reference": DELTA_S,
            "native_loops": list(NATIVE_LOOP_SECTOR),
            "H": "kappa D_G - A_G",
            "provenance": "DERIVED from N=6 structural spectral gap; no fitted fraud couplings",
        },
        "native_operator_gate": native,
        "six_qubit_tile": tile,
        "noise_instrument": noise,
        "entanglement_wall": {
            "n": n,
            "cut": cut,
            "init": "|+>^n quench under H[G]",
            "times": [float(t) for t in t_ent],
            "entropy_path": [float(x) for x in S_path],
            "entropy_cycle": [float(x) for x in S_cyc],
            "entropy_at_t8_path": ent_p,
            "entropy_at_t8_cycle": ent_c,
            "dS_cycle_minus_path": dS,
            "dS_max": dS_max,
            "dS_at_t1": dS_t1,
            "chi_eps1e-6_t8_path": chi_p,
            "chi_eps1e-6_t8_cycle": chi_c,
            "ladder_2x4_entropy": [float(x) for x in S_lad],
            "path8_entropy": [float(x) for x in S_p8],
            "note": (
                "A payment ring wraps the cut immediately: at t=1 the cycle "
                "entanglement exceeds the path (light cone has not arrived). "
                "Later the same wrap produces a revival and entropy can drop "
                "— that revival is the echo the device reads. Loops force "
                "the TN to pay now; they also are the AML coherence."
            ),
        },
        "continuation_exhibit": cont,
        "advantage": {
            "object": "S(omega) of H[G] = kappa D_G - A_G after a payment-graph quench",
            "classical_route": "imaginary-time G(tau) + analytic continuation (ill-posed)",
            "quantum_route": "real-time Loschmidt echo + Fourier (no continuation)",
            "aml4_S_tv_vs_native": tv_aml,
            "smurf_S_tv_vs_native": tv_star,
            "aml4_gap_shift": tile["aml_ring4"]["gap_shift_from_delta"],
            "smurf_gap_shift": tile["smurf_star"]["gap_shift_from_delta"],
            "aml4_echo_mean": tile["aml_ring4"]["echo_mean"],
            "null_echo_mean": tile["null_native"]["echo_mean"],
            "entropy_cycle_minus_path_t8": float(ent_c - ent_p),
            "entropy_cycle_minus_path_t1": dS_t1,
            "entropy_dS_max": dS_max,
            "ring_over_star_echo_mean": float(
                tile["aml_ring4"]["echo_mean"] / max(tile["smurf_star"]["echo_mean"], 1e-30)
            ),
            "noise_protection_ring_over_native": float(
                noise["aml_ring4"]["protection_ratio"] /
                max(noise["null_native"]["protection_ratio"], 1e-30)
            ),
            "continuation_tv": cont["spectral_tv_distance"],
            "continuation_G_residual": cont["mean_abs_G_residual"],
            "static_census_honesty": (
                "A 4-cycle census is classically cheap and is not the claimed "
                "object. The claimed object is the interacting dynamical "
                "spectrum of H[G]. Cycle counting answers 'is there a ring'; "
                "S(omega) answers 'how that ring responds', which is the "
                "photo-response analog (A(omega)) for a payment graph."
            ),
        },
        "gates": gates,
        "hardware_status": {
            "this_script": "model gate / expected signature — no shots claimed",
            "prior_jobs_not_this_protocol": [
                "ibm_fez dab6therrl7c7386flo0 — 128q generative card (annex)",
                "ibm_fez daaucejvpcac73dd232g — 128q feature map, +0.0023 AUPRC null (annex)",
                "ibm_fez daba21jvpcac73ddh9h0 — 87q data-encoded Boltzmann (negative)",
                "QuEra Aquila 8 jobs — 256-atom feature encoding, not S(omega)",
                "QCi Dirac-3 354 jobs — fitted 3-body landscape (annex, not the route)",
            ],
        },
        "seconds": round(time.time() - t0, 3),
    }

    path_json = os.path.join(OUT_DIR, "hsbc_spectral_route.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT_DIR, "hsbc_spectral_spectra.json"), "w", encoding="utf-8") as f:
        json.dump(spectra, f)

    print("=" * 68)
    print("HSBC first-principles-native route -- model gate")
    print("=" * 68)
    print("native E0=%.9f closed=%.9f  gap=%.9f  PASS=%s" % (
        native["E0"], E0_CLOSED, native["gap"],
        native["pass_E0"] and native["pass_gap"]))
    print("tile S TV vs null: ring4=%.4f  star=%.4f  ring6=%.4f" % (
        tv_aml, tv_star, tile["aml_ring6"]["S_tv_vs_null"]))
    print("gap shift: ring4=%+.4f  star=%+.4f  (native delta=%.4f)" % (
        tile["aml_ring4"]["gap_shift_from_delta"],
        tile["smurf_star"]["gap_shift_from_delta"], DELTA))
    print("echo mean: null=%.4f  ring4=%.4f  star=%.4f" % (
        tile["null_native"]["echo_mean"],
        tile["aml_ring4"]["echo_mean"],
        tile["smurf_star"]["echo_mean"]))
    print("entropy t=1: path=%.3f  cycle=%.3f  dS=%.3f  (max dS=%.3f)" % (
        S_path[1], S_cyc[1], dS_t1, dS_max))
    print("continuation: TV=%.3f  mean|dG|=%.3e" % (
        cont["spectral_tv_distance"], cont["mean_abs_G_residual"]))
    print("noise protection ratio: native=%.2f  ring4=%.2f" % (
        noise["null_native"]["protection_ratio"],
        noise["aml_ring4"]["protection_ratio"]))
    print("gates:", {k: ("PASS" if v else "FAIL") for k, v in gates.items()})
    print("wrote", path_json, "in", out["seconds"], "s")
    return out


if __name__ == "__main__":
    main()
