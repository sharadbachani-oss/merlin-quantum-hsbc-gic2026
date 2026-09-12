# -*- coding: utf-8 -*-
"""Dirac-3 assessed for this route, and declined -- with the reason measured.

THE QUESTION.  Every time point in the response record is a separate
circuit on the gate device, so record length is the quantum bill.
Fourier peak-picking is limited by 2*pi/T: at one layer the girth-4 and
star lines sit 0.0304 apart, which puts the Rayleigh limit at T >= 206.8.
Recovering those lines from a shorter record is a constrained
deconvolution,

    min || A w - y ||^2     over  w >= 0,  sum(w) = S

with A[t,j] = exp(-i E_j t) on a candidate grid fixed by the DERIVED
operator scale -- topology-agnostic, since topology is what we infer, and
the Born weights live on the simplex by construction rather than by
imposition.  That is exactly Dirac-3's native form, and the same shape
already flown on this hardware family in the Cleveland track.

THE FINDING.  The formulation works: it reaches 94.4% recovery at 24
circuits where Fourier needs 128 for 100%, roughly a 2x to 5x reduction in
circuits per graded subgraph.

THE REASON DIRAC IS DECLINED.  That objective is a positive-semidefinite
quadratic over a simplex.  It is convex, so it is solved to global
optimality in polynomial time by a laptop -- the numbers below come from
scipy's non-negative least squares.  There is no hardness for a sampler to
exploit, and the convex solve already reaches 100%, so a non-convex
cardinality-constrained variant has no accuracy headroom left to capture.

Flying Dirac here would produce a job receipt, not an advantage, and it
would compute the object class this submission argues away from.  The
instance and a frozen pre-registration are emitted anyway so the
assessment is auditable rather than asserted.  NOT FLOWN.
"""
import hashlib, json, os, sys, time
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    "rg", os.path.join(HERE, "hsbc_response_grading.py"))
rg = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rg)
lw = rg.lw
FAMS = ["aml_ring6", "aml_ring4", "smurf_star"]


def true_lines(name, k, m=160):
    """Ritz values and Born weights of H[G] in the prepared state."""
    n, nat, ex, pl = lw.layered(name, k)
    d = lw.diag_D(n, nat + ex + pl)
    _, g = lw.ground_state(n, nat)
    N = 1 << n
    V = np.empty((m, N))
    a = np.zeros(m)
    b = np.zeros(m)
    V[0] = g.real / np.linalg.norm(g.real)
    j = m
    for i in range(m):
        w = lw.apply_H(V[i].astype(np.complex128), n, d).real
        a[i] = float(V[i] @ w)
        Vi = V[: i + 1]
        w -= Vi.T @ (Vi @ w)
        nb = float(np.linalg.norm(w))
        if nb < 1e-10:
            j = i + 1
            break
        b[i] = nb
        if i + 1 < m:
            V[i + 1] = w / nb
    T = np.diag(a[:j]) + np.diag(b[: j - 1], 1) + np.diag(b[: j - 1], -1)
    ev, U = np.linalg.eigh(T)
    return ev, U[0] ** 2


def record(ev, wt, P, dt, shots, rng):
    """L(t) over P circuits, each estimated from a finite number of shots."""
    t = np.arange(P) * dt
    L = (wt[None, :] * np.exp(-1j * np.outer(t, ev))).sum(axis=1)
    if shots:
        s = 1.0 / float(np.sqrt(shots))
        L = L + rng.normal(0, s, P) + 1j * rng.normal(0, s, P)
    return t, L


def fourier_w1(t, L, lo=0.05, n_pad=16, hi=None):
    x = L * np.hanning(len(L))
    y = np.fft.fftshift(np.fft.fft(x, n=n_pad * len(L)))
    om = np.fft.fftshift(np.fft.fftfreq(len(y), d=t[1] - t[0])) * 2 * np.pi
    S = np.abs(y) ** 2
    m = (om > lo) if hi is None else ((om > lo) & (om <= hi))
    return float(om[m][int(np.argmax(S[m]))])


def grid_from_operator(k, name="aml_ring6", fine=0.01, coarse=0.25):
    """Candidate ENERGIES for H[G], fixed by the derived operator scale.

    Bound band E in [-3*delta*k*1.15, 0] gets fine spacing because that is
    where the typology grade lives; the free band up to the operator-norm
    bound kappa*|E_edges| + n gets coarse spacing so the model can account
    for the weight that sits there.  Neither depends on the graph's
    topology, which is what we are inferring."""
    n, nat, ex, pl = lw.layered(name, k)
    lo = -3.0 * lw.DELTA * k * 1.15
    hi = lw.KAPPA * len(nat + ex + pl) + n
    return np.concatenate([np.arange(lo, 0.0, fine),
                           np.arange(0.0, hi + coarse, coarse)])


def deconvolve_w1(t, L, grid, S=1.0, penalty=50.0, thresh=0.02):
    """min ||Aw - y||^2 with w >= 0 and sum(w) = S, A[t,j] = exp(-i E_j t).

    Convex reference solve.  The same objective is what Dirac-3 receives as
    a degree-2 polynomial over simplex-constrained variables."""
    from scipy.optimize import nnls
    A = np.exp(-1j * np.outer(t, grid))
    M = np.vstack([A.real, A.imag, penalty * np.ones((1, len(grid)))])
    y = np.concatenate([L.real, L.imag, [penalty * S]])
    w, _ = nnls(M, y)
    if w.sum() <= 0:
        return float("nan"), w
    bound = grid < 0                                  # the graded band
    wb = np.where(bound, w, 0.0)
    if wb.max() <= thresh * w.max():
        return float("nan"), w
    return float(-grid[int(np.argmax(wb))]), w


def sweep(k=1, dt=0.10, shots=8192, trials=12, tol=0.015, seed=21):
    rng = np.random.default_rng(seed)
    truth = {f: true_lines(f, k) for f in FAMS}
    w1_true = {}
    for f in FAMS:
        ev, wt = truth[f]
        neg = np.where(ev < 0)[0]
        w1_true[f] = -float(ev[neg[np.argmax(wt[neg])]])
    print("  true leading lines:  " +
          "  ".join("%s=%.4f" % (f.split("_")[-1], w1_true[f]) for f in FAMS))
    sep = abs(w1_true["aml_ring4"] - w1_true["smurf_star"])
    print("  hard pair separation = %.4f  ->  Rayleigh limit T >= %.1f"
          % (sep, 2 * np.pi / sep))
    grid = grid_from_operator(k)
    hi_band = 3.0 * lw.DELTA * k * 1.15
    out = {"k": k, "dt": dt, "shots": shots, "trials": trials, "tol": tol,
           "true_w1": w1_true, "hard_pair_separation": sep,
           "rayleigh_T": float(2 * np.pi / sep), "grid_points": len(grid),
           "grid_spacing": 0.02, "sweep": []}
    for P in [16, 24, 32, 48, 64, 128, 256, 512]:
        hit_f = hit_d = n = 0
        for f in FAMS:
            ev, wt = truth[f]
            for _ in range(trials):
                t, L = record(ev, wt, P, dt, shots, rng)
                n += 1
                if abs(fourier_w1(t, L, hi=hi_band) - w1_true[f]) <= tol:
                    hit_f += 1
                if abs(deconvolve_w1(t, L, grid)[0] - w1_true[f]) <= tol:
                    hit_d += 1
        out["sweep"].append({"circuits": P, "record_T": P * dt,
                             "fourier_accuracy": hit_f / n,
                             "deconv_accuracy": hit_d / n})
        print("   P=%3d circuits (T=%5.1f)   Fourier %5.1f%%   "
              "constrained deconvolution %5.1f%%"
              % (P, P * dt, 100 * hit_f / n, 100 * hit_d / n), flush=True)
    return out, grid


def emit_instance(k, dt, P, grid, sum_constraint=1.0):
    """Degree-2 polynomial for Dirac-3: ||Aw - y||^2 over the simplex."""
    ev, wt = true_lines("aml_ring6", k)
    rng = np.random.default_rng(7)
    t, L = record(ev, wt, P, dt, 8192, rng)
    A = np.exp(-1j * np.outer(t, grid))
    Q = (A.conj().T @ A).real
    c = -2.0 * (A.conj().T @ L).real
    terms = []
    for i in range(len(grid)):
        terms.append({"idx": [i], "val": float(c[i])})
        for j in range(i, len(grid)):
            v = float(Q[i, j] * (1.0 if i == j else 2.0))
            if abs(v) > 1e-9:
                terms.append({"idx": [i, j], "val": v})
    return {"route": "hsbc-response-deconvolution", "k": k, "dt": dt,
            "circuits": P, "n_vars": len(grid),
            "sum_constraint": sum_constraint,
            "candidate_grid": grid.tolist(),
            "max_degree": 2, "terms": terms}


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print("Dirac-3 role: response deconvolution at k=%d layers" % k)
    res, grid = sweep(k)
    inst = emit_instance(k, res["dt"], 16, grid)
    ip = os.path.join(HERE, "results", "dirac_superres_instance.json")
    json.dump(inst, open(ip, "w"))
    pre = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "route": inst["route"], "device": "dirac-3", "k": k,
           "n_vars": inst["n_vars"], "sum_constraint": inst["sum_constraint"],
           "num_samples": 20, "relaxation_schedule": 4,
           "gates": {
               "G1": "recovered leading line within 0.10 of the exact value",
               "G2": "objective within 10 percent of the convex reference solve",
               "G3": "girth-4 and star instances resolved apart at P=16 circuits"},
           "reference": {"convex_solver": "scipy nnls with sum penalty"},
           "inst_sha": hashlib.sha256(
               json.dumps(inst["terms"], sort_keys=True).encode()).hexdigest()}
    pre["sha256"] = hashlib.sha256(
        json.dumps(pre, sort_keys=True).encode()).hexdigest()
    json.dump(pre, open(os.path.join(HERE, "results",
                                     "PREREG_dirac_superres.json"), "w"), indent=1)
    res["instance"] = {"path": ip, "n_vars": inst["n_vars"],
                       "n_terms": len(inst["terms"]),
                       "prereg_sha": pre["sha256"]}
    json.dump(res, open(os.path.join(HERE, "results",
                                     "hsbc_dirac_superres.json"), "w"), indent=1)
    print("\n  instance: %d variables, %d terms" % (inst["n_vars"], len(inst["terms"])))
    print("  prereg frozen sha %s  (NOT FLOWN)" % pre["sha256"][:16])
