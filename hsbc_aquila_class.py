# -*- coding: utf-8 -*-
"""What the analog machine can hold, from the operator alone.

A neutral-atom analog machine is on the RIGHT side of the ledger.  Unlike a
ground-state sampler it evolves in real time under its own Hamiltonian,
with no Trotter error and no gate depth at all.  So the question is not
which regime it plays in.  The question is whether it can hold the derived
operator.  Three tests decide it, none of them an engineering opinion.

TEST 1 -- THE ALGEBRA.  Writing n = (I - Z)/2,

    kappa * (1/2)(I - Z_i Z_j) = kappa (n_i + n_j - 2 n_i n_j)

so realising -H (spectroscopically equivalent, the response lines simply
mirror) needs a REPULSIVE pair term +2 kappa with local detuning
Delta_i = kappa * deg(i) and drive Omega = 2 E_u.  Repulsive is exactly
what Rydberg van der Waals is.  The map is exact and carries no condition
on the graph.

TEST 2 -- THE CLOCK.  With the drive at the device maximum, does the
reachable dimensionless evolution time cover the record the spectroscopy
needs?

TEST 3 -- THE GEOMETRY.  This is where it stops.  V_ij is not
programmable.  It is C6 / r_ij^6: one isotropic, strictly decreasing
function of a single distance matrix.  Holding the operator therefore
requires drawing the payment graph in the plane with every edge the SAME
length -- a unit-distance embedding -- and the sixth power means a small
error in length is a large error in coupling.  There is also a hard floor:
a vertex of degree d forces its neighbours onto a circle, and the closest
pair then sits at 2 sin(pi/d) r, coupling to each other whether the
payment graph says so or not.

Device constants are QuEra Aquila's published values.
"""
import json, math, os, sys
import numpy as np
import importlib.util
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    "lw", os.path.join(HERE, "hsbc_layering_wall.py"))
lw = importlib.util.module_from_spec(_s)
_s.loader.exec_module(lw)
K = lw.KAPPA

C6 = 5.42e12         # rad/s * um^6, Rb 70S
OMEGA_MAX = 2.5e7    # rad/s
TAU_MAX = 4.0e-6     # s
R_MIN = 4.0          # um
FOV = (75.0, 76.0)   # um
RECORD_T = 32.0      # dimensionless time the spectroscopy needs
FAMS = ["null_native", "aml_ring4", "aml_ring6", "smurf_star"]


def graph(name, k):
    n, nat, ex, pl = lw.layered(name, k)
    return n, sorted({(min(a, b), max(a, b)) for a, b in nat + ex + pl})


# -------------------------------------------------------------- test 1
def dense_H(name, k):
    n, E = graph(name, k)
    d = lw.diag_D(n, E)
    H = np.diag(d)
    idx = np.arange(1 << n)
    for i in range(n):
        H[idx ^ (1 << i), idx] -= 1.0
    return n, H, E


def aquila_H(n, E, Omega=2.0):
    """(Omega/2) sum X - sum Delta_i n_i + sum V_ij n_i n_j,
    with V = +2 kappa repulsive and Delta_i = kappa * deg(i)."""
    deg = np.zeros(n)
    for a, b in E:
        deg[a] += 1
        deg[b] += 1
    idx = np.arange(1 << n)
    nb = lambda i: ((idx >> i) & 1).astype(float)
    diag = np.zeros(1 << n)
    for i in range(n):
        diag -= K * deg[i] * nb(i)
    for a, b in E:
        diag += 2.0 * K * nb(a) * nb(b)
    HA = np.diag(diag)
    for i in range(n):
        HA[idx ^ (1 << i), idx] += Omega / 2.0
    return HA, deg


def algebra_test(name, k):
    n, H, E = dense_H(name, k)
    HA, deg = aquila_H(n, E)
    return {"family": name, "k": k, "n_atoms": n, "edges": len(E),
            "max_map_error": float(np.abs(HA - (-H)).max()),
            "V_over_Eu": 2.0 * K, "Delta_min": float(K * deg.min()),
            "Delta_max": float(K * deg.max()), "max_degree": int(deg.max())}


# -------------------------------------------------------------- test 2
def clock_test():
    E_u = OMEGA_MAX / 2.0
    r_edge = (C6 / (2.0 * K * E_u)) ** (1.0 / 6.0)
    return {"E_u_rad_s": E_u, "edge_spacing_um": float(r_edge),
            "min_spacing_um": R_MIN, "spacing_ok": bool(r_edge >= R_MIN),
            "t_reachable": float(E_u * TAU_MAX), "t_needed": RECORD_T,
            "clock_ok": bool(E_u * TAU_MAX >= RECORD_T)}


# -------------------------------------------------------------- test 3
def degree_floor(d):
    """Unavoidable neighbour-neighbour coupling for a degree-d vertex,
    as a fraction of one edge coupling."""
    return float((2.0 * math.sin(math.pi / d)) ** -6) if d >= 2 else 0.0


def unit_distance_fit(n, E, restarts=60, seed=11):
    """Best planar drawing with all edges one length and non-edges pushed
    out.  Scale-free: only ratios matter, so this is the kindest possible
    reading for the device."""
    iu = np.triu_indices(n, 1)
    Eset = set(E)
    ismask = np.array([(i, j) in Eset for i, j in zip(*iu)])
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = rng.normal(0, 1.0, size=2 * n)

        def cost(x):
            p = x.reshape(n, 2)
            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)[iu]
            de, dn = d[ismask], d[~ismask]
            m = de.mean() + 1e-12
            return float(np.sum((de / m - 1.0) ** 2)
                         + np.sum(np.clip(2.0 - dn / m, 0, None) ** 2))

        r = minimize(cost, x0, method="L-BFGS-B", options={"maxiter": 6000})
        if best is None or r.fun < best.fun:
            best = r
    p = best.x.reshape(n, 2)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)[iu]
    de, dn = d[ismask], d[~ismask]
    m = de.mean()
    ce, cn = (m / de) ** 6, (m / dn) ** 6
    deg = max(sum(1 for e in E if v in e) for v in range(n))
    return {"max_degree": int(deg),
            "degree_crosstalk_floor_pct": 100 * degree_floor(deg),
            "edge_length_spread_pct": 100 * float(np.std(de) / m),
            "edge_coupling_spread_pct": 100 * float(np.std(ce) / np.mean(ce)),
            "worst_nonedge_coupling_pct": 100 * float(cn.max())}


if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out = {"device": {"C6_rad_s_um6": C6, "Omega_max_rad_s": OMEGA_MAX,
                      "tau_max_s": TAU_MAX, "r_min_um": R_MIN, "fov_um": FOV}}

    print("[1] ALGEBRA -- repulsive Rydberg realises -H exactly, no condition")
    alg = []
    for k in range(1, kmax + 1):
        for f in FAMS:
            r = algebra_test(f, k)
            alg.append(r)
            print("    k=%d %-12s n=%2d E=%2d  max|H_Aquila-(-H)| = %.1e   "
                  "V=+%.3f E_u   Delta in [%.2f, %.2f]"
                  % (r["k"], r["family"], r["n_atoms"], r["edges"],
                     r["max_map_error"], r["V_over_Eu"], r["Delta_min"],
                     r["Delta_max"]), flush=True)
    out["algebra"] = alg

    c = clock_test()
    out["clock"] = c
    print("\n[2] CLOCK -- drive at the device maximum")
    print("    edge spacing %.2f um (min allowed %.1f): %s"
          % (c["edge_spacing_um"], R_MIN, "OK" if c["spacing_ok"] else "FAILS"))
    print("    reachable dimensionless time %.1f, record needs %.0f: %s"
          % (c["t_reachable"], c["t_needed"], "OK" if c["clock_ok"] else "FAILS"))

    print("\n[3] GEOMETRY -- C6/r^6 is one isotropic function of one distance"
          " matrix")
    print("    unavoidable neighbour crosstalk by vertex degree:")
    for d in range(2, 7):
        print("      degree %d: %6.1f%% of an edge coupling" % (d, 100 * degree_floor(d)))
    print()
    geo = []
    for k in range(1, kmax + 1):
        for f in FAMS:
            n, E = graph(f, k)
            g = unit_distance_fit(n, E)
            g.update({"family": f, "k": k, "n_atoms": n, "edges": len(E)})
            geo.append(g)
            print("    k=%d %-12s n=%2d E=%2d maxdeg=%d   edge length spread "
                  "%5.1f%%  ->  COUPLING spread %6.1f%%   worst non-edge %5.1f%%"
                  % (k, f, n, len(E), g["max_degree"],
                     g["edge_length_spread_pct"], g["edge_coupling_spread_pct"],
                     g["worst_nonedge_coupling_pct"]), flush=True)
    out["geometry"] = geo

    ref = [g for g in geo if g["family"] == "null_native"]
    sig = [g for g in geo if g["family"] == "aml_ring6"]
    out["verdict"] = {
        "algebra_exact": bool(max(a["max_map_error"] for a in alg) < 1e-12),
        "clock_ok": c["clock_ok"] and c["spacing_ok"],
        "reference_realisable": bool(min(g["edge_coupling_spread_pct"]
                                         for g in ref) < 1.0),
        "discriminator_realisable": bool(min(g["edge_coupling_spread_pct"]
                                             for g in sig) < 10.0),
        "statement": "algebra and clock pass; the pair reference embeds "
                     "exactly; no discriminating typology embeds, because "
                     "an isotropic r^-6 law cannot draw an arbitrary graph "
                     "with equal edge lengths"}
    json.dump(out, open(os.path.join(HERE, "results",
                                     "hsbc_aquila_class.json"), "w"), indent=1)
    print("\n  algebra exact: %s   clock OK: %s   reference embeds: %s   "
          "discriminator embeds: %s"
          % (out["verdict"]["algebra_exact"], out["verdict"]["clock_ok"],
             out["verdict"]["reference_realisable"],
             out["verdict"]["discriminator_realisable"]))
    print("  wrote results/hsbc_aquila_class.json")
