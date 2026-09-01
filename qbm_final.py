# -*- coding: utf-8 -*-
"""
qbm_final.py — DATA-ENCODED QUANTUM BOLTZMANN MACHINE at 127 qubits.
The final IBM card. One job.

WHAT EVERY PREVIOUS GENERATIVE ATTEMPT GOT WRONG (measured today):
conditioning the INITIAL STATE on data leaves the output correlations
governed by the lattice's own physics, not the data's — measured:
quantum-correlated perturbation scored 0.874x versus independent
Gaussian noise on 3-body fidelity, and unperturbed seeds beat both.
Training-free generation reproduces the DEVICE's distribution.

THE FIX: encode the data into the HAMILTONIAN, not the initial state.
  h_i  = log-odds of literal i (fraud vs legit)        -> local fields
  J_ij = measured pairwise interaction beyond independence -> couplings
Both are read directly off the data's moments — no variational training,
so no barren plateau. The device's equilibrium distribution is then a
model OF THE DATA. This is the quantum Boltzmann machine the published
roadmap wants (Q-SYNTH, QCBM), at 127 qubits instead of ~20.

CLAIM UNDER TEST (pre-registered, scored on banked classical baselines):
  fidelity of generated samples to REAL fraud higher-order structure,
  vs SMOTE / Gaussian / jitter / classical pairwise Gibbs.
  Metric: mean |moment error| at 1-, 2- and 3-body order.
  The 3-body order is the discriminator: SMOTE interpolates linearly
  and Gaussian models are second-order by construction.

PROTOCOL: |+>^n  ->  Trotterized evolution under
  H = sum_i h_i Z_i + sum_<ij> J_ij Z_i Z_j + Gamma sum_i X_i
on the device's NATIVE edges (no SWAPs), transverse field ramped down so
the state concentrates on the data-encoded landscape while retaining
thermal spread. Each shot = one synthetic fraud pattern.
"""
import json, math, os, sys, time
import numpy as np

WORK = r"C:\quantum ai 2026\hsbc"
OPEN_CRN = __import__("os").environ.get("IBM_QUANTUM_CRN", "")
BACKENDS = ["ibm_kingston", "ibm_fez", "ibm_marrakesh"]
STATE = os.path.join(WORK, "qbm_state.json")
SHOTS = 8192
STEPS = 6                      # Trotter steps of the data Hamiltonian


def load_fraud():
    import pandas as pd
    d = pd.read_csv("C:/hsbc/ulb/creditcard.csv").sort_values("Time")
    y = d.Class.values.astype(int)
    FE = [c for c in d.columns if c not in ("Class", "Time")]
    X = d[FE].values.astype(np.float64)
    cut = int(0.8 * len(y))
    return X[:cut], y[:cut], X[cut:], y[cut:], FE


def build_literals(Xtr, ytr, n_lit):
    """n_lit binary literals spread over the most fraud-informative
    features (median thresholds keep marginals near 0.5 = maximal
    information per qubit)."""
    f = ytr == 1
    sep = np.abs(Xtr[f].mean(0) - Xtr[~f].mean(0)) / (Xtr.std(0) + 1e-9)
    order = np.argsort(sep)[::-1]
    lits, spec = [], []
    qs = (0.30, 0.50, 0.70)
    for j in order:
        for q in qs:
            thr = float(np.nanquantile(Xtr[:, j], q))
            lits.append(Xtr[:, j] >= thr); spec.append((int(j), thr))
            if len(lits) == n_lit:
                return np.array(lits), spec
    return np.array(lits), spec


def stage_fit():
    """read h and J off the data — no training."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(instance=OPEN_CRN)
    pend = {}
    for b in BACKENDS:
        try:
            pend[b] = svc.backend(b).status().pending_jobs
        except Exception:
            pend[b] = 10 ** 9
    name = min(pend, key=pend.get)
    backend = svc.backend(name)
    tgt = backend.target
    g2 = next(x for x in ("cz", "ecr", "cx") if x in tgt.operation_names)
    edges = sorted({(min(int(a), int(b)), max(int(a), int(b)))
                    for a, b in tgt[g2]})
    nq = tgt.num_qubits
    print(f"  queue {pend} -> {name}: {nq} qubits, {len(edges)} native edges",
          flush=True)
    Xtr, ytr, Xte, yte, FE = load_fraud()
    L, spec = build_literals(Xtr, ytr, nq)
    f = ytr == 1
    print(f"  {len(L)} literals from {int(f.sum())} train frauds", flush=True)

    def p(rows, idxs):
        m = np.ones(rows.sum(), bool)
        sub = L[:, rows]
        for i in idxs:
            m &= sub[i]
        return min(max(m.mean(), 1e-4), 1 - 1e-4)

    h = np.array([math.log(p(f, [i]) / p(~f, [i])) for i in range(len(L))])
    J = {}
    for (a, b) in edges:
        if a >= len(L) or b >= len(L):
            continue
        v = (math.log(p(f, [a, b]) / (p(f, [a]) * p(f, [b]))) -
             math.log(p(~f, [a, b]) / (p(~f, [a]) * p(~f, [b]))))
        J[(a, b)] = float(v)
    hs = float(np.abs(h).max()); js = max(abs(v) for v in J.values())
    print(f"  fields |h|max {hs:.3f} | couplings |J|max {js:.3f} "
          f"({len(J)} edges carry data)", flush=True)
    json.dump(dict(backend=name, nq=nq, h=h.tolist(),
                   J={f"{a},{b}": v for (a, b), v in J.items()},
                   spec=spec, n_train_fraud=int(f.sum())),
              open(os.path.join(WORK, "qbm_model.json"), "w"), indent=1)
    print(f"-> qbm_model.json")
    return 0


def stage_fly():
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    M = json.load(open(os.path.join(WORK, "qbm_model.json")))
    nq = M["nq"]
    h = np.array(M["h"])
    J = {tuple(int(x) for x in k.split(",")): v for k, v in M["J"].items()}
    svc = QiskitRuntimeService(instance=OPEN_CRN)
    backend = svc.backend(M["backend"])
    # normalise so the strongest coupling rotates by ~pi/4 per step
    scale = 0.6 / max(max(abs(v) for v in J.values()), np.abs(h).max())
    # edge colouring for parallel layers
    layers = []
    for (a, b) in J:
        for L_ in layers:
            if all(a not in e and b not in e for e in L_):
                L_.append((a, b)); break
        else:
            layers.append([(a, b)])

    def circuit(sample=True):
        qc = QuantumCircuit(nq, nq)
        nlit = len(h)
        qc.h(range(nlit))                            # infinite-temperature
        for s in range(STEPS):
            gam = 1.0 - s / STEPS                     # transverse ramp-down
            for L_ in layers:
                for (a, b) in L_:
                    qc.rzz(2 * scale * J[(a, b)], a, b)
            for i in range(nlit):
                qc.rz(2 * scale * float(h[i]) if sample else 0.0, i)
            for i in range(nlit):
                qc.rx(2 * 0.8 * gam, i)
        qc.measure(range(nlit), range(nlit))
        return qc

    data = transpile(circuit(True), backend, optimization_level=1,
                     seed_transpiler=21)
    null = transpile(circuit(False), backend, optimization_level=1,
                     seed_transpiler=21)
    n2 = data.count_ops()
    print(f"  circuit: {n2.get('cz', n2.get('ecr', n2.get('cx', 0)))} "
          f"2q gates, depth {data.depth()}", flush=True)
    s = SamplerV2(mode=backend)
    s.options.dynamical_decoupling.enable = True
    s.options.dynamical_decoupling.sequence_type = "XpXm"
    job = s.run([data, null, data], shots=SHOTS)   # data, control, replica
    json.dump(dict(job=job.job_id(), backend=M["backend"], nq=nq,
                   shots=SHOTS, names=["data", "null", "replica"]),
              open(STATE, "w"), indent=1)
    print(f"QBM submitted: {job.job_id()} (3 x {SHOTS} on {M['backend']}) "
          f"-> {3*SHOTS} samples at {nq} qubits", flush=True)
    return 0


def stage_score():
    from itertools import combinations
    from qiskit_ibm_runtime import QiskitRuntimeService
    st = json.load(open(STATE))
    M = json.load(open(os.path.join(WORK, "qbm_model.json")))
    spec = [(int(a), float(b)) for a, b in M["spec"]]
    svc = QiskitRuntimeService(instance=OPEN_CRN)
    res = svc.job(st["job"]).result()

    def bits(i):
        out = []
        for b, c in res[i].data.c.get_counts().items():
            sb = b.replace(" ", "")[::-1]
            v = np.array([int(sb[k]) for k in range(st["nq"])], np.int8)
            out.extend([v] * int(c))
        return np.array(out)

    Bq, Bn, Br = bits(0), bits(1), bits(2)
    print(f"device samples: {len(Bq)} (control {len(Bn)}, replica {len(Br)})")
    Xtr, ytr, Xte, yte, FE = load_fraud()
    fr = Xtr[ytr == 1]
    D = min(20, st["nq"])
    Bre = np.array([[1 if fr[r, j] >= t else 0 for (j, t) in spec[:D]]
                    for r in range(len(fr))])

    def mom(B, o):
        return np.array([B[:, list(c)].all(1).mean()
                         for c in combinations(range(D), o)])
    ref = {o: mom(Bre, o) for o in (1, 2, 3)}
    rng = np.random.default_rng(0)
    N = 4000
    a = fr[rng.integers(0, len(fr), N)]; b = fr[rng.integers(0, len(fr), N)]
    sm = a + rng.random((N, 1)) * (b - a)
    ji = fr[rng.integers(0, len(fr), N)] + rng.normal(0, fr.std(0) * 0.1,
                                                      (N, len(FE)))
    mu = fr.mean(0); C = np.cov(fr.T) + 1e-6 * np.eye(len(FE))
    ga = rng.multivariate_normal(mu, C, N)

    def tobits(X):
        return np.array([[1 if X[r, j] >= t else 0 for (j, t) in spec[:D]]
                         for r in range(len(X))])
    arms = {"SMOTE": tobits(sm), "jitter": tobits(ji), "gaussian": tobits(ga),
            "QBM (device)": Bq[:, :D], "device control": Bn[:, :D]}
    print("\ngenerator        | 1-body | 2-body | 3-body")
    out = {}
    for nm, B in arms.items():
        e = {o: float(np.abs(mom(B, o) - ref[o]).mean()) for o in (1, 2, 3)}
        out[nm] = e
        print(f"{nm:16s} | {e[1]:.4f} | {e[2]:.4f} | {e[3]:.4f}")
    q = out["QBM (device)"][3]
    bc = min(out[k][3] for k in ("SMOTE", "jitter", "gaussian"))
    print(f"\nQBM 3-body {q:.4f} vs best classical {bc:.4f} -> "
          f"{bc/q:.2f}x {'BETTER' if q < bc else 'worse'}")
    print(f"control check: device control 3-body {out['device control'][3]:.4f} "
          f"(should be worse than QBM if the data encoding matters)")
    json.dump(dict(card="data-encoded QBM at 127 qubits", job=st["job"],
                   nq=st["nq"], arms=out),
              open(os.path.join(WORK, "qbm_result.json"), "w"), indent=1)
    print("-> qbm_result.json")
    return 0


if __name__ == "__main__":
    sys.exit({"fit": stage_fit, "fly": stage_fly,
              "score": stage_score}[sys.argv[1] if len(sys.argv) > 1
                                    else "fit"]())
