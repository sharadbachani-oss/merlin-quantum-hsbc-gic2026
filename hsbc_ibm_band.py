# -*- coding: utf-8 -*-
"""
hsbc_ibm_band.py — IBM leg: gray-zone transactions through the project's
own beyond-classical feature map. FREE (open-plan trio), so this is the
band-lift experiment that runs BEFORE any Aquila spend.

Encoding: each transaction's top-16 features -> per-rung preparation
angle (8 rungs x prep) + per-rung local Z-tilt (8 rungs x drive) on the
64-rung 2D lattice; evolution to k in the hard window; features = the
collective spectrum X(q) at 4 wavevectors x 2 depths + 8 coarse rung-
sector parities = 16 quantum features per transaction.

Instrument discipline (internal):
  - ONE parameterized template per depth, transpiled ONCE with a fixed
    seed; per-transaction circuits are angle rebindings of the same
    physical circuit (the equal-structure law).
  - Interleaved NULL-encoding reference circuits every 16 transactions;
    features are differentials vs the running null.
  - Replica pair on one fixed transaction -> measured sigma.

Stages:
  model  — 12-rung statevector mechanics gate (variability, null
           separation, replica agreement) — mechanics only
  fly    — stratified band sample (default 96 txns + 8 nulls + 2
           replicas) x k in {8, 10}, 8,192 shots, one job
  grade  — assemble the quantum feature table -> hsbc_ibm_features.csv
           (the re-ranker consumes it; pre-registered analysis in
           g4_grade.py)
"""
import json, math, os, sys, time
import numpy as np

WORK = r"C:\quantum ai 2026\hsbc"
OPEN_CRN = __import__("os").environ.get("IBM_QUANTUM_CRN", "")
OPEN_BACKENDS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]
MU = 3.0 / (3.0 - math.sqrt(5.0))
DT, GB = 0.30, 1.0
KS = (8, 10)
N_TXN, SHOTS = 96, 8192
STATE = os.path.join(WORK, "ibm_band_state.json")
sys.path.insert(0, r"C:\fable\python")


def load_band(n_feat=16):
    """reproduce the frozen band and produce per-txn feature vectors in
    [0,1] (top model-important features, train-quantile normalized)."""
    import pandas as pd, xgboost as xgb
    gz = json.load(open(os.path.join(WORK, "g1_baseline_merged.json")))
    lo, hi = gz["gray_zone"]["score_lo"], gz["gray_zone"]["score_hi"]
    df = pd.read_csv(r"C:/hsbc/train_transaction.csv")
    idf = pd.read_csv(r"C:/hsbc/train_identity.csv")
    df = df.merge(idf, on="TransactionID", how="left")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    y = df.pop("isFraud").values
    df = df.drop(columns=["TransactionID"])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category").cat.codes
    X = df.values.astype(np.float32)
    cols = [str(c) for c in df.columns]
    del df, idf
    import gc; gc.collect()
    cut = int(0.8 * len(y))
    spw = (1 - y[:cut].mean()) / y[:cut].mean()
    clf = xgb.XGBClassifier(n_estimators=380, max_depth=9, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.6,
                            tree_method="hist", scale_pos_weight=spw,
                            n_jobs=18, random_state=21)
    clf.fit(X[:cut], y[:cut], verbose=False)
    p = clf.predict_proba(X[cut:])[:, 1]
    band = np.where((p >= lo) & (p <= hi))[0]
    fi = np.argsort(clf.feature_importances_)[::-1][:n_feat]
    qlo = np.nanpercentile(X[:cut][:, fi], 5, axis=0)
    qhi = np.nanpercentile(X[:cut][:, fi], 95, axis=0)
    F = np.clip((np.nan_to_num(X[cut:][band][:, fi]) - qlo) /
                np.maximum(qhi - qlo, 1e-9), 0, 1)
    return band, y[cut:][band], p[band], F, [cols[i] for i in fi]


def build_template(k, n_rungs, rungs, bonds, nq):
    """parameterized: 8 prep-sector angles + 8 tilt-sector angles."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    pa = [Parameter(f"a{i}") for i in range(8)]
    pt = [Parameter(f"t{i}") for i in range(8)]
    qc = QuantumCircuit(nq, nq)
    sec = max(1, n_rungs // 8)
    for r, (a, b) in enumerate(rungs):
        qc.ry(pa[min(r // sec, 7)], a)          # data-dependent prep
    for _ in range(k):
        for (a, b) in rungs:
            qc.rzz(-MU * DT, a, b)
        for (a, b) in rungs:
            qc.rx(-2 * DT, a); qc.rx(-2 * DT, b)
        for layer in bonds:
            for (a, b) in layer:
                qc.rxx(2 * GB * DT, a, b)
        for r, (a, b) in enumerate(rungs):      # data-dependent tilt
            qc.rz(pt[min(r // sec, 7)] * DT, a)
    qc.measure(range(nq), range(nq))
    return qc, pa, pt


def txn_angles(fv):
    """16 features -> 8 prep angles + 8 tilt angles."""
    prep = (0.15 + 0.7 * fv[:8]) * math.pi        # RY in (0.15pi, 0.85pi)
    tilt = (fv[8:16] - 0.5) * 2.0                 # RZ tilt in [-1, 1]
    return prep, tilt


def features_from_counts(counts, rungs):
    import fable_a1_flight as A
    m1 = np.zeros(len(rungs)); tot = 0
    for bits, c in counts.items():
        sb = bits.replace(" ", "")[::-1]
        par = np.array([1 - 2 * (int(sb[a]) ^ int(sb[b]))
                        for a, b in rungs], float)
        m1 += c * par; tot += c
    m1 /= tot
    _, _, Xq = A.grade_counts(counts, rungs)
    sec = max(1, len(rungs) // 8)
    coarse = [float(m1[i * sec:(i + 1) * sec].mean()) for i in range(8)]
    return [Xq[q] for q in (0.5, 1.0, 2.0, 4.0)] + coarse


def stage_model():
    """12-rung statevector mechanics gate."""
    from qiskit.quantum_info import Statevector
    rungs = [(2 * i, 2 * i + 1) for i in range(12)]
    bonds = [[(2 * i, 2 * i + 2) for i in range(0, 11, 2)],
             [(2 * i + 1, 2 * i + 3) for i in range(1, 10, 2)]]
    nq = 24
    qc, pa, pt = build_template(4, 12, rungs, bonds, nq)
    rng = np.random.default_rng(7)
    feats = []
    for trial in range(6):
        fv = rng.uniform(0, 1, 16)
        prep, tilt = txn_angles(fv)
        bound = qc.assign_parameters(
            {**{pa[i]: prep[i] for i in range(8)},
             **{pt[i]: tilt[i] for i in range(8)}})
        sv = Statevector(bound.remove_final_measurements(inplace=False))
        P = np.abs(sv.data) ** 2
        draws = rng.choice(len(P), 4096, p=P / P.sum())
        cts = {}
        for d in draws:
            b = format(d, f"0{nq}b")
            cts[b] = cts.get(b, 0) + 1
        feats.append(features_from_counts(cts, rungs))
    M = np.array(feats)
    var = float(M.std(axis=0).mean())
    print(f"mechanics: feature variability across 6 random txns = {var:.4f} "
          f"({'PASS' if var > 1e-3 else 'FAIL — encoding dead'})")
    return 0 if var > 1e-3 else 1


def stage_fly():
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    import fable_a1_flight as A
    svc = QiskitRuntimeService(instance=OPEN_CRN)
    pend = {}
    for b in OPEN_BACKENDS:
        try:
            pend[b] = svc.backend(b).status().pending_jobs
        except Exception:
            pend[b] = 10 ** 6
    name = min(pend, key=pend.get)
    backend = svc.backend(name)
    print(f"  queue {pend} -> {name}", flush=True)
    rungs, bonds, nq = A.colour_edges(backend)
    n_r = len(rungs)
    print(f"  {n_r} rungs on {name}", flush=True)
    band, yb, pb, F, fnames = load_band()
    rng = np.random.default_rng(21)
    yi = np.where(yb == 1)[0]; ni = np.where(yb == 0)[0]
    n_f = int(round(N_TXN * yb.mean()))          # stratified: preserve ratio
    pick = np.concatenate([rng.choice(yi, n_f, False),
                           rng.choice(ni, N_TXN - n_f, False)])
    rng.shuffle(pick)
    tag = time.strftime("%Y%m%d_%H%M%S")
    circs, names = [], []
    tts = {}
    for k in KS:
        t, pa, pt = build_template(k, n_r, rungs, bonds, nq)
        tts[k] = (transpile(t, backend, optimization_level=1,
                            seed_transpiler=21), pa, pt)
    n2q = tts[KS[0]][0].count_ops()
    print(f"  k={KS[0]} template 2q gates: "
          f"{n2q.get('cz', n2q.get('ecr', n2q.get('cx', 0)))}", flush=True)
    for j, bi in enumerate(pick):
        prep, tilt = txn_angles(F[bi])
        for k in KS:
            tt, pa, pt = tts[k]
            bind = {**{pa[i]: float(prep[i]) for i in range(8)},
                    **{pt[i]: float(tilt[i]) for i in range(8)}}
            bind = {p_: v for p_, v in bind.items() if p_ in tt.parameters}
            circs.append(tt.assign_parameters(bind))
            names.append(f"txn{j}_k{k}")
        if j % 16 == 0:                          # interleaved null reference
            for k in KS:
                tt, pa, pt = tts[k]
                bind = {**{pa[i]: 0.5 * math.pi for i in range(8)},
                        **{pt[i]: 0.0 for i in range(8)}}
                bind = {p_: v for p_, v in bind.items() if p_ in tt.parameters}
                circs.append(tt.assign_parameters(bind))
                names.append(f"null{j}_k{k}")
    for rep in (1, 2):                           # replica pair, txn 0
        prep, tilt = txn_angles(F[pick[0]])
        tt, pa, pt = tts[KS[0]]
        bind = {**{pa[i]: float(prep[i]) for i in range(8)},
                **{pt[i]: float(tilt[i]) for i in range(8)}}
        bind = {p_: v for p_, v in bind.items() if p_ in tt.parameters}
        circs.append(tt.assign_parameters(bind))
        names.append(f"rep{rep}_k{KS[0]}")
    s = SamplerV2(mode=backend)
    s.options.dynamical_decoupling.enable = True
    s.options.dynamical_decoupling.sequence_type = "XpXm"
    job = s.run(circs, shots=SHOTS)
    json.dump(dict(tag=tag, backend=name, job=job.job_id(), names=names,
                   rungs=[list(r) for r in rungs],
                   pick=[int(v) for v in pick],
                   y=[int(yb[i]) for i in pick],
                   p_classical=[float(pb[i]) for i in pick],
                   feat_names=fnames),
              open(STATE, "w"), indent=1)
    print(f"IBM BAND submitted: {job.job_id()} ({len(circs)} x {SHOTS} "
          f"on {name})", flush=True)
    return 0


def stage_grade():
    from qiskit_ibm_runtime import QiskitRuntimeService
    st = json.load(open(STATE))
    svc = QiskitRuntimeService(instance=OPEN_CRN)
    res = svc.job(st["job"]).result()
    rungs = [tuple(r) for r in st["rungs"]]
    rows, nulls = {}, {}
    for i, nm in enumerate(st["names"]):
        f = features_from_counts(res[i].data.c.get_counts(), rungs)
        (nulls if nm.startswith("null") else rows)[nm] = f
    nk = {k: np.mean([v for n_, v in nulls.items() if n_.endswith(f"_k{k}")],
                     axis=0) for k in KS}
    import csv
    out = os.path.join(WORK, "hsbc_ibm_features.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        hdr = ["txn", "y", "p_classical"] + \
              [f"q{k}_{i}" for k in KS for i in range(12)]
        w.writerow(hdr)
        for j in range(len(st["pick"])):
            feats = []
            for k in KS:
                feats += (np.array(rows[f"txn{j}_k{k}"]) - nk[k]).tolist()
            w.writerow([j, st["y"][j], st["p_classical"][j]] + feats)
    r1 = np.array(rows[f"rep1_k{KS[0]}"]); r2 = np.array(rows[f"rep2_k{KS[0]}"])
    print(f"replica agreement: mean |d| = {np.abs(r1 - r2).mean():.4f}")
    print(f"-> {out}  ({len(st['pick'])} transactions x {2 * 12} quantum "
          f"features, null-differenced)")
    return 0


if __name__ == "__main__":
    sys.exit({"model": stage_model, "fly": stage_fly,
              "grade": stage_grade}[sys.argv[1] if len(sys.argv) > 1
                                    else "model"]())
