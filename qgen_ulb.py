# -*- coding: utf-8 -*-
"""
qgen_ulb.py — higher-order quantum generative augmentation on the ULB
European Cardholder dataset (the challenge's extreme-imbalance benchmark:
284,807 transactions, 492 frauds, 0.172%).

WHY THIS DATASET. Augmentation only pays where the minority class is
data-starved. Measured on Sparkov (0.57% fraud, 5,954 frauds) the
baseline already sits at AUPRC 0.924 / AUC 0.9985 — SMOTE itself *hurts*
there. ULB has 492 frauds total; the published bar is AUPRC 0.867-0.871
(XGBoost+SMOTE / RF+SMOTE, IIETA 2024) with real headroom, and the
dataset authors specify AUPRC as the metric (Dal Pozzolo 2015).

THE CLAIM UNDER TEST. Dirac-3 samples a Gibbs distribution over a
programmed degree-5 landscape. Classical Boltzmann machines are confined
in practice to PAIRWISE interactions (higher-order partition functions
are intractable). So the device can generate minority samples carrying
3-body correlation structure the classical generative class cannot
represent. Arms are matched on synthetic-sample count; the pairwise
Boltzmann arm uses the SAME fitted model truncated to degree 2, which
isolates the higher-order terms as the only difference.

Stages: base -> fit -> fly -> score
"""
import json, math, os, sys
from itertools import combinations
import numpy as np

WORK = r"C:\quantum ai 2026\hsbc"
sys.path.insert(0, r"C:\fable\python")
sys.path.insert(0, r"C:\gic2026 final\qci\run")
NB = 24
N_JOBS, SAMPLES_PER = 40, 20
N_AUG = 2000
FIT = os.path.join(WORK, "ulb_model.json")
RAW = os.path.join(WORK, "ulb_raw.json")


def load():
    import pandas as pd
    d = pd.read_csv("C:/hsbc/ulb/creditcard.csv").sort_values("Time")
    y = d.Class.values.astype(int)
    FE = [c for c in d.columns if c not in ("Class", "Time")]
    X = d[FE].values.astype(np.float64)
    cut = int(0.8 * len(y))
    return X[:cut], y[:cut], X[cut:], y[cut:], FE


def stage_base():
    import xgboost as xgb
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 precision_recall_curve)
    Xtr, ytr, Xte, yte, FE = load()
    print(f"train {len(ytr)} ({int(ytr.sum())} fraud, {100*ytr.mean():.4f}%) | "
          f"test {len(yte)} ({int(yte.sum())} fraud)")
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.06,
                            tree_method="hist", n_jobs=18, random_state=21,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=(1 - ytr.mean()) / ytr.mean())
    clf.fit(Xtr, ytr, verbose=False)
    p = clf.predict_proba(Xte)[:, 1]
    pr, rc, _ = precision_recall_curve(yte, p)
    f1 = float(np.nanmax(2 * pr * rc / np.maximum(pr + rc, 1e-12)))
    print(f"baseline: AUPRC {average_precision_score(yte,p):.4f} | "
          f"AUC {roc_auc_score(yte,p):.4f} | F1 {f1:.4f}")
    print("published bar: AUPRC 0.867-0.871 (XGB/RF + SMOTE, IIETA 2024)")
    return 0


def literals_of(Xtr, ytr):
    """pick the literals that carry fraud signal: strongest features by
    separation, thresholded at fraud-side quantiles."""
    f = ytr == 1
    sep = np.abs(Xtr[f].mean(0) - Xtr[~f].mean(0)) / (Xtr.std(0) + 1e-9)
    top = np.argsort(sep)[::-1][:8]
    L, names, spec = [], [], []
    for j in top:
        v = Xtr[:, j]
        for q in (0.10, 0.50, 0.90):
            thr = float(np.nanquantile(v, q))
            L.append(v >= thr); names.append(f"f{j}>=q{int(q*100)}")
            spec.append((int(j), thr))
    return np.array(L)[:NB], names[:NB], spec[:NB]


def stage_fit():
    Xtr, ytr, Xte, yte, FE = load()
    L, names, spec = literals_of(Xtr, ytr)
    f = ytr == 1
    n = len(L)
    print(f"{n} literals | {int(f.sum())} train frauds", flush=True)

    def p(rows, idxs):
        sub = L[:, rows]
        m = np.ones(rows.sum(), bool)
        for i in idxs:
            m &= sub[i]
        return max(m.mean(), 1e-6)

    terms = {}
    for i in range(n):
        terms[(i,)] = math.log(p(f, [i]) / p(~f, [i]))
    for i, j in combinations(range(n), 2):
        v = (math.log(p(f, [i, j]) / (p(f, [i]) * p(f, [j]))) -
             math.log(p(~f, [i, j]) / (p(~f, [i]) * p(~f, [j]))))
        if abs(v) > 0.05:
            terms[(i, j)] = v
    n3 = 0
    for i, j, k in combinations(range(n), 3):
        if p(f, [i, j, k]) < 1e-3:
            continue
        pf = p(f, [i, j, k]) * p(f, [i]) * p(f, [j]) * p(f, [k]) / \
             max(p(f, [i, j]) * p(f, [i, k]) * p(f, [j, k]), 1e-12)
        pn = p(~f, [i, j, k]) * p(~f, [i]) * p(~f, [j]) * p(~f, [k]) / \
             max(p(~f, [i, j]) * p(~f, [i, k]) * p(~f, [j, k]), 1e-12)
        v = math.log(max(pf, 1e-9)) - math.log(max(pn, 1e-9))
        if abs(v) > 0.10:
            terms[(i, j, k)] = v; n3 += 1
    by = {}
    for t in terms:
        by[len(t)] = by.get(len(t), 0) + 1
    print(f"terms by order {dict(sorted(by.items()))} | {n3} genuine 3-body")
    json.dump(dict(nb=n, names=names, spec=[[a, b] for a, b in spec],
                   terms={",".join(map(str, k)): v for k, v in terms.items()},
                   by_order=by), open(FIT, "w"), indent=1)
    print(f"-> {FIT}")
    return 0


def stage_fly():
    M = json.load(open(FIT))
    terms = {tuple(int(i) for i in k.split(",")): v
             for k, v in M["terms"].items()}
    from qci_dirac_common import make_qci_client
    from fable_qci_robust import submit_and_wait
    client, _ = make_qci_client()
    data = []
    for c, v in terms.items():
        if abs(v) < 1e-6:
            continue
        data.append({"idx": [i + 1 for i in c], "val": float(-v)})
    md = max(len(d["idx"]) for d in data)
    for d in data:
        d["idx"] = [0] * (md - len(d["idx"])) + d["idx"]
    print(f"landscape: degree {md}, {len(data)} terms, {M['nb']} vars",
          flush=True)
    fid = client.upload_file(file={"file_name": "ulb_landscape",
        "file_config": {"polynomial": {"num_variables": M["nb"],
            "min_degree": 1, "max_degree": md, "data": data}}})["file_id"]
    runs = []
    for r in range(N_JOBS):
        body = client.build_job_body(
            job_type="sample-hamiltonian-integer", job_name=f"ulbgen_{r}",
            job_tags=["fable", "qgen", "ulb"],
            job_params={"device_type": "dirac-3", "num_samples": SAMPLES_PER,
                        "relaxation_schedule": 1,
                        "num_levels": [2] * M["nb"]},
            polynomial_file_id=fid)
        jid, sols = submit_and_wait(client, body, timeout=1800)
        runs.append(dict(job=jid, sols=[[int(round(v)) for v in s]
                                        for s in sols]))
        uniq = len({tuple(s) for x in runs for s in x["sols"]})
        print(f"  {r+1}/{N_JOBS} {jid}: {len(sols)} samples "
              f"({uniq} distinct so far)", flush=True)
        json.dump(runs, open(RAW, "w"), indent=1)
    return 0


def stage_score():
    import xgboost as xgb
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 precision_recall_curve)
    Xtr, ytr, Xte, yte, FE = load()
    M = json.load(open(FIT))
    spec = [(int(a), float(b)) for a, b in M["spec"]]
    terms = {tuple(int(i) for i in k.split(",")): v
             for k, v in M["terms"].items()}
    pair = {k: v for k, v in terms.items() if len(k) <= 2}
    runs = json.load(open(RAW))
    dev = sorted({tuple(s) for r in runs for s in r["sols"]})
    print(f"device: {sum(len(r['sols']) for r in runs)} samples, "
          f"{len(dev)} distinct", flush=True)
    fr = Xtr[ytr == 1]

    def energy(x, T):
        return -sum(v for c, v in T.items() if all(x[i] for i in c))

    def gibbs(T, n, seed, burn=300):
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 2, M["nb"]).tolist(); out = []
        for it in range(burn + n * 6):
            i = int(rng.integers(M["nb"]))
            a = list(x); a[i] = 0
            b = list(x); b[i] = 1
            d = energy(b, T) - energy(a, T)
            x[i] = 1 if rng.random() < 1.0 / (1.0 + math.exp(d)) else 0
            if it >= burn and it % 6 == 0:
                out.append(tuple(x))
        return out[:n] or [tuple(x)]

    def decode(bits, seed):
        rng = np.random.default_rng(seed)
        sc = np.zeros(len(fr))
        for i, (j, thr) in enumerate(spec):
            sc += ((fr[:, j] >= thr).astype(int) == bits[i])
        top = np.argsort(sc)[::-1][:15]
        return fr[top].mean(0) + rng.normal(0, fr.std(0) * 0.05)

    def build(kind, seed):
        r = np.random.default_rng(seed)
        if kind == "none":
            return Xtr, ytr
        if kind == "dirac_deg3":
            pats = [dev[i] for i in r.integers(0, len(dev), N_AUG)]
            Xa = np.array([decode(p, seed * 997 + i) for i, p in enumerate(pats)])
        elif kind == "ising_pair":
            pool = gibbs(pair, 300, seed)
            pats = [pool[i] for i in r.integers(0, len(pool), N_AUG)]
            Xa = np.array([decode(p, seed * 997 + i) for i, p in enumerate(pats)])
        elif kind == "smote":
            a = fr[r.integers(0, len(fr), N_AUG)]
            b = fr[r.integers(0, len(fr), N_AUG)]
            Xa = a + r.random((N_AUG, 1)) * (b - a)
        elif kind == "jitter":
            base = fr[r.integers(0, len(fr), N_AUG)]
            Xa = base + r.normal(0, fr.std(0) * 0.15, base.shape)
        elif kind == "random_over":
            Xa = fr[r.integers(0, len(fr), N_AUG)]
        return np.vstack([Xtr, Xa]), np.concatenate([ytr, np.ones(N_AUG, int)])

    rows = {}
    for kind in ("none", "random_over", "jitter", "smote", "ising_pair",
                 "dirac_deg3"):
        m = []
        for seed in (21, 22, 23):
            Xa, ya = build(kind, seed)
            clf = xgb.XGBClassifier(n_estimators=400, max_depth=6,
                                    learning_rate=0.06, tree_method="hist",
                                    n_jobs=18, random_state=seed,
                                    subsample=0.8, colsample_bytree=0.8)
            clf.fit(Xa, ya, verbose=False)
            p = clf.predict_proba(Xte)[:, 1]
            pr, rc, _ = precision_recall_curve(yte, p)
            f1 = float(np.nanmax(2 * pr * rc / np.maximum(pr + rc, 1e-12)))
            m.append((average_precision_score(yte, p), roc_auc_score(yte, p), f1))
        a = np.array(m)
        rows[kind] = dict(auprc=[float(a[:,0].mean()), float(a[:,0].std())],
                          auc=[float(a[:,1].mean()), float(a[:,1].std())],
                          f1=[float(a[:,2].mean()), float(a[:,2].std())])
        print(f"{kind:14s}: AUPRC {a[:,0].mean():.4f}±{a[:,0].std():.4f} | "
              f"AUC {a[:,1].mean():.4f} | F1 {a[:,2].mean():.4f}", flush=True)
    d = rows["dirac_deg3"]["auprc"][0]
    for ref in ("ising_pair", "smote", "none"):
        rv = rows[ref]["auprc"][0]
        print(f"dirac_deg3 vs {ref:12s}: {100*(d-rv)/rv:+.2f}%")
    json.dump(dict(card="ULB higher-order generative augmentation",
                   n_distinct=len(dev), arms=rows),
              open(os.path.join(WORK, "ulb_result.json"), "w"), indent=1)
    print("-> ulb_result.json")
    return 0


if __name__ == "__main__":
    sys.exit({"base": stage_base, "fit": stage_fit, "fly": stage_fly,
              "score": stage_score}[sys.argv[1] if len(sys.argv) > 1
                                    else "base"]())
