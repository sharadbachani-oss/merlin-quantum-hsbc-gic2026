# -*- coding: utf-8 -*-
"""fewshot_ladder.py -- fair reproduction of the archived ULB few-shot augmentation result (laptop, 2026-09-14).

PROTOCOL, FIXED BEFORE RUNNING
  data      ULB creditcard.csv, time-sorted 80/20 split (qgen_ulb.load), test window untouched
  k         labelled frauds available to the CLASSIFIER: 5, 10, 20, all (394); 40 seeds choose which k
  generator ulb_model.json as archived (fitted on the full training-window fraud population = WARM generator;
            this is disclosed, not hidden); the only thing that varies with k is what the classifier and the
            decoder see
  decode    anchored on the k labelled frauds ONLY (fair); the archived protocol anchored on all 394 and is
            re-run as the 'dirac_deg3_warmdecode' arm to reproduce the archived 0.607 number
  arms      none | random_over | jitter | smote | ising_pair (Gibbs, degree<=2 of the same model)
            | gibbs_deg3 (classical Gibbs, full degree-3 model) | exact_deg3 (EXACT Boltzmann sampling by
            enumeration of all 2^24 states -- the true classical floor of a 24-variable landscape)
            | dirac_deg3 (the 82 distinct device patterns from 40 Dirac-3 jobs) | dirac_deg3_warmdecode
  classifier XGBoost exactly as qgen_ulb.stage_score (400 trees, depth 6, lr 0.06, hist, subsample 0.8)
  metrics   AUPRC, AUC-ROC, F1max, frauds caught in the top-75 alerts of the test window
  stats     per (k, arm): mean +- sd over 40 seeds; paired differences dirac - {smote, gibbs_deg3, exact_deg3}
            with 95% bootstrap CI (10,000 resamples) and Wilcoxon p
  decision  the augmentation gain is claimed at k only if dirac beats smote AND exact_deg3 with CI excluding 0;
            dirac ~ exact_deg3 means the gain is the MODEL's, quantum attribution 'none at 24 variables'
Writes fewshot_ladder.json after every (k, arm).
"""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qgen_ulb as Q
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

OUT = os.path.join(Q.WORK, "fewshot_ladder.json")
KS = [5, 10, 20, "all"]; SEEDS = list(range(40)); N_AUG = Q.N_AUG; BUDGET = 75
ARMS = ["none", "random_over", "jitter", "smote", "ising_pair", "gibbs_deg3", "exact_deg3",
        "dirac_deg3", "dirac_deg3_warmdecode"]

Xtr, ytr, Xte, yte, FE = Q.load()
M = json.load(open(Q.FIT)); NB = M["nb"]
spec = [(int(a), float(b)) for a, b in M["spec"]]
terms = {tuple(int(i) for i in k.split(",")): v for k, v in M["terms"].items()}
pair = {k: v for k, v in terms.items() if len(k) <= 2}
runs = json.load(open(Q.RAW)); dev = sorted({tuple(s) for r in runs for s in r["sols"]})
fr_all = Xtr[ytr == 1]; legit = Xtr[ytr == 0]
print(f"train {len(ytr)} ({len(fr_all)} frauds) | test {len(yte)} ({int(yte.sum())} frauds) | "
      f"device patterns {len(dev)}", flush=True)


def energy(x, T):
    return -sum(v for c, v in T.items() if all(x[i] for i in c))


def gibbs(T, n, seed, burn=300):
    rng = np.random.default_rng(seed); x = rng.integers(0, 2, NB).tolist(); out = []
    for it in range(burn + n * 6):
        i = int(rng.integers(NB)); a = list(x); a[i] = 0; b = list(x); b[i] = 1
        d = energy(b, T) - energy(a, T)
        x[i] = 1 if rng.random() < 1.0 / (1.0 + math.exp(d)) else 0
        if it >= burn and it % 6 == 0:
            out.append(tuple(x))
    return out[:n] or [tuple(x)]


# ---- exact Boltzmann distribution over all 2^24 states of the degree-3 model (the classical floor) ----
t0 = time.time()
Nst = 1 << NB; idx = np.arange(Nst, dtype=np.uint32)
cols = [((idx >> np.uint32(i)) & 1).astype(np.uint8) for i in range(NB)]              # 24 contiguous columns of 16.8M
E = np.zeros(Nst, dtype=np.float64)
for c, v in terms.items():
    m = cols[c[0]].copy()
    for i in c[1:]:
        m &= cols[i]
    E -= v * m
bits = np.stack(cols, axis=1)                                                            # 16.8M x 24 for sampling
del cols
P = np.exp(-(E - E.min())); P /= P.sum()
dev_idx = [int(sum(b << i for i, b in enumerate(p))) for p in dev]
print(f"exact distribution over {Nst} states in {time.time()-t0:.0f}s | "
      f"entropy {float(-(P[P>0]*np.log(P[P>0])).sum()):.2f} nats | "
      f"device-pattern mass {float(P[dev_idx].sum()):.4f} | P(top pattern) {float(P.max()):.4f}", flush=True)


def sample_exact(n, seed):
    rng = np.random.default_rng(seed); s = rng.choice(Nst, size=n, p=P)
    return [tuple(int(b) for b in bits[i]) for i in s]


def decode(bits_, seed, pool):
    rng = np.random.default_rng(seed); sc = np.zeros(len(pool))
    for i, (j, thr) in enumerate(spec):
        sc += ((pool[:, j] >= thr).astype(int) == bits_[i])
    top = np.argsort(sc)[::-1][:min(15, len(pool))]
    return pool[top].mean(0) + rng.normal(0, fr_all.std(0) * 0.05)


def build(kind, seed, frk):
    r = np.random.default_rng(seed)
    Xb = np.vstack([legit, frk]); yb = np.concatenate([np.zeros(len(legit), int), np.ones(len(frk), int)])
    if kind == "none":
        return Xb, yb
    pool_dec = fr_all if kind.endswith("warmdecode") else frk
    pats = None
    if kind.startswith("dirac"):
        pats = [dev[i] for i in r.integers(0, len(dev), N_AUG)]
    elif kind == "ising_pair":
        pool = gibbs(pair, 300, seed); pats = [pool[i] for i in r.integers(0, len(pool), N_AUG)]
    elif kind == "gibbs_deg3":
        pool = gibbs(terms, 300, seed); pats = [pool[i] for i in r.integers(0, len(pool), N_AUG)]
    elif kind == "exact_deg3":
        pats = sample_exact(N_AUG, seed)
    if pats is not None:
        Xa = np.array([decode(p, seed * 997 + i, pool_dec) for i, p in enumerate(pats)])
    elif kind == "smote":
        a = frk[r.integers(0, len(frk), N_AUG)]; b = frk[r.integers(0, len(frk), N_AUG)]
        Xa = a + r.random((N_AUG, 1)) * (b - a)
    elif kind == "jitter":
        base = frk[r.integers(0, len(frk), N_AUG)]; Xa = base + r.normal(0, fr_all.std(0) * 0.15, base.shape)
    elif kind == "random_over":
        Xa = frk[r.integers(0, len(frk), N_AUG)]
    return np.vstack([Xb, Xa]), np.concatenate([yb, np.ones(N_AUG, int)])


def fit_score(Xa, ya, seed):
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.06, tree_method="hist", n_jobs=18,
                            random_state=seed, subsample=0.8, colsample_bytree=0.8)
    clf.fit(Xa, ya, verbose=False); p = clf.predict_proba(Xte)[:, 1]
    pr, rc, _ = precision_recall_curve(yte, p)
    f1 = float(np.nanmax(2 * pr * rc / np.maximum(pr + rc, 1e-12)))
    caught = int(yte[np.argsort(p)[::-1][:BUDGET]].sum())
    return dict(auprc=float(average_precision_score(yte, p)), auc=float(roc_auc_score(yte, p)), f1=f1, caught=caught)


def paired(a, b, nboot=10000, seed=0):
    from scipy.stats import wilcoxon
    d = np.array(a, dtype=float) - np.array(b, dtype=float); rng = np.random.default_rng(seed)
    bs = [rng.choice(d, len(d)).mean() for _ in range(nboot)]
    try:
        p = float(wilcoxon(d).pvalue)
    except Exception:
        p = None
    return dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                wilcoxon_p=p)


res = dict(protocol=__doc__, n_test_frauds=int(yte.sum()), budget=BUDGET, results={})
for k in KS:
    kk = len(fr_all) if k == "all" else k; key = str(k); res["results"][key] = {}
    per_seed = {arm: [] for arm in ARMS}
    for arm in ARMS:
        t0 = time.time()
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            frk = fr_all[rng.choice(len(fr_all), kk, replace=False)]
            per_seed[arm].append(fit_score(*build(arm, seed, frk), seed))
        a = {m: [s[m] for s in per_seed[arm]] for m in ("auprc", "auc", "f1", "caught")}
        res["results"][key][arm] = {m: [float(np.mean(v)), float(np.std(v))] for m, v in a.items()}
        res["results"][key][arm]["per_seed_auprc"] = a["auprc"]
        res["results"][key][arm]["per_seed_caught"] = a["caught"]
        print(f"k={key:>4} {arm:22s} AUPRC {np.mean(a['auprc']):.4f}+-{np.std(a['auprc']):.4f}  "
              f"AUC {np.mean(a['auc']):.4f}  F1 {np.mean(a['f1']):.3f}  "
              f"caught@{BUDGET} {np.mean(a['caught']):.1f}+-{np.std(a['caught']):.1f}  ({time.time()-t0:.0f}s)",
              flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    R = res["results"][key]
    R["paired"] = {f"dirac_deg3-minus-{ref}": paired(R["dirac_deg3"]["per_seed_auprc"], R[ref]["per_seed_auprc"])
                   for ref in ("smote", "gibbs_deg3", "exact_deg3", "none")}
    R["paired_caught"] = {f"dirac_deg3-minus-{ref}": paired(R["dirac_deg3"]["per_seed_caught"],
                                                            R[ref]["per_seed_caught"])
                          for ref in ("smote", "exact_deg3")}
    for n_, v in R["paired"].items():
        print(f"   {n_}: {v['mean']:+.4f} CI {v['ci95']} p={v['wilcoxon_p']}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
print("-> " + OUT)
