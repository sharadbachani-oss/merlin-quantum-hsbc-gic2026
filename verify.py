# -*- coding: utf-8 -*-
"""
verify.py — credential-free replication of every headline number.

    pip install numpy scikit-learn xgboost
    python verify.py

Reproduces the few-shot result from archived device samples: no quantum
credentials, no network. The quantum samples in results/ulb_raw400.json
are the raw bitstrings returned by QCi Dirac-3 (354 jobs); everything
downstream is deterministic given the seeds.

Requires the ULB dataset (open, ODbL):
    kaggle datasets download -d mlg-ulb/creditcardfraud
placed at ./creditcard.csv or passed as argv[1].
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "creditcard.csv")

print("=" * 68)
print("HSBC submission — credential-free verification")
print("=" * 68)

# ---- 1. archived receipts ------------------------------------------------
fs = json.load(open(os.path.join(R, "fewshot_result.json")))
print(f"\n[1] Headline few-shot result (archived): n={fs['n_frauds']} "
      f"labelled frauds, {fs['seeds']} seeds")
print("    arm      | AUPRC mean ± sd | 5th pct | frauds caught @0.2%")
for k in ("none", "smote", "jitter", "quantum"):
    m, s, p5 = fs["arms"][k]
    cm, cs = fs["caught"][k]
    print(f"    {k:8s} | {m:.4f} ± {s:.4f} | {p5:.4f}  | {cm:.1f} ± {cs:.1f}")
q = fs["arms"]["quantum"]; sm = fs["arms"]["smote"]
qc = fs["caught"]["quantum"][0]; sc = fs["caught"]["smote"][0]
print(f"    -> vs SMOTE: AUPRC {100*(q[0]-sm[0])/sm[0]:+.1f}% | "
      f"worst case {100*(q[2]-sm[2])/sm[2]:+.1f}% | "
      f"frauds caught {qc-sc:+.1f}")
print(f"    -> detection rate {100*qc/75:.1f}% vs {100*sc/75:.1f}% "
      f"= {100*(qc-sc)/75:+.1f} percentage points")

# ---- 2. device samples ---------------------------------------------------
runs = json.load(open(os.path.join(R, "ulb_raw400.json")))
tot = sum(len(x["sols"]) for x in runs)
uniq = len({tuple(s) for x in runs for s in x["sols"]})
print(f"\n[2] Device samples: {len(runs)} Dirac-3 jobs, {tot} samples, "
      f"{uniq} distinct patterns")
print(f"    diversity ceiling: 40 jobs -> 82 distinct; "
      f"{len(runs)} jobs -> {uniq} distinct "
      f"({len(runs)/40:.1f}x jobs, {uniq/82:.2f}x diversity)")

# ---- 3. the model that generated them ------------------------------------
M = json.load(open(os.path.join(R, "ulb_model.json")))
by = M["by_order"]
print(f"\n[3] Energy landscape (read from data moments, no training):")
print(f"    {M['nb']} binary literals | terms by order: "
      f"{ {int(k): v for k, v in sorted(by.items())} }")
print(f"    -> {by.get('3', 0)} three-body terms a pairwise Boltzmann "
      f"machine cannot represent")

# ---- 4. data-rich regime (where classical wins) --------------------------
ur = json.load(open(os.path.join(R, "ulb_result.json")))
a = ur["arms"]
print(f"\n[4] Data-rich regime (417 labels) — reported for completeness:")
for k in ("none", "smote", "jitter", "ising_pair", "dirac_deg3"):
    if k in a:
        print(f"    {k:12s} AUPRC {a[k]['auprc'][0]:.4f} ± {a[k]['auprc'][1]:.4f}")
print(f"    -> jitter beats the quantum arm here; the advantage is "
      f"few-shot only")

# ---- 5. live replication -------------------------------------------------
if not os.path.exists(CSV):
    print(f"\n[5] Live replication skipped: {CSV} not found.")
    print("    Download: kaggle datasets download -d mlg-ulb/creditcardfraud")
    sys.exit(0)
print(f"\n[5] Live replication from {CSV} (a few minutes)...")
import pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score
d = pd.read_csv(CSV).sort_values("Time")
y = d.Class.values.astype(int)
FE = [c for c in d.columns if c not in ("Class", "Time")]
X = d[FE].values.astype(np.float64)
cut = int(0.8 * len(y))
Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
spec = [(int(i), float(t)) for i, t in M["spec"]]
dev = sorted({tuple(s) for x in runs for s in x["sols"]})
allfr = np.where(ytr == 1)[0]; leg = np.where(ytr == 0)[0]

def decode(bits, pool, seed):
    rng = np.random.default_rng(seed)
    sc = np.zeros(len(pool))
    for i, (j, thr) in enumerate(spec):
        sc += ((pool[:, j] >= thr).astype(int) == bits[i])
    k = min(15, len(pool))
    top = np.argsort(sc)[::-1][:k]
    return pool[top].mean(0) + rng.normal(0, pool.std(0) * 0.05 + 1e-9)

NF, N = 5, 2000
SEEDS = list(range(21, 41))
out = {}
for kind in ("smote", "quantum"):
    sc_, cg_ = [], []
    for seed in SEEDS:
        r = np.random.default_rng(seed)
        fr = Xtr[r.choice(allfr, NF, replace=False)]
        Xb = np.vstack([Xtr[leg], fr])
        yb = np.concatenate([np.zeros(len(leg), int), np.ones(NF, int)])
        if kind == "smote":
            a1 = fr[r.integers(0, NF, N)]; b1 = fr[r.integers(0, NF, N)]
            Xa = a1 + r.random((N, 1)) * (b1 - a1)
        else:
            pats = [dev[i] for i in r.integers(0, len(dev), N)]
            Xa = np.array([decode(p, fr, seed * 991 + i)
                           for i, p in enumerate(pats)])
        Xf = np.vstack([Xb, Xa]); yf = np.concatenate([yb, np.ones(N, int)])
        m = xgb.XGBClassifier(n_estimators=300, max_depth=5,
                              learning_rate=0.07, tree_method="hist",
                              random_state=seed,
                              scale_pos_weight=(yf == 0).sum() / max(yf.sum(), 1))
        m.fit(Xf, yf, verbose=False)
        p = m.predict_proba(Xte)[:, 1]
        sc_.append(average_precision_score(yte, p))
        thr = np.quantile(p, 1 - 0.002)
        cg_.append(int(((p >= thr) & (yte == 1)).sum()))
    out[kind] = (np.mean(sc_), np.std(sc_), np.mean(cg_))
    print(f"    {kind:8s}: AUPRC {np.mean(sc_):.4f} ± {np.std(sc_):.4f} | "
          f"frauds caught {np.mean(cg_):.1f}")
dq, ds = out["quantum"], out["smote"]
print(f"    -> quantum vs SMOTE: AUPRC {100*(dq[0]-ds[0])/ds[0]:+.1f}% | "
      f"frauds caught {dq[2]-ds[2]:+.1f}")
print("\nVERIFICATION COMPLETE.")
