# -*- coding: utf-8 -*-
"""mechanism_collective.py -- exploratory (no rule fixed in advance), laptop, 2026-09-14 evening.

Question: does the entity-graph response oracle react COLLECTIVELY to a coordinated neighbourhood (a ring of
transactions coupled to each other, as in card-not-present fraud rings) rather than as a sum of local pairwise
effects (as in loosely coupled legitimate neighbourhoods)?  If yes, the response carries a mechanism for fraud
signal that a per-transaction or additive model cannot represent, and that mechanism grows with neighbourhood
size -- which is where the classical simulation wall sits.

Protocol (exploratory):
  n = 6 neighbours (14 qubits, exact), fields and target couplings drawn once per draw and held fixed.
  Ladder in m = number of neighbour-neighbour couplings switched on (0..15 of the 15 pairs, in a fixed random
  order), all at shared count 1 and tight timing (|dt| << tau, so the effective coupling is full).
  Collective share at full clique: response(clique) minus [response(0) + sum over pairs of (response(pair) -
  response(0))], i.e. the deviation from pairwise additivity, relative to the full response.
  Also: burst (m = 15, tight timing) vs local (m = 15, timings spread over the week) separation on the response
  features, with the cheap scalars and the effective-coupling sum as classical controls.
Outputs mechanism_collective.json.
"""
import os, sys, json, math, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hsbc_field_response as F

N_NB = 6; DRAWS = 24; PAIRS = [(q, q2) for q in range(1, N_NB + 1) for q2 in range(q + 1, N_NB + 1)]
LADDER = [0, 1, 2, 3, 5, 8, 11, 15]

def draw(seed):
    rng = np.random.default_rng(seed)
    zs = rng.normal(size=N_NB + 1); ws = list(rng.choice([1, 2, 3], size=N_NB, p=[0.6, 0.3, 0.1]))
    dts_tight = list(3600.0 + rng.uniform(0, 600, size=N_NB))                       # all within 10 min, 1 h ago
    dts_spread = list(np.exp(rng.uniform(math.log(60), math.log(F.TRAIL_S), size=N_NB)))
    order = list(rng.permutation(len(PAIRS)))
    return zs, ws, dts_tight, dts_spread, order

def job(args):
    seed = args
    zs, ws, dt_t, dt_s, order = draw(seed)
    out = dict(seed=seed, ladder={}, pairs={}, spread=None)
    base = F.response_features(zs, ws, dt_t, {})
    out["ladder"]["0"] = base.tolist()
    for m in LADDER[1:]:
        nb = {PAIRS[i]: 1 for i in order[:m]}
        out["ladder"][str(m)] = F.response_features(zs, ws, dt_t, nb).tolist()
    for i in range(len(PAIRS)):
        out["pairs"][str(i)] = (F.response_features(zs, ws, dt_t, {PAIRS[i]: 1}) - base).tolist()
    nb_all = {p: 1 for p in PAIRS}
    out["spread"] = F.response_features(zs, ws, dt_s, nb_all).tolist()
    out["scalars_tight"] = F.cheap_scalars(zs, ws, dt_t, nb_all).tolist()
    out["scalars_spread"] = F.cheap_scalars(zs, ws, dt_s, nb_all).tolist()
    return out

if __name__ == "__main__":
    t0 = time.time()
    with Pool(8) as pool:
        res = pool.map(job, list(range(DRAWS)))
    print(f"{DRAWS} draws in {time.time()-t0:.0f}s", flush=True)
    names = [f"{k}@t{t}" for t in F.TIMES for k in ("self", "nb_mean", "nb_absmax")]
    L = {m: np.array([r["ladder"][str(m)] for r in res]) for m in LADDER}
    add = np.array([np.array(r["ladder"]["0"]) + sum(np.array(v) for v in r["pairs"].values()) for r in res])
    full = L[15]
    resid = full - add
    share = np.abs(resid).mean(0) / np.maximum(np.abs(full).mean(0), 1e-12)
    print("feature            ", "  ".join(f"{n:>13s}" for n in names))
    for m in LADDER:
        print(f"m={m:2d} mean response", "  ".join(f"{v:13.4f}" for v in L[m].mean(0)))
    print("additive prediction", "  ".join(f"{v:13.4f}" for v in add.mean(0)))
    print("collective share   ", "  ".join(f"{v:13.3f}" for v in share))
    # burst vs spread separation on response features vs classical controls
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    Xr = np.vstack([full, np.array([r["spread"] for r in res])]); y = np.r_[np.ones(DRAWS), np.zeros(DRAWS)]
    Xs = np.vstack([np.array([r["scalars_tight"] for r in res]), np.array([r["scalars_spread"] for r in res])])
    aucs = {}
    for nm, X in (("response", Xr), ("cheap_scalars", Xs), ("effective_J_sum_only", Xs[:, -1:]), ("scalars_without_timing", Xs[:, :7])):
        aucs[nm] = float(cross_val_score(GradientBoostingClassifier(n_estimators=100, max_depth=2), X, y, cv=4, scoring="roc_auc").mean())
    print("burst vs spread AUC:", aucs)
    json.dump(dict(protocol=__doc__, draws=DRAWS, ladder=LADDER, feature_names=names,
                   ladder_mean={str(m): L[m].mean(0).tolist() for m in LADDER}, ladder_sd={str(m): L[m].std(0).tolist() for m in LADDER},
                   additive_prediction_mean=add.mean(0).tolist(), collective_share=share.tolist(), burst_vs_spread_auc=aucs, raw=res),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mechanism_collective.json"), "w"), indent=1)
    print("-> mechanism_collective.json")
