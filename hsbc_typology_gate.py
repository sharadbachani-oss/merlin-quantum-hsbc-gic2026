#!/usr/bin/env python3
"""hsbc_typology_gate.py — does the framework's relaxation spectrum of an entity
neighbourhood carry COLLECTIVE fraud/laundering signal that cheap graph statistics
do not? The pre-flight gate for a fresh HSBC submission.

WHY. Twenty receipted HSBC formulations are null, including the field-carrying
response oracle (paired AUPRC lift +0.0009, CI spanning zero). Every one of them
appended a per-neighbourhood feature to a tree on IEEE-CIS, where the raw
neighbourhood itself adds nothing (+raw_neighbourhood == baseline): individual
card-not-present fraud is not a collective object on that data. The framework's
one admissible advantage class is real-time collective response, and the package
already measured that laundering MOTIFS (4-ring, 6-ring, smurfing star) produce
relaxation spectra at total-variation distance 1.0 from the null graph under
H = kappa D_G - A_G (results/hsbc_spectral_route.json). What was never done is
the label test on data where collective typologies EXIST. Elliptic (Bitcoin
illicit-entity graph, public, on this PC) is that data.

THE CLAIMED OBJECT (unchanged from the receipt). For each labelled transaction:
take its transaction-graph neighbourhood (<= NMAX nodes), build H on that graph
exactly as hsbc_spectral_route.build_H does, quench from |+>^n, read the
Loschmidt echo L(t) and its spectral density S(omega), the ground energy and gap,
band weights at the receipted motif lines, echo contrast, and the entanglement
ladder across the middle cut. Those are the SPECTRAL features.

THE FLOOR (cheapest adversary, first). Cheap graph scalars on the same
neighbourhood: n, m, degree stats, clustering, triangle count, short-cycle counts
(girth <= 6), plus Elliptic's own aggregated-neighbourhood features. A GBM must
NOT be able to reconstruct the spectral features from the scalars (CV R2 < 0.8 on
>= 3 spectral features), else the spectrum is a graph statistic in disguise.

THE LABEL TEST. LightGBM arms on the standard temporal split (steps 1-34 train,
35-49 test), five temporal folds over the test window, illicit as positive:
  baseline          = Elliptic features
  +graph_scalars    = + cheap scalars                 (the STRONGEST control)
  +spectral         = + cheap scalars + spectral
  +shuffled         = + cheap scalars + spectral permuted across rows
  +random           = + cheap scalars + N(0,1) columns of the same width
Paired lift = AUPRC(+spectral) - AUPRC(+graph_scalars) per fold, 95% CI.

DECISION RULE (fixed here, before running):
  PASS: floor passes AND paired lift mean >= 0.02 AUPRC with CI95 > 0.
  FAIL: anything else. FAIL means the fresh HSBC submission is not written.

USAGE
  python hsbc_typology_gate.py smoke                    # 300 neighbourhoods, checks the pipeline
  python hsbc_typology_gate.py features [--nmax 12] [--workers 30] [--max-per-class 6000]
  python hsbc_typology_gate.py floor
  python hsbc_typology_gate.py label
ENV  ELLIPTIC_DIR (default C:\\hsbc\\elliptic\\elliptic_bitcoin_dataset)
Outputs next to this file: typology_features_elliptic.npz, typology_floor.json,
typology_gate.json.
"""
import json, math, os, sys, time, argparse
from collections import defaultdict, Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ELL = os.environ.get("ELLIPTIC_DIR", r"C:\hsbc\elliptic\elliptic_bitcoin_dataset")
SQRT5 = math.sqrt(5.0)
KAPPA = 3.0 / (3.0 - SQRT5)
DELTA = math.sqrt((KAPPA / 2) ** 2 + 4) - KAPPA / 2
TIMES = np.arange(0.0, 40.0, 0.25)          # resolves the receipted lines 0.4 .. 2.5 and the gap
MOTIF_BANDS = [(0.25, 0.60), (0.60, 1.10), (1.10, 2.00), (2.00, 3.00)]   # receipted line positions
TRAIN_STEPS, TEST_STEPS = range(1, 35), range(35, 50)


# ------------------------------------------------------------- data ---------
def load_elliptic():
    import pandas as pd
    feats = pd.read_csv(os.path.join(ELL, "elliptic_txs_features.csv"), header=None)
    classes = pd.read_csv(os.path.join(ELL, "elliptic_txs_classes.csv"))
    edges = pd.read_csv(os.path.join(ELL, "elliptic_txs_edgelist.csv"))
    feats = feats.rename(columns={0: "txId", 1: "step"})
    lab = classes.set_index("txId")["class"].astype(str)
    y = lab.reindex(feats["txId"]).map({"1": 1, "2": 0}).values      # illicit=1, licit=0, unknown=NaN
    tx = feats["txId"].values
    idx = {t: i for i, t in enumerate(tx)}
    adj = defaultdict(set)
    for a, b in zip(edges["txId1"].values, edges["txId2"].values):
        if a in idx and b in idx:
            adj[idx[a]].add(idx[b]); adj[idx[b]].add(idx[a])
    X = feats.drop(columns=["txId"]).values.astype(np.float32)   # step + 165 features
    step = feats["step"].values.astype(int)
    smax = int(step.max())
    if smax < 49:
        print("WARNING: features file is PARTIAL — time steps 1..%d of 49 (%d rows of 203,769). "
              "The standard temporal split (train <= 34, test >= 35) is not available." % (smax, len(step)), flush=True)
    return X, y, step, adj


def split_steps(step, smoke):
    """standard Elliptic split; in smoke mode only, fall back to 70/30 of the available steps"""
    smax = int(step.max())
    if smax >= 49:
        return 34, [(35, 37), (38, 40), (41, 43), (44, 46), (47, 49)], "standard"
    if not smoke:
        raise SystemExit("REFUSING the gate on a partial features file (max step %d). Fetch the full "
                         "elliptic_txs_features.csv (203,769 rows, steps 1..49) and re-run." % smax)
    cut = max(2, int(0.7 * smax)); rest = list(range(cut + 1, smax + 1))
    k = max(1, len(rest) // 2)
    folds = [(rest[i], rest[min(i + k - 1, len(rest) - 1)]) for i in range(0, len(rest), k)]
    return cut, folds, "SMOKE-ONLY fallback split at step %d (not the standard split)" % cut


def neighbourhood(i, adj, nmax):
    """BFS from i, closest first, capped at nmax nodes; returns node list (i first) and edges."""
    order, seen, frontier = [i], {i}, [i]
    while frontier and len(order) < nmax:
        nxt = []
        for u in frontier:
            for v in sorted(adj[u], key=lambda v: len(adj[v]), reverse=True):
                if v not in seen and len(order) < nmax:
                    seen.add(v); order.append(v); nxt.append(v)
        frontier = nxt
    pos = {v: k for k, v in enumerate(order)}
    E = sorted({(pos[u], pos[v]) if pos[u] < pos[v] else (pos[v], pos[u])
                for u in order for v in adj[u] if v in pos and u != v})
    return order, E


# ----------------------------------------------------- spectral features ----
def build_H(n, edges):
    dim = 1 << n
    H = np.zeros((dim, dim))
    s = np.arange(dim)
    zbits = [(1.0 - 2.0 * ((s >> q) & 1)) for q in range(n)]
    diag = np.zeros(dim)
    for a, b in edges:
        diag += (KAPPA / 2.0) * (1.0 - zbits[a] * zbits[b])
    H[s, s] = diag
    for q in range(n):
        H[s ^ (1 << q), s] -= 1.0
    return H


def spectral_features(n, edges):
    if n < 2:
        return None
    H = build_H(n, edges)
    w, V = np.linalg.eigh(H)
    psi0 = np.ones(1 << n) / math.sqrt(1 << n)
    c = V.T @ psi0
    L = np.exp(-1j * np.outer(TIMES, w)) @ (c * c)                     # Loschmidt echo
    x = L * np.hanning(len(L))
    yf = np.fft.fftshift(np.fft.fft(x, n=8 * len(x)))
    om = np.fft.fftshift(np.fft.fftfreq(len(yf), d=0.25)) * 2 * np.pi
    S = np.abs(yf) ** 2; S /= max(S.sum(), 1e-30)
    a = np.abs(L); k0 = int(np.argmin(a[: max(4, len(a) // 5)]))
    contrast = float(a[k0:].max() / a[0])
    bands = [float(S[(om >= lo) & (om <= hi)].sum()) for lo, hi in MOTIF_BANDS]
    peak = float(om[np.argmax(S * (om > 0.05))])
    cut = n // 2
    ent = []
    for t in (2.0, 8.0):
        psi = V @ (np.exp(-1j * w * t) * c)
        M = psi.reshape((1 << cut, 1 << (n - cut)))
        sv = np.linalg.svd(M, compute_uv=False) ** 2; sv = sv[sv > 1e-16]; sv /= sv.sum()
        ent.append(float(-(sv * np.log(sv)).sum()))
    return np.array([w[0], w[1] - w[0], (w[1] - w[0]) - DELTA, contrast, peak, *bands, *ent,
                     float(np.abs(L).mean()), float(np.abs(L[-40:]).mean())])


SPECTRAL_NAMES = ["E0", "gap", "gap_shift", "echo_contrast", "peak_omega", "band1", "band2",
                  "band3", "band4", "S_t2", "S_t8", "echo_mean", "echo_late"]


# ------------------------------------------------------- cheap scalars -------
def cheap_scalars(n, edges):
    deg = np.zeros(n); nb = defaultdict(set)
    for a, b in edges:
        deg[a] += 1; deg[b] += 1; nb[a].add(b); nb[b].add(a)
    m = len(edges)
    tri = sum(len(nb[a] & nb[b]) for a, b in edges) / 3.0
    # short cycles through the target (node 0) of length 4 and 6 via BFS layers
    def cycles_through0(L):
        cnt = 0
        for u in nb[0]:
            for v in nb[0]:
                if u < v:
                    if L == 4:
                        cnt += len((nb[u] & nb[v]) - {0})
                    else:
                        for x in nb[u] - {0, v}:
                            cnt += len((nb[x] & nb[v]) - {0, u})
        return cnt
    clust = 0.0
    for v in range(n):
        d = len(nb[v])
        if d >= 2:
            e = sum(1 for a in nb[v] for b in nb[v] if a < b and b in nb[a])
            clust += 2.0 * e / (d * (d - 1))
    return np.array([n, m, deg[0], deg.mean(), deg.max(), m / max(n, 1), tri, cycles_through0(4),
                     cycles_through0(6), clust / max(n, 1)])


SCALAR_NAMES = ["n", "m", "deg0", "deg_mean", "deg_max", "density", "triangles", "c4_through0",
                "c6_through0", "clustering"]

# ------------------------------------------------------------- stages -------
_G = {}


def _init(edge_arr, nmax):
    """workers rebuild adjacency from a compact int32 (m, 2) array: cheap to ship on Windows spawn"""
    adj = defaultdict(set)
    for a, b in edge_arr:
        adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    _G.update(adj=adj, nmax=nmax)


def _one(i):
    order, E = neighbourhood(i, _G["adj"], _G["nmax"])
    n = len(order)
    sf = spectral_features(n, E)
    cs = cheap_scalars(n, E)
    return i, n, (sf if sf is not None else np.full(len(SPECTRAL_NAMES), np.nan)), cs


def features(nmax, workers, max_per_class, smoke=False, shard=None):
    """Serial within a process. Parallelism is by SHARDING across processes
    (--shard k/N), which is robust where multiprocessing pools hang on Windows.
    `workers` is kept only for the in-process pool path, used if > 1."""
    t0 = time.time()
    X, y, step, adj = load_elliptic()
    lab = np.where(~np.isnan(y))[0]
    rng = np.random.default_rng(7)
    ill = lab[y[lab] == 1]; lic = lab[y[lab] == 0]
    if smoke:
        ill, lic = rng.choice(ill, 150, replace=False), rng.choice(lic, 150, replace=False)
    else:
        if max_per_class and len(lic) > max_per_class:
            lic = rng.choice(lic, max_per_class, replace=False)
        if max_per_class and len(ill) > max_per_class:
            ill = rng.choice(ill, max_per_class, replace=False)
    sel = np.sort(np.concatenate([ill, lic]))
    tag = "_smoke" if smoke else ""
    if shard is not None:
        k, N = shard
        sel = sel[k::N]; tag += "_shard%dof%d" % (k, N)
    print("elliptic loaded %s in %.0fs | labelled %d | selected %d (illicit %d) | nmax %d | %s"
          % (X.shape, time.time() - t0, len(lab), len(sel), int((y[sel] == 1).sum()), nmax,
             ("shard %d/%d" % shard if shard else "workers %d" % workers)), flush=True)
    edge_arr = np.array([(a, b) for a in adj for b in adj[a] if a < b], dtype=np.int32)
    if workers > 1 and shard is None:
        from multiprocessing import Pool
        with Pool(workers, initializer=_init, initargs=(edge_arr, nmax)) as P:
            res = P.map(_one, sel.tolist(), chunksize=16)
    else:
        _init(edge_arr, nmax)
        res = []
        for j, i in enumerate(sel):
            res.append(_one(int(i)))
            if j % 200 == 0:
                print("  %d/%d  (%.0fs)" % (j + 1, len(sel), time.time() - t0), flush=True)
    ii = np.array([r[0] for r in res]); nn = np.array([r[1] for r in res])
    SF = np.stack([r[2] for r in res]); CS = np.stack([r[3] for r in res])
    ok = ~np.isnan(SF).any(axis=1)
    print("features done in %.0fs | usable %d/%d | n: median %d max %d" % (time.time() - t0, ok.sum(), len(ok), np.median(nn), nn.max()), flush=True)
    out = os.path.join(HERE, "typology_features_elliptic%s.npz" % tag)
    np.savez(out, idx=ii[ok], n=nn[ok], spectral=SF[ok], scalars=CS[ok], X=X[ii[ok]], y=y[ii[ok]], step=step[ii[ok]],
             names_spectral=np.array(SPECTRAL_NAMES), names_scalars=np.array(SCALAR_NAMES))
    print("-> %s" % out)
    return out


def merge(N, tag=""):
    """concatenate the N shard files into typology_features_elliptic<tag>.npz"""
    parts = [np.load(os.path.join(HERE, "typology_features_elliptic%s_shard%dof%d.npz" % (tag, k, N))) for k in range(N)]
    keys = ["idx", "n", "spectral", "scalars", "X", "y", "step"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["names_spectral"], out["names_scalars"] = parts[0]["names_spectral"], parts[0]["names_scalars"]
    order = np.argsort(out["idx"])
    for k in keys:
        out[k] = out[k][order]
    p = os.path.join(HERE, "typology_features_elliptic%s.npz" % tag)
    np.savez(p, **out)
    print("merged %d shards -> %s (%d rows, illicit %d)" % (N, p, len(out["y"]), int((out["y"] == 1).sum())))
    return p


def floor(path=None):
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_predict, KFold
    d = np.load(path or os.path.join(HERE, "typology_features_elliptic.npz"))
    SF, CS, X = d["spectral"], d["scalars"], d["X"]
    # the floor's inputs: cheap scalars + Elliptic's own aggregated (neighbourhood) features (cols 94..)
    agg = X[:, 94:] if X.shape[1] > 94 else X
    Z = np.hstack([CS, agg])
    r2 = []
    for j, nm in enumerate(SPECTRAL_NAMES):
        t = SF[:, j]
        if np.std(t) < 1e-12:
            r2.append(1.0); continue
        m = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, verbose=-1)
        p = cross_val_predict(m, Z, t, cv=KFold(5, shuffle=True, random_state=1))
        r2.append(float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()))
    below = int(sum(1 for v in r2 if v < 0.8))
    verdict = "GO" if below >= 3 else "STOP: spectrum is reconstructible from cheap scalars"
    rec = dict(card="cheap-scalar floor on the typology spectrum (Elliptic)", n=int(len(SF)),
               r2_gbm_cv5={nm: round(v, 4) for nm, v in zip(SPECTRAL_NAMES, r2)},
               features_below_0p8=below, rule="GO iff >= 3 spectral features with CV R2 < 0.8", verdict=verdict)
    print(json.dumps(rec, indent=1))
    json.dump(rec, open(os.path.join(HERE, "typology_floor.json"), "w"), indent=1)
    return verdict.startswith("GO")


def label(path=None, smoke=False):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    d = np.load(path or os.path.join(HERE, "typology_features_elliptic.npz"))
    SF, CS, X, y, step = d["spectral"], d["scalars"], d["X"], d["y"].astype(int), d["step"]
    cut, fold_ranges, split_name = split_steps(step, smoke)
    print("split:", split_name, flush=True)
    tr = np.where(step <= cut)[0]; te = np.where(step > cut)[0]
    folds = [np.where((step >= a) & (step <= b))[0] for a, b in fold_ranges]
    folds = [f for f in folds if len(f) and y[f].sum() > 0 and (y[f] == 0).sum() > 0]
    if len(folds) < 2 or len(tr) == 0:
        raise SystemExit("not enough labelled test folds for a paired test (folds=%d, train=%d)" % (len(folds), len(tr)))
    rng = np.random.default_rng(3)
    SHUF = SF[rng.permutation(len(SF))]; RND = rng.normal(size=SF.shape)
    arms = {"baseline": lambda i: X[i],
            "+graph_scalars": lambda i: np.hstack([X[i], CS[i]]),
            "+spectral": lambda i: np.hstack([X[i], CS[i], SF[i]]),
            "+shuffled": lambda i: np.hstack([X[i], CS[i], SHUF[i]]),
            "+random": lambda i: np.hstack([X[i], CS[i], RND[i]])}

    def fit(Xtr, ytr, Xte):
        spw = (1 - ytr.mean()) / max(ytr.mean(), 1e-6)
        m = lgb.LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.6, scale_pos_weight=spw, random_state=21, verbose=-1)
        m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]
    out = {}
    for name, fn in arms.items():
        aps, aucs = [], []
        for f in folds:
            p = fit(fn(tr), y[tr], fn(f)); aps.append(float(average_precision_score(y[f], p))); aucs.append(float(roc_auc_score(y[f], p)))
        out[name] = dict(auprc_mean=float(np.mean(aps)), auprc_sd=float(np.std(aps)), auc_mean=float(np.mean(aucs)), folds=aps)
        print("%-16s AUPRC %.4f ± %.4f  AUC %.4f" % (name, out[name]["auprc_mean"], out[name]["auprc_sd"], out[name]["auc_mean"]), flush=True)
    dlt = np.array(out["+spectral"]["folds"]) - np.array(out["+graph_scalars"]["folds"])
    k = len(dlt)
    ci = [float(dlt.mean() - 1.96 * dlt.std(ddof=1) / math.sqrt(k)), float(dlt.mean() + 1.96 * dlt.std(ddof=1) / math.sqrt(k))]
    passed = bool(dlt.mean() >= 0.02 and ci[0] > 0)
    rec = dict(card="typology spectrum label test on Elliptic, temporal split, illicit positive",
               split=split_name, standard_split=bool(split_name == "standard"),
               n_train=int(len(tr)), n_test=int(len(te)), illicit_rate_test=float(y[te].mean()), folds=k,
               arms=out, paired_lift_spectral_vs_graph_scalars=dict(mean=float(dlt.mean()), ci95=ci),
               rule="PASS iff mean lift >= 0.02 AUPRC and CI95 lower bound > 0", verdict="PASS" if passed else "FAIL")
    print("paired lift +spectral vs +graph_scalars:", rec["paired_lift_spectral_vs_graph_scalars"], "->", rec["verdict"])
    json.dump(rec, open(os.path.join(HERE, "typology_gate.json"), "w"), indent=1)
    return passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage")
    ap.add_argument("--nmax", type=int, default=12); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-per-class", type=int, default=6000)
    ap.add_argument("--shard", default=None, help="k/N: this process computes every N-th selected transaction starting at k")
    ap.add_argument("--n-shards", type=int, default=None, help="merge: number of shard files to concatenate")
    ap.add_argument("--tag", default="", help="merge: filename tag, e.g. _smoke")
    ap.add_argument("--path", default=None, help="floor/label: features file to grade (default typology_features_elliptic.npz)")
    ap.add_argument("--smoke-split", action="store_true", help="label: allow the smoke-only fallback split")
    a = ap.parse_args()
    shard = tuple(int(v) for v in a.shard.split("/")) if a.shard else None
    if a.stage == "smoke":
        p = features(min(a.nmax, 10), 1, 150, smoke=True, shard=shard)
        if shard is None:
            floor(p); label(p, smoke=True)
    elif a.stage == "features":
        features(a.nmax, a.workers, a.max_per_class, shard=shard)
    elif a.stage == "merge":
        if not a.n_shards:
            raise SystemExit("merge needs --n-shards N")
        merge(a.n_shards, a.tag)
    elif a.stage == "floor":
        sys.exit(0 if floor(a.path) else 1)
    elif a.stage == "label":
        sys.exit(0 if label(a.path, smoke=a.smoke_split) else 1)
