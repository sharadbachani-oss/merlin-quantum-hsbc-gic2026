"""HSBC - field-carrying entity-graph response feature, FLOOR FIRST.

Redesign after floor_response_feature.json killed the graph-shape oracle (its Hamiltonian saw only neighbour count and
weight multiset). Here transaction-level quantities enter the operator itself:
  local field      h_p Z_{a_p}         h_p = ALPHA * z_p,   z_p = standardised log(1+amount) of node p (target included)
  target coupling  J_{0q} Z Z          J   = G * w_q * exp(-dt_q / TAU)        (shared-entity count, decayed by recency)
  nb-nb coupling   J_{qq'} Z Z         J   = G * s_{qq'} * exp(-|dt_q-dt_q'| / TAU)   (entities shared between neighbours)
Pair internal operator, product ground state, signed RZ(+-pi/2) probe on the target, evolve, half-difference = retarded
response. Nine features: (self, mean-neighbour, max|neighbour|) x TIMES.

Modes
  python hsbc_field_response.py floor            synthetic neighbourhoods drawn from data-like distributions; grades the
                                                 9 features against CHEAP SCALARS (quadratic, and a gradient-boosted tree
                                                 with 5-fold CV) and checks that fixed graph shape no longer fixes the feature.
                                                 Writes floor_field_response.json.
  python hsbc_field_response.py label [NPROC]    real IEEE-CIS: frozen protocol of hsbc_response_test.py (same split, gray
                                                 zone, arm subsample, folds, LightGBM budget) with one extra control arm,
                                                 +raw_neighbourhood (the per-neighbour inputs the oracle sees, given
                                                 directly to the tree). Writes hsbc_field_response.json.
Decision rule, fixed before the floor ran: go to the label test only if at least three of the nine features have
gradient-boosted CV R^2 < 0.8 against the cheap scalars AND the within-shape spread ratio exceeds 0.3. Label bar as
before: paired gray-zone AUPRC lift over the STRONGEST control (+raw_neighbourhood), 95% CI above zero.
"""
import os, sys, time, json, math
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import expm_multiply
HERE = os.path.dirname(os.path.abspath(__file__))
MU = 3.9270509831248424; G = 0.5; ALPHA = 0.5; K = 6; TAU = 86400.0; TRAIL_S = 7 * 86400; TIMES = (0.5, 1.0, 2.0)
N_TRAIN_SUB = 60000; N_ARM_TRAIN = 8000
if os.environ.get("HSBC_TIMES"): TIMES = tuple(float(x) for x in os.environ["HSBC_TIMES"].split(","))
TAG = os.environ.get("HSBC_TAG", "")

Z1 = csr_matrix(np.diag([1.0, -1.0])); X1 = csr_matrix(np.array([[0, 1], [1, 0]], float)); I1 = identity(2, format="csr")
def op_on(q, op, nq):
    mats = [I1] * nq; mats[q] = op; M = mats[0]
    for m in mats[1:]: M = kron(M, m, format="csr")
    return M
_h4 = MU * 0.5 * (np.eye(4) - np.kron(np.diag([1, -1]), np.diag([1, -1]))) - np.kron(np.array([[0, 1], [1, 0]]), np.eye(2)) - np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
_g4 = np.linalg.eigh(_h4)[1][:, 0]

def response_features(zs, ws, dts, nbnb):
    """zs: standardised log-amounts, node 0 = target; ws, dts: per neighbour; nbnb: dict {(q,q2): shared count}."""
    n = len(zs)
    if n == 1: return np.zeros(9)
    nq = 2 * n; I = identity(2 ** nq, format="csr")
    Za = [op_on(2 * p, Z1, nq) for p in range(n)]
    H = csr_matrix((2 ** nq, 2 ** nq))
    for p in range(n):
        H = H + MU * 0.5 * (I - Za[p] @ op_on(2 * p + 1, Z1, nq)) - op_on(2 * p, X1, nq) - op_on(2 * p + 1, X1, nq) + ALPHA * zs[p] * Za[p]
    for q, (w, dt_) in enumerate(zip(ws, dts), start=1):
        H = H + G * w * math.exp(-dt_ / TAU) * (Za[0] @ Za[q])
    for (q, q2), s in nbnb.items():
        H = H + G * s * math.exp(-abs(dts[q - 1] - dts[q2 - 1]) / TAU) * (Za[q] @ Za[q2])
    psi = _g4
    for _ in range(n - 1): psi = np.kron(psi, _g4)
    feats = []
    for t in TIMES:
        vals = {}
        for sgn in (+1, -1):
            rz = np.exp(-1j * sgn * (math.pi / 4) * np.repeat([1.0, -1.0], 2 ** (nq - 1)))
            phi = expm_multiply(-1j * H * t, rz * psi)
            vals[sgn] = (np.vdot(phi, Za[0] @ phi).real, [np.vdot(phi, Za[q] @ phi).real for q in range(1, n)])
        self_r = 0.5 * (vals[+1][0] - vals[-1][0]); nr = 0.5 * (np.array(vals[+1][1]) - np.array(vals[-1][1]))
        feats += [self_r, float(nr.mean()), float(np.abs(nr).max())]
    return np.array(feats)

def cheap_scalars(zs, ws, dts, nbnb):
    n = len(zs) - 1
    if n == 0: return np.zeros(12)
    Jt = [G * w * math.exp(-d / TAU) for w, d in zip(ws, dts)]
    Jnn = sum(G * s * math.exp(-abs(dts[q - 1] - dts[q2 - 1]) / TAU) for (q, q2), s in nbnb.items())
    return np.array([n, sum(ws), max(ws), np.mean(ws), zs[0], np.mean(zs[1:]), np.max(zs[1:]),
                     math.log(min(dts)), np.mean(np.log(dts)), sum(Jt), max(Jt), Jnn])
SCALAR_NAMES = ["n", "sum_w", "max_w", "mean_w", "z_target", "mean_z_nb", "max_z_nb", "log_min_dt", "mean_log_dt", "sum_J_target", "max_J_target", "sum_J_nbnb"]

def _sample_shape(rng):
    n = int(rng.integers(1, K + 1))
    ws = list(rng.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1]))
    nbnb = {}
    for q in range(1, n + 1):
        for q2 in range(q + 1, n + 1):
            if rng.random() < 0.3: nbnb[(q, q2)] = int(rng.choice([1, 2], p=[0.8, 0.2]))
    return n, ws, nbnb
def _sample_fields(rng, n):
    zs = rng.normal(size=n + 1); dts = np.exp(rng.uniform(math.log(60), math.log(TRAIL_S), size=n))
    return zs, list(dts)

def floor(M=600, n_shapes=40, reps=5):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict
    rng = np.random.default_rng(7); F = []; S = []; t0 = time.time()
    for m in range(M):
        n, ws, nbnb = _sample_shape(rng); zs, dts = _sample_fields(rng, n)
        F.append(response_features(zs, ws, dts, nbnb)); S.append(cheap_scalars(zs, ws, dts, nbnb))
        if m % 100 == 0: print(f"  floor {m}/{M}  {time.time()-t0:.0f}s", flush=True)
    F = np.array(F); S = np.array(S)
    Xq = np.c_[S, S ** 2, np.ones(len(S))]
    for a in range(S.shape[1]):
        for b in range(a + 1, S.shape[1]): Xq = np.c_[Xq, S[:, a] * S[:, b]]
    r2_quad, r2_gbm = [], []
    for k in range(9):
        yk = F[:, k]; ss = ((yk - yk.mean()) ** 2).sum() + 1e-15
        beta = np.linalg.lstsq(Xq, yk, rcond=None)[0]; r2_quad.append(float(1 - ((yk - Xq @ beta) ** 2).sum() / ss))
        pred = cross_val_predict(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05), S, yk, cv=5)
        r2_gbm.append(float(1 - ((yk - pred) ** 2).sum() / ss))
    within = []; rng2 = np.random.default_rng(11)
    for _ in range(n_shapes):
        n, ws, nbnb = _sample_shape(rng2); fs = []
        for _ in range(reps):
            zs, dts = _sample_fields(rng2, n); fs.append(response_features(zs, ws, dts, nbnb))
        within.append(np.std(np.array(fs), axis=0))
    ratio = (np.mean(within, axis=0) / (F.std(axis=0) + 1e-12)).tolist()
    n_low = sum(r < 0.8 for r in r2_gbm); go = bool(n_low >= 3 and np.mean(ratio) > 0.3)
    out = dict(card="cheap-scalar floor on the FIELD-CARRYING response feature", date=time.strftime("%Y-%m-%d"),
               M=M, cheap_scalars=SCALAR_NAMES, r2_quadratic_in_sample=[round(x, 4) for x in r2_quad],
               r2_gbm_cv5=[round(x, 4) for x in r2_gbm], within_shape_spread_ratio=[round(x, 3) for x in ratio],
               features_with_gbm_r2_below_0p8=int(n_low), mean_spread_ratio=round(float(np.mean(ratio)), 3),
               decision_rule="go to label test iff >=3 features with GBM CV R2 < 0.8 and mean spread ratio > 0.3",
               verdict="GO: response carries configuration information the cheap scalars do not" if go else "STOP: feature is a function of cheap scalars - withdrawn again")
    json.dump(out, open(os.path.join(HERE, "floor_field_response.json"), "w"), indent=1)
    print(f"\n{'feature':>8} {'quad R2':>8} {'GBM cvR2':>9} {'spread':>7}")
    names = [f"{a}@{t}" for t in TIMES for a in ("self", "mean", "max")]
    for k in range(9): print(f"{names[k]:>8} {r2_quad[k]:8.3f} {r2_gbm[k]:9.3f} {ratio[k]:7.2f}")
    print("\n" + out["verdict"]); return out

def _label_worker(args):
    i, zs, ws, dts, nbnb = args
    return i, response_features(zs, ws, dts, nbnb)

def label(nproc=4):
    import pandas as pd, lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    from collections import defaultdict
    import multiprocessing as mp
    CSV = os.path.join(HERE, "data", "train_transaction.csv"); IDF = os.path.join(HERE, "data", "train_identity.csv")
    rng = np.random.default_rng(0); t0 = time.time()
    df = pd.read_csv(CSV); idf = pd.read_csv(IDF); df = df.merge(idf, on="TransactionID", how="left")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    y = df["isFraud"].values; cut = int(0.8 * len(y)); print(f"loaded {df.shape} in {time.time()-t0:.0f}s; split at {cut}", flush=True)
    ent = {k: df[k].astype(str).values for k in ("card1", "addr1", "P_emaildomain")}
    dt = df["TransactionDT"].values.astype(np.int64); amt = df["TransactionAmt"].values
    la = np.log1p(amt); z_all = (la - la[:cut].mean()) / la[:cut].std()
    feat = df.drop(columns=["isFraud", "TransactionID"]).copy()
    for c in feat.columns:
        if not pd.api.types.is_numeric_dtype(feat[c]): feat[c] = feat[c].astype("category").cat.codes
    X = feat.values.astype(np.float32)
    def fit_score(Xtr, ytr, Xte):
        spw = (1 - ytr.mean()) / max(ytr.mean(), 1e-6)
        clf = lgb.LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                                 scale_pos_weight=spw, n_jobs=max(2, (os.cpu_count() or 4) - 2), random_state=21, verbose=-1)
        clf.fit(Xtr, ytr); return clf.predict_proba(Xte)[:, 1]
    sub = rng.choice(cut, size=min(N_TRAIN_SUB, cut), replace=False); sub.sort()
    p_te = fit_score(X[sub], y[sub], X[cut:])
    lo, hi = np.quantile(p_te, [0.90, 0.92]); band = (p_te >= lo) & (p_te <= hi); gz_idx = np.where(band)[0] + cut
    print(f"gray zone [{lo:.3f},{hi:.3f}] n={len(gz_idx)} fraud {y[gz_idx].mean():.3f}", flush=True)
    last = {k: defaultdict(list) for k in ent}
    def neighbourhood(i):
        cand = defaultdict(int)
        for k in ent:
            v = ent[k][i]
            if v in ("nan", ""): continue
            for j in last[k][v]:
                if dt[i] - dt[j] <= TRAIL_S and j < i: cand[j] += 1
        if not cand: return [], [], {}
        js = sorted(cand, key=lambda j: -dt[j])[:K]; ws = [cand[j] for j in js]
        nbnb = {}
        for a in range(len(js)):
            for b in range(a + 1, len(js)):
                s = sum(1 for k in ent if ent[k][js[a]] == ent[k][js[b]] and ent[k][js[a]] not in ("nan", ""))
                if s: nbnb[(a + 1, b + 1)] = s
        return js, ws, nbnb
    fr = np.where(y[:cut] == 1)[0]; le = np.where(y[:cut] == 0)[0]; nf = int(round(N_ARM_TRAIN * y[:cut].mean()))
    arm_sub = np.sort(np.concatenate([rng.choice(fr, nf, replace=False), rng.choice(le, N_ARM_TRAIN - nf, replace=False)]))
    targets = set(gz_idx.tolist()) | set(arm_sub.tolist()); nb = {}
    for i in range(len(df)):
        if i in targets: nb[i] = neighbourhood(i)
        for k in ent:
            v = ent[k][i]
            if v not in ("nan", ""): last[k][v].append(i)
    print("neighbourhoods built", flush=True)
    idx_all = np.array(sorted(targets)); resp = np.zeros((len(df), 9)); rel = np.zeros((len(df), 4)); raw = np.zeros((len(df), 3 * K + 1))
    train_fraud = np.zeros(len(df)); train_fraud[:cut] = y[:cut]
    jobs = []
    for i in idx_all:
        js, ws, nbnb = nb[i]
        zs = np.r_[z_all[i], z_all[js]] if js else np.array([z_all[i]]); dts = [float(dt[i] - dt[j]) for j in js]
        jobs.append((int(i), zs, ws, dts, nbnb))
        if js:
            rel[i] = [len(js), amt[js].mean(), amt[js].max(), train_fraud[js].mean()]
            r = np.zeros(3 * K + 1); r[0] = z_all[i]
            for q, j in enumerate(js): r[1 + 3 * q: 4 + 3 * q] = [ws[q], math.log1p(dts[q]), z_all[j]]
            raw[i] = r
    t0 = time.time(); done = 0
    with mp.Pool(nproc) as pool:
        for i, f in pool.imap_unordered(_label_worker, jobs, chunksize=16):
            resp[i] = f; done += 1
            if done % 1000 == 0: print(f"  oracle {done}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(HERE, f"field_resp_features{TAG}.npy"), resp[idx_all])
    folds = np.array_split(gz_idx, 5); rng_s = np.random.default_rng(3)
    shuf = resp.copy(); shuf[idx_all] = resp[idx_all][rng_s.permutation(len(idx_all))]
    rnd = rng_s.normal(size=resp.shape)
    arms = {"baseline": lambda idx: X[idx], "+relational": lambda idx: np.hstack([X[idx], rel[idx]]),
            "+raw_neighbourhood": lambda idx: np.hstack([X[idx], rel[idx], raw[idx]]),
            "+response": lambda idx: np.hstack([X[idx], rel[idx], raw[idx], resp[idx]]),
            "+shuffled_response": lambda idx: np.hstack([X[idx], rel[idx], raw[idx], shuf[idx]]),
            "+random": lambda idx: np.hstack([X[idx], rel[idx], raw[idx], rnd[idx]])}
    out = {}
    for name, fn in arms.items():
        aps, aucs = [], []
        for f in folds:
            tr = arm_sub; p = fit_score(fn(tr), y[tr], fn(f)); aps.append(float(average_precision_score(y[f], p))); aucs.append(float(roc_auc_score(y[f], p)))
        out[name] = dict(auprc_mean=float(np.mean(aps)), auprc_sd=float(np.std(aps)), auc_mean=float(np.mean(aucs)), folds=aps)
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
    d = np.array(out["+response"]["folds"]) - np.array(out["+raw_neighbourhood"]["folds"])
    paired = dict(mean_lift=float(d.mean()), ci95=[float(d.mean() - 1.96 * d.std(ddof=1) / math.sqrt(5)), float(d.mean() + 1.96 * d.std(ddof=1) / math.sqrt(5))])
    res = dict(gray_zone=dict(lo=float(lo), hi=float(hi), n=int(len(gz_idx)), fraud_rate=float(y[gz_idx].mean())), K=K, tau_s=TAU, alpha=ALPHA, g=G, times=TIMES,
               arms=out, paired_lift_response_vs_raw=paired, oracle_wall_s=round(time.time() - t0, 1))
    json.dump(res, open(os.path.join(HERE, f"hsbc_field_response{TAG}.json"), "w"), indent=1); print("paired lift vs +raw_neighbourhood:", paired); print("done")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "floor"
    if mode == "floor": floor()
    else: label(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
