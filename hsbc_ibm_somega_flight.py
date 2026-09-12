# -*- coding: utf-8 -*-
"""
hsbc_ibm_somega_flight.py — one-command retrieve / (gated) submit for the
6-qubit Heron S(ω) echo family.

    python hsbc_ibm_somega_flight.py              # default: retrieve-or-search
    python hsbc_ibm_somega_flight.py retrieve
    python hsbc_ibm_somega_flight.py emit
    python hsbc_ibm_somega_flight.py score        # on-disk counts only
    python hsbc_ibm_somega_flight.py submit       # REFUSED unless both
                                                  #   --i-accept-the-queue
                                                  #   and HSBC_SOMEGA_SUBMIT=1

Does not invent shots. Does not fly Dirac or the 128q feature map.
Default is NO SUBMIT. Prior annex jobs are listed only to be rejected.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hsbc_hardware_protocol as P  # noqa: E402

WORK = HERE
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
STATE = os.path.join(WORK, "hsbc_somega_state.json")
RECEIPT = os.path.join(OUT, "hsbc_somega_receipt.json")
COUNTS = os.path.join(OUT, "hsbc_somega_counts.json")
SCORE = os.path.join(OUT, "hsbc_somega_score.json")

OPEN_CRN = ("crn:v1:bluemix:public:quantum-computing:us-east:"
            "a/0601b1157c194f5f9dc0b0fe3ff89f31:"
            "4448fcf5-e662-487a-a307-4e13163054a7::")
OPEN_BACKENDS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]
TAG = "hsbc_somega_v1"
SHOTS = 8192

# Known IBM jobs that are NOT this protocol.
REJECT = {
    "daaucejvpcac73dd232g": "128q feature map (annex)",
    "dab6therrl7c7386flo0": "128q generative card (annex)",
    "daba21jvpcac73ddh9h0": "87q data-encoded Boltzmann (annex)",
    "d95c8revtlqs73fv8es0": "first-principles Walsh interference echo lock, not payment-graph S(ω)",
    "d95c8rmvtlqs73fv8esg": "first-principles witness, not this family",
}


def _load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE, encoding="utf-8"))
    return {}


def _save_state(st):
    json.dump(st, open(STATE, "w", encoding="utf-8"), indent=2)


def _write_receipt(payload):
    payload.setdefault("card", "HSBC S(ω) 6q Heron family")
    payload.setdefault("shots_invented", False)
    json.dump(payload, open(RECEIPT, "w", encoding="utf-8"), indent=2)
    return RECEIPT


def _connect():
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(instance=OPEN_CRN)


def emit_manifest():
    proto = P.main()
    fam = P.heron_somega_family(evolve="HG")
    man = {
        "tag": TAG,
        "shots": SHOTS,
        "times": fam["times"],
        "names": fam["names"],
        "n_circuits": fam["n_circuits"],
        "families": list(fam["families"]),
        "frozen_bands": fam["frozen_bands"],
        "evolve": "HG",
        "qasm_example": "hsbc_ibm_somega_t4.8.qasm",
        "protocol_status": proto["status"],
    }
    path = os.path.join(OUT, "hsbc_somega_manifest.json")
    json.dump(man, open(path, "w", encoding="utf-8"), indent=2)
    print("emit:", fam["n_circuits"], "circuits x", SHOTS, "shots ->", path)
    return fam, man


def _qiskit_circuits(fam):
    from qiskit import QuantumCircuit
    circs = []
    for name in fam["names"]:
        fam_name, ttag = name.rsplit("_t", 1)
        t = float(ttag)
        circ = None
        for c in fam["families"][fam_name]:
            if abs(c["t"] - t) < 1e-9:
                circ = QuantumCircuit.from_qasm_str(c["qasm"])
                circ.name = name
                break
        if circ is None:
            raise RuntimeError("missing circuit " + name)
        circs.append(circ)
    return circs


def survival_from_counts(counts) -> float:
    tot = sum(int(v) for v in counts.values())
    if tot <= 0:
        return 0.0
    z = 0
    for bits, n in counts.items():
        sb = str(bits).replace(" ", "")
        if set(sb) <= {"0"}:
            z += int(n)
    return z / tot


def spectral_from_L(times, L):
    x = np.asarray(L, dtype=np.float64)
    w = np.hanning(len(x))
    y = np.fft.fftshift(np.fft.fft((x - x.mean()) * w, n=8 * len(x)))
    dt = float(times[1] - times[0])
    omega = np.fft.fftshift(np.fft.fftfreq(len(y), d=dt)) * 2.0 * math.pi
    S = np.abs(y) ** 2
    S = S / max(float(S.sum()), 1e-30)
    return omega, S


def band_weight(omega, S, lo, hi):
    m = (omega >= lo) & (omega <= hi)
    return float(S[m].sum())


def peak_abs(omega, S, lo=0.05, hi=6.0):
    m = (omega >= lo) & (omega <= hi)
    if not np.any(m):
        return 0.0, 0.0
    k = int(np.argmax(S[m]))
    return float(omega[m][k]), float(S[m][k])


def score_counts(counts_by_name, times=None):
    """Score retrieved counts against frozen bands. Refuses if empty."""
    if not counts_by_name:
        return {
            "status": "NO_COUNTS",
            "sendable": False,
            "note": "no retrieved counts — not scored, not invented",
        }
    times = np.array(times if times is not None else P.FROZEN_BANDS["times"],
                     dtype=float)
    B = P.FROZEN_BANDS
    families = ["null", "ring4", "ring6", "star", "echo"]
    Lfam = {}
    for fam in families:
        L = []
        for t in times:
            key = f"{fam}_t{t:.1f}"
            if key not in counts_by_name:
                return {
                    "status": "INCOMPLETE_FAMILY",
                    "sendable": False,
                    "missing": key,
                    "note": "counts present but not the 5-family protocol",
                }
            L.append(survival_from_counts(counts_by_name[key]))
        Lfam[fam] = np.array(L, dtype=float)

    rows = {}
    for fam, L in Lfam.items():
        omega, S = spectral_from_L(times, L)
        pk, ph = peak_abs(omega, S)
        rows[fam] = {
            "L": [float(x) for x in L],
            "echo_mean": float(np.mean(L)),
            "peak_omega": pk,
            "weight_near_delta": band_weight(omega, S, *B["delta_window"]),
            "weight_near_delta_s": band_weight(omega, S, *B["delta_s_window"]),
        }

    om_pk = rows["echo"]["peak_omega"]
    rel = abs(om_pk - B["echo_closed_form"]) / B["echo_closed_form"]
    gates = {
        "echo_within_5pct": bool(rel <= B["echo_rel_tol"]),
        "null_survival": bool(rows["null"]["echo_mean"] >= B["null_survival_min"]),
        "null_delta_weight": bool(rows["null"]["weight_near_delta"]
                                  > rows["null"]["weight_near_delta_s"]),
        "ring6_Delta_s_gt_star": bool(
            rows["ring6"]["weight_near_delta_s"]
            > rows["star"]["weight_near_delta_s"]
        ),
        "ring6_Delta_s_band": bool(
            rows["ring6"]["weight_near_delta_s"] >= B["ring6_delta_s_weight_min"]
        ),
        "star_Delta_s_band": bool(
            rows["star"]["weight_near_delta_s"] <= B["star_delta_s_weight_max"]
        ),
    }
    passed = all(gates.values())
    return {
        "status": "SCORED_FROM_RETRIEVED_COUNTS",
        "sendable": bool(passed),
        "gates": gates,
        "families": rows,
        "echo_rel_err": float(rel),
        "frozen_bands": B,
        "note": "survival is |L|^2 (all-zero). FFT is of that real series.",
    }


def _job_looks_like_family(job) -> tuple[bool, str]:
    jid = job.job_id()
    if jid in REJECT:
        return False, REJECT[jid]
    tags = list(getattr(job, "tags", None) or [])
    name = str(getattr(job, "name", "") or "")
    blob = " ".join(tags + [name]).lower()
    if TAG in blob or "hsbc_somega" in blob or "somega_v1" in blob:
        return True, "tag/name match"
    return False, "no hsbc_somega tag"


def stage_retrieve():
    """Search IBM + local state for this exact family. Never invents."""
    emit_manifest()
    st = _load_state()
    found = []
    reasons = []
    counts = None
    job_id = st.get("job") or (sys.argv[2] if len(sys.argv) > 2 else None)

    # local counts already on disk
    if os.path.exists(COUNTS):
        on_disk = json.load(open(COUNTS, encoding="utf-8"))
        if on_disk.get("counts") and on_disk.get("protocol_tag") == TAG:
            print("local counts:", COUNTS, "job", on_disk.get("job"))
            scored = score_counts(on_disk["counts"])
            json.dump(scored, open(SCORE, "w", encoding="utf-8"), indent=2)
            rec = {
                "status": "RETRIEVED_LOCAL",
                "job": on_disk.get("job"),
                "sendable": scored.get("sendable"),
                "score": SCORE,
                "rejected_priors": REJECT,
            }
            _write_receipt(rec)
            return 0 if scored.get("sendable") else 1

    try:
        svc = _connect()
    except Exception as e:
        rec = {
            "status": "PROTOCOL_ONLY",
            "matching_job": None,
            "sendable": False,
            "ibm_connect": "FAIL " + type(e).__name__,
            "rejected_priors": REJECT,
            "next": "python hsbc_ibm_somega_flight.py retrieve",
            "submit_is": "OFF unless HSBC_SOMEGA_SUBMIT=1 and --i-accept-the-queue",
        }
        _write_receipt(rec)
        print("IBM connect failed:", type(e).__name__, e)
        print("receipt:", RECEIPT)
        return 2

    # explicit id
    if job_id:
        try:
            job = svc.job(job_id)
            ok, why = _job_looks_like_family(job)
            reasons.append({job_id: why})
            if not ok and job_id in REJECT:
                print("REFUSED: %s is %s" % (job_id, REJECT[job_id]))
            elif ok or (st.get("job") == job_id and st.get("tag") == TAG):
                found.append(job)
        except Exception as e:
            reasons.append({job_id: "lookup FAIL " + type(e).__name__})

    # tagged search
    try:
        recent = list(svc.jobs(limit=80))
    except TypeError:
        recent = list(svc.jobs())[:80]
    except Exception as e:
        recent = []
        reasons.append({"list": type(e).__name__ + " " + str(e)[:160]})

    inspected = []
    for j in recent:
        try:
            jid = j.job_id()
            inspected.append(jid)
            ok, why = _job_looks_like_family(j)
            if jid in REJECT:
                reasons.append({jid: REJECT[jid]})
                continue
            if ok:
                found.append(j)
                reasons.append({jid: why})
        except Exception as e:
            reasons.append({"row": type(e).__name__})

    # de-dupe
    seen = set()
    uniq = []
    for j in found:
        if j.job_id() not in seen:
            seen.add(j.job_id())
            uniq.append(j)
    found = uniq

    if not found:
        rec = {
            "status": "PROTOCOL_ONLY",
            "matching_job": None,
            "sendable": False,
            "inspected_job_ids": inspected[:40],
            "rejected_priors": REJECT,
            "reasons": reasons[-20:],
            "n_circuits": 40,
            "shots": SHOTS,
            "family": ["null", "ring4", "ring6", "star", "echo"],
            "next_flight": (
                "python hsbc_ibm_somega_flight.py submit "
                "--i-accept-the-queue   # still requires HSBC_SOMEGA_SUBMIT=1"
            ),
            "default": "NO SUBMIT",
        }
        _write_receipt(rec)
        st.update(tag=TAG, matching_job=None, last_search=time.strftime("%Y%m%d_%H%M%S"))
        _save_state(st)
        print("no matching 6q S(omega) family job")
        print("receipt:", RECEIPT)
        return 0

    job = found[0]
    job_id = job.job_id()
    print("retrieving", job_id)
    res = job.result()
    names = st.get("names")
    if not names:
        # try job tags / inputs length
        n = len(res)
        fam = P.heron_somega_family(evolve="HG")
        names = fam["names"]
        if n != len(names):
            rec = {
                "status": "JOB_SHAPE_MISMATCH",
                "job": job_id,
                "n_result": n,
                "n_family": len(names),
                "sendable": False,
                "note": "job exists but is not the 40-circuit family",
            }
            _write_receipt(rec)
            print("shape mismatch:", n, "vs", len(names))
            return 1
    counts = {}
    for i, nm in enumerate(names):
        try:
            counts[nm] = dict(res[i].data.c.get_counts())
        except Exception:
            counts[nm] = dict(res[i].data.meas.get_counts())
    backend = None
    try:
        backend = job.backend().name
    except Exception:
        backend = st.get("backend")
    blob = {
        "card": "HSBC S(ω) COUNTS (retrieved, not invented)",
        "protocol_tag": TAG,
        "job": job_id,
        "backend": backend,
        "names": names,
        "shots": SHOTS,
        "counts": counts,
    }
    json.dump(blob, open(COUNTS, "w", encoding="utf-8"), indent=1)
    scored = score_counts(counts)
    json.dump(scored, open(SCORE, "w", encoding="utf-8"), indent=2)
    st.update(tag=TAG, job=job_id, backend=backend, names=names, counts=COUNTS)
    _save_state(st)
    rec = {
        "status": "RETRIEVED",
        "job": job_id,
        "backend": backend,
        "sendable": scored.get("sendable"),
        "gates": scored.get("gates"),
        "counts": COUNTS,
        "score": SCORE,
    }
    _write_receipt(rec)
    print("retrieved", job_id, "->", COUNTS)
    print("sendable:", scored.get("sendable"), scored.get("gates"))
    return 0 if scored.get("sendable") else 1


def stage_submit():
    """Submit ONLY if the operator set both the flag and the env var."""
    accept = "--i-accept-the-queue" in sys.argv
    env_ok = os.environ.get("HSBC_SOMEGA_SUBMIT") == "1"
    if not (accept and env_ok):
        print("REFUSED: submit is gated.")
        print("  required: HSBC_SOMEGA_SUBMIT=1 and --i-accept-the-queue")
        print("  default is NO SUBMIT. retrieve first:")
        print("    python hsbc_ibm_somega_flight.py retrieve")
        rec = {
            "status": "PROTOCOL_ONLY",
            "matching_job": _load_state().get("job"),
            "sendable": False,
            "submit": "REFUSED (gate closed)",
            "next_flight": (
                "set HSBC_SOMEGA_SUBMIT=1 && "
                "python hsbc_ibm_somega_flight.py submit --i-accept-the-queue"
            ),
        }
        _write_receipt(rec)
        return 2

    fam, man = emit_manifest()
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2
    svc = _connect()
    pend = {}
    for b in OPEN_BACKENDS:
        try:
            pend[b] = svc.backend(b).status().pending_jobs
        except Exception:
            pend[b] = 10 ** 6
    name = min(pend, key=pend.get)
    backend = svc.backend(name)
    print("queue", pend, "->", name)
    circs = _qiskit_circuits(fam)
    qcs = transpile(circs, backend, optimization_level=1)
    s = SamplerV2(mode=backend)
    s.options.dynamical_decoupling.enable = True
    s.options.dynamical_decoupling.sequence_type = "XpXm"
    job = s.run(qcs, shots=SHOTS)
    st = dict(tag=TAG, backend=name, job=job.job_id(), names=fam["names"],
              shots=SHOTS, n_circuits=len(qcs), submitted=time.strftime("%Y%m%d_%H%M%S"))
    _save_state(st)
    rec = {
        "status": "SUBMITTED",
        "job": job.job_id(),
        "backend": name,
        "n_circuits": len(qcs),
        "shots": SHOTS,
        "sendable": False,
        "note": "submitted; retrieve when done. no counts invented.",
    }
    _write_receipt(rec)
    print("HSBC S(omega) submitted:", job.job_id(),
          "(%d x %d on %s)" % (len(qcs), SHOTS, name))
    return 0


def stage_score():
    if not os.path.exists(COUNTS):
        print("REFUSED: no retrieved counts. run retrieve.")
        return 2
    blob = json.load(open(COUNTS, encoding="utf-8"))
    if blob.get("protocol_tag") != TAG:
        print("REFUSED: counts file is not this protocol tag")
        return 2
    scored = score_counts(blob.get("counts") or {})
    json.dump(scored, open(SCORE, "w", encoding="utf-8"), indent=2)
    print("score ->", SCORE, "sendable:", scored.get("sendable"))
    rec = {
        "status": scored.get("status"),
        "job": blob.get("job"),
        "sendable": scored.get("sendable"),
        "gates": scored.get("gates"),
    }
    _write_receipt(rec)
    return 0 if scored.get("sendable") else 1


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "retrieve"
    if cmd in ("emit", "design"):
        emit_manifest()
        rec = {
            "status": "PROTOCOL_ONLY",
            "matching_job": None,
            "sendable": False,
            "emitted": True,
            "default": "NO SUBMIT",
        }
        _write_receipt(rec)
        return 0
    if cmd in ("retrieve", "search"):
        return stage_retrieve()
    if cmd == "score":
        return stage_score()
    if cmd in ("submit", "fly"):
        return stage_submit()
    print("usage: python hsbc_ibm_somega_flight.py "
          "[retrieve|emit|score|submit]")
    return 2


if __name__ == "__main__":
    sys.exit(main())
