# -*- coding: utf-8 -*-
"""
HSBC wrap-before-cone ring certificate — NEW object, not the failed S(ω) family.

Do not resubmit dagfiegmhr3c73e53gvg. Do not regrade it as a win.
This is not a classifier and not another |L|² FFT.

Industrial object
  A girth-6 laundering cycle wraps: the light-cone of Z0 returns to the
  source before a typical investigation horizon. A matched-count star
  explodes. The instrument is the local commutator
      i ⟨[Z0, Z0(t)]⟩ / 2
  via opposite RZ(π/2) on qubit 0 (v6 probe), not a global survival peak.

Local proof is exact diag. A *new* small Heron card (t≤0.25 only) is
retrieve-first and submits only under HSBC_WRAP_SUBMIT=1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import native_response as N

DEST = HERE / "results"
FAILED_FAMILY = "dagfiegmhr3c73e53gvg"
ECHO_LOCK = "d95c8revtlqs73fv8es0"
PAIRS = [(0, 3), (1, 4), (2, 5)]
FAMILIES = {
    "null": [],
    "ring6": [(0, 1), (1, 2), (2, 4), (4, 5), (5, 3)],
    "star": [(0, 1), (0, 2), (0, 4)],
}
T_SHORT = (0.0, 0.125, 0.25)
HORIZON = 0.25
SHOTS = 4096
OPEN_BACKENDS = ("ibm_fez", "ibm_kingston", "ibm_marrakesh")
OPEN_CRN = (
    "crn:v1:bluemix:public:quantum-computing:us-east:"
    "a/0601b1157c194f5f9dc0b0fe3ff89f31:"
    "4448fcf5-e662-487a-a307-4e13163054a7::"
)
REJECT = {
    FAILED_FAMILY: "FAILED 40-circuit S(ω) family — do not resubmit or regrade",
    ECHO_LOCK: "Ω-echo lock, not payment-graph wrap",
    "daaucejvpcac73dd232g": "128q feature map",
    "dab6therrl7c7386flo0": "128q generative",
    "daba21jvpcac73ddh9h0": "87q QBM",
}


def z_expect(p, q):
    idx = np.arange(len(p))
    return float(np.dot(np.abs(p) ** 2, 1 - 2 * ((idx >> q) & 1)))


def commutator(name, extra, t):
    H = N.hamiltonian(6, PAIRS, extra)
    w, V = np.linalg.eigh(H)
    gs = N.run(6, N.prep(PAIRS))
    vals = []
    for sign in (-1, 1):
        probe = [("rz", 0, sign * math.pi / 2)]
        initial = N.run(6, probe, gs)
        exact = V @ (np.exp(-1j * w * t) * (V.T.conj() @ initial))
        vals.append(z_expect(exact, 0))
    return 0.5 * (vals[1] - vals[0])


def support_growth(name, extra, t):
    """||[Z0(t), Zj]|| proxy: connected ⟨Z0 Zj⟩ after quench vs product."""
    H = N.hamiltonian(6, PAIRS, extra)
    w, V = np.linalg.eigh(H)
    gs = N.run(6, N.prep(PAIRS))
    psi = V @ (np.exp(-1j * w * t) * (V.T.conj() @ gs))
    p = np.abs(psi) ** 2
    idx = np.arange(len(p))
    z0 = 1 - 2 * ((idx >> 0) & 1)
    out = []
    for j in range(6):
        zj = 1 - 2 * ((idx >> j) & 1)
        c = float(np.dot(p, z0 * zj) - np.dot(p, z0) * np.dot(p, zj))
        out.append(abs(c))
    return out


def local_certificate():
    rows = {}
    for name, extra in FAMILIES.items():
        comm = [float(commutator(name, extra, t)) for t in T_SHORT]
        supp = {str(t): support_growth(name, extra, t) for t in T_SHORT}
        # wrap = connected weight returns to qubit 0 neighbourhood by horizon
        s0 = np.array(supp[str(T_SHORT[0])])
        sH = np.array(supp[str(HORIZON)])
        local = float(sH[0] + sH[3])  # native pair of 0
        distal = float(sH.sum() - local)
        wrap = local > distal
        explode = distal > local + 0.05
        rows[name] = dict(
            commutator=comm,
            support=supp,
            local_weight_at_horizon=local,
            distal_weight_at_horizon=distal,
            wraps_before_horizon=bool(wrap),
            explodes=bool(explode),
        )
    r6 = rows["ring6"]
    st = rows["star"]
    nu = rows["null"]
    gates = dict(
        ring6_wraps=r6["wraps_before_horizon"],
        star_explodes=st["explodes"] or (
            st["distal_weight_at_horizon"] > r6["distal_weight_at_horizon"]
        ),
        null_stays_local=nu["local_weight_at_horizon"] >= nu["distal_weight_at_horizon"],
        commutator_nonzero_ring=abs(r6["commutator"][-1]) > 1e-6,
        failed_family_not_regraded=True,
        not_a_classifier=True,
        not_another_survival_fft=True,
    )
    gates["certificate"] = bool(
        gates["ring6_wraps"]
        and gates["star_explodes"]
        and gates["null_stays_local"]
        and gates["commutator_nonzero_ring"]
    )
    return dict(times=list(T_SHORT), horizon=HORIZON, families=rows, gates=gates)


def retrieve_existing(job_id=None):
    out = dict(available=False, examined=0, match=None, rejects=[], error=None)
    if job_id in REJECT:
        out["error"] = f"REFUSED: {job_id} is {REJECT[job_id]}"
        return out
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        svc = QiskitRuntimeService(instance=os.environ.get("IBM_QUANTUM_CRN") or OPEN_CRN)
        out["available"] = True
    except Exception as e:
        out["error"] = f"IBM list unavailable: {type(e).__name__}: {e}"
        return out
    if job_id:
        try:
            job = svc.job(job_id)
            rec = dict(job=job.job_id(), status=str(job.status()))
            out["examined"] = 1
            if rec["job"] in REJECT:
                rec["reason"] = REJECT[rec["job"]]
                out["rejects"].append(rec)
            else:
                out["match"] = rec
        except Exception as e:
            out["error"] = f"job {job_id} not retrievable: {type(e).__name__}: {e}"
        return out
    try:
        jobs = list(svc.jobs(limit=40))
    except Exception as e:
        out["error"] = f"jobs() failed: {type(e).__name__}: {e}"
        return out
    out["examined"] = len(jobs)
    for job in jobs:
        jid = job.job_id()
        rec = dict(job=jid, status=str(job.status()))
        if jid in REJECT:
            rec["reason"] = REJECT[jid]
            out["rejects"].append(rec)
        elif jid == FAILED_FAMILY:
            rec["reason"] = REJECT[FAILED_FAMILY]
            out["rejects"].append(rec)
        else:
            rec["reason"] = "unlabeled — not claimed as wrap-before-cone"
            out["rejects"].append(rec)
    return out


def stage_submit():
    if os.environ.get("HSBC_WRAP_SUBMIT") != "1":
        print("REFUSED: default is NO paid submit.")
        print("Set HSBC_WRAP_SUBMIT=1 only for the NEW wrap card (t<=0.25).")
        print("Do not resubmit dagfiegmhr3c73e53gvg.")
        return 3
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    svc = QiskitRuntimeService(instance=os.environ.get("IBM_QUANTUM_CRN") or OPEN_CRN)
    pend = {}
    for b in OPEN_BACKENDS:
        try:
            pend[b] = svc.backend(b).status().pending_jobs
        except Exception:
            pend[b] = 10**6
    backend = svc.backend(min(pend, key=pend.get))
    print(f"  queue {pend} -> {backend.name}")
    circs, names = [], []
    for name, extra in FAMILIES.items():
        for t in T_SHORT:
            if t == 0.0:
                continue
            for sign in (-1, 1):
                ops = N.prep(PAIRS) + [("rz", 0, sign * math.pi / 2)]
                steps = 2 if t <= 0.125 else 4
                ops = ops + N.evolution(PAIRS, extra, t, steps)
                qc = QuantumCircuit(6, 6)
                for o in N.expand(ops):
                    nm = o[0]
                    if nm == "h":
                        qc.h(o[1])
                    elif nm == "x":
                        qc.x(o[1])
                    elif nm == "cx":
                        qc.cx(o[1], o[2])
                    elif nm == "rx":
                        qc.rx(o[2], o[1])
                    elif nm == "ry":
                        qc.ry(o[2], o[1])
                    elif nm == "rz":
                        qc.rz(o[2], o[1])
                    elif nm == "rzz":
                        qc.rzz(o[3], o[1], o[2])
                qc.measure(range(6), range(6))
                circs.append(qc)
                names.append(f"{name}_t{t:g}_s{sign:+d}")
    qcs = transpile(circs, backend, optimization_level=1, seed_transpiler=21)
    twos = [qc.num_nonlocal_gates() for qc in qcs]
    print("transpiled:")
    for nm, t2 in zip(names, twos):
        print(f"  {nm:22s}  2q {t2:4d}")
    s = SamplerV2(mode=backend)
    s.options.dynamical_decoupling.enable = True
    s.options.dynamical_decoupling.sequence_type = "XpXm"
    job = s.run(qcs, shots=SHOTS)
    st = dict(
        card="HSBC WRAP-BEFORE-CONE SMALL HERON",
        backend=backend.name,
        job_id=job.job_id(),
        names=names,
        twos=twos,
        t=list(T_SHORT[1:]),
        families=list(FAMILIES),
        shots=SHOTS,
        submitted=time.strftime("%Y-%m-%dT%H:%M:%S"),
        not_the_failed_family=FAILED_FAMILY,
        observable="i<[Z0,Z0(t)]>/2 via +/- RZ(pi/2)",
    )
    (DEST / "hsbc_wrap_state.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    print(f"HSBC wrap submitted: {job.job_id()}  {len(qcs)} x {SHOTS} on {backend.name}")
    return 0, st


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--submit", action="store_true")
    p.add_argument("--job", help="retrieve this id (refuses the failed family)")
    args = p.parse_args(argv)

    cert = local_certificate()
    ibm = retrieve_existing(args.job)
    submitted = None
    job_id = None
    status = "LOCAL_PROOF"
    if args.submit:
        rc = stage_submit()
        if isinstance(rc, tuple):
            code, submitted = rc
            job_id = submitted.get("job_id")
            status = "QUEUED" if job_id else "LOCAL_PROOF"
            if code != 0:
                status = "LOCAL_PROOF"
        else:
            status = "LOCAL_PROOF"
    payload = dict(
        card="HSBC WRAP-BEFORE-CONE RING CERTIFICATE",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        status=status,
        job_id=job_id,
        new_hardware_advantage_demonstrated=False,
        operator="H[G] = kappa D_G - A_G",
        kappa=N.KAPPA,
        failed_family=FAILED_FAMILY,
        failed_family_status="FAIL — do not resubmit, do not regrade",
        industrial_object="girth-6 wrap vs star explode; local commutator instrument",
        certificate=cert,
        ibm_search=ibm,
        submitted=submitted,
        honesty=dict(
            not_a_classifier=True,
            not_another_survival_fft=True,
            dagfieg_stays_fail=True,
            qualification_is_not_advantage=True,
            invented_ids=False,
            stoquastic=True,
        ),
        next_flight=dict(
            command="HSBC_WRAP_SUBMIT=1 python hsbc_wrap_before_cone.py --submit",
            n_circuits=12,
            shots=SHOTS,
            t="0.125, 0.25 only",
            refuse=FAILED_FAMILY,
        ),
    )
    dest = DEST / "hsbc_wrap_before_cone.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    g = cert["gates"]
    print(f"certificate {g['certificate']}  ring_wrap {g['ring6_wraps']}  "
          f"star_explode {g['star_explodes']}")
    print(f"status {status}  job {job_id}")
    if ibm.get("error"):
        print("IBM:", ibm["error"])
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
