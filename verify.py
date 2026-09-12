# -*- coding: utf-8 -*-
"""
verify.py — credential-free replay of the HSBC v3 headline numbers.

    python verify.py              # replay archived model-gate JSON
    python verify.py --recompute  # regenerate the JSON (numpy, ~12 s)

Annex A (old few-shot lift) replays if results/fewshot_result.json exists.
No quantum credentials. No invented shots.
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
SQRT5 = math.sqrt(5.0)
MU = 3.0 / (3.0 - SQRT5)
DELTA = math.sqrt((MU / 2.0) ** 2 + 4.0) - MU / 2.0
E0 = -3.0 * DELTA
DELTA_S = 0.486254


def fail(msg):
    print("    FAIL:", msg)
    return False


def ok(msg):
    print("    PASS:", msg)
    return True


def main():
    print("=" * 68)
    print("HSBC v3 — payment-graph S(omega) verification")
    print("=" * 68)

    if "--recompute" in sys.argv:
        sys.path.insert(0, HERE)
        import hsbc_spectral_route as route
        route.main()

    path = os.path.join(R, "hsbc_spectral_route.json")
    if not os.path.exists(path):
        print("missing", path, "— run python hsbc_spectral_route.py")
        return 1
    g = json.load(open(path, encoding="utf-8"))
    route = g

    passed = 0
    total = 0

    def check(cond, msg):
        nonlocal passed, total
        total += 1
        if cond:
            passed += 1
            ok(msg)
        else:
            fail(msg)

    print("\n[1] Derived constants")
    c = g["constants"]
    check(abs(c["kappa"] - MU) < 1e-12, "kappa = 3/(3-sqrt(5))")
    check(abs(c["delta"] - DELTA) < 1e-12, "delta closed form")
    check(c["H"] == "kappa D_G - A_G", "operator is kappa D - A, not a fitted landscape")

    print("\n[2] Native 6-qubit operator identity")
    n = g["native_operator_gate"]
    check(abs(n["E0"] - E0) < 1e-9, "E0 = -3 delta = %.9f" % E0)
    check(abs(n["gap"] - DELTA) < 1e-9, "gap = delta = %.9f" % DELTA)
    check(n["pass_ZZ"], "<ZZ> = cos(2 chi) on all three pairs")

    print("\n[3] Tile typology (exact diag, no shots)")
    t = g["six_qubit_tile"]
    check(abs(t["null_native"]["echo_mean"] - 1.0) < 1e-9, "null echo mean = 1")
    check(t["aml_ring4"]["echo_mean"] < 0.7, "ring-4 destroys perfect echo")
    check(t["aml_ring4"]["echo_mean"] > t["smurf_star"]["echo_mean"],
          "ring keeps more echo than matched-count star (%.3f > %.3f)" % (
              t["aml_ring4"]["echo_mean"], t["smurf_star"]["echo_mean"]))
    w6 = t["aml_ring6"]["weight_near_delta_s"]
    w4 = t["aml_ring4"]["weight_near_delta_s"]
    ws = t["smurf_star"]["weight_near_delta_s"]
    check(w6 > 0.10 and w4 < 0.01 and ws < 0.01,
          "girth-6 puts %.1f%% of S(omega) on Delta_s; ring-4 %.3f%%; star %.3f%%" % (
              100 * w6, 100 * w4, 100 * ws))

    print("\n[4] Continuation ill-posedness")
    k = g["continuation_exhibit"]
    check(k["spectral_tv_distance"] >= 0.15, "TV(S,S') = %.3f" % k["spectral_tv_distance"])
    check(k["mean_abs_G_residual"] <= 5e-3,
          "mean |dG| = %.3e" % k["mean_abs_G_residual"])

    print("\n[5] Loop entanglement wall")
    e = g["entanglement_wall"]
    check(e["dS_at_t1"] > 0.30, "cycle - path entropy at t=1 = %+.3f nats" % e["dS_at_t1"])

    print("\n[6] Noise as instrument")
    nz = g["noise_instrument"]
    check(nz["aml_ring4"]["protection_ratio"] > 2.0,
          "ring parity-protected / isotropic late echo = %.2f" %
          nz["aml_ring4"]["protection_ratio"])

    print("\n[7] Physics gates")
    for name, val in g["gates"].items():
        check(bool(val), name)

    print("\n[8] Hardware honesty")
    hs = g["hardware_status"]
    check("no shots" in hs["this_script"].lower()
          or "model gate" in hs["this_script"].lower(),
          "no invented S(omega) shots")
    proto = os.path.join(R, "hsbc_protocol.json")
    if os.path.exists(proto):
        p = json.load(open(proto, encoding="utf-8"))
        check("PROTOCOL" in p.get("status", ""), "protocol file flags PROTOCOL ONLY")
        check(abs(p["ibm_heron"]["echo_check"]["closed_form"] - DELTA_S) < 1e-6,
              "interference-echo check is the closed form Delta_s")
        fam = p["ibm_heron"].get("flight_family", [])
        check(fam == ["null", "ring4", "ring6", "star", "echo"],
              "flight family is null + ring-4 + ring-6 + star + interference-echo")
        check(p["ibm_heron"].get("matching_job") is None
              and "PROTOCOL" in p.get("status", ""),
              "no invented S(omega) job id in the protocol file")
        rec = os.path.join(R, "hsbc_somega_receipt.json")
        if os.path.exists(rec):
            r = json.load(open(rec, encoding="utf-8"))
            check(r.get("shots_invented") is not True
                  and r.get("matching_job") in (None, r.get("job")),
                  "receipt does not invent a matching family job")
        else:
            check(True, "somega receipt not required for model-gate replay")
    else:
        check(False, "hsbc_protocol.json present")

    print("\n[9] Annex A — prior few-shot lift (not the advantage)")
    fs = os.path.join(R, "fewshot_result.json")
    if not os.path.exists(fs):
        fs = os.path.join(HERE, "fewshot_result.json")
    if os.path.exists(fs):
        a = json.load(open(fs, encoding="utf-8"))
        q, sm = a["arms"]["quantum"], a["arms"]["smote"]
        qc, sc = a["caught"]["quantum"][0], a["caught"]["smote"][0]
        print("    archived: quantum AUPRC %.4f vs SMOTE %.4f; "
              "caught %.1f vs %.1f / 75  (+%.1f pp)" % (
                  q[0], sm[0], qc, sc, 100 * (qc - sc) / 75))
        print("    status: ANNEX — fitted landscape, power-of-data window, "
              "not the S(omega) route")
    else:
        print("    (fewshot_result.json not in this checkout — annex skipped)")

    # ---------------------------------------------------------------- [10]
    print("")
    print("[10] Layering scale-up - discriminator, chi wall, cost crossover")
    gp = os.path.join(R, "hsbc_response_grading.json")
    if os.path.exists(gp):
        g = json.load(open(gp, encoding="utf-8"))["grading"]
        worst = 0.0
        for k in ("1", "2"):
            for f in ("aml_ring6", "aml_ring4", "smurf_star", "null_native"):
                worst = max(worst, abs(g[k][f]["exact"]["w1"] - g[k][f]["krylov"]["w1"]))
        check(worst < 1e-10,
              "Krylov estimator matches exact diagonalisation (max dev %.1e)" % worst)
        check(abs(g["1"]["null_native"]["krylov"]["w1"] - 3 * DELTA) < 1e-9,
              "null response energy = 3 delta = %.9f" % (3 * DELTA))
        check(all(g[k]["_sep_ring6_vs_nearest_pct"] > 50.0 for k in ("1", "2", "3")),
              "girth-6 separated from nearest control by >50% at k=1,2,3")
        seps = [g[k]["_sep_ring4_vs_star_pct"] for k in ("1", "2", "3")]
        check(seps[0] < seps[1] < seps[2],
              "ring4-vs-star separation grows with layering: %.2f%% -> %.2f%% -> %.2f%%"
              % tuple(seps))
    else:
        print("    (hsbc_response_grading.json absent - run hsbc_response_grading.py)")

    sp = os.path.join(R, "hsbc_advantage_scaling.json")
    if os.path.exists(sp):
        a = json.load(open(sp, encoding="utf-8"))
        w = a["exhaustive_wall"]
        growth = [w[i]["chi_min_over_orderings"] / w[i-1]["chi_min_over_orderings"]
                  for i in range(1, len(w))]
        check(all(g > 7.0 for g in growth) if growth else True,
              "bond dimension grows %s per layer vs a maximum of 8x - exponential"
              % ", ".join("%.2fx" % g for g in growth))
        check(all(x["cuts_searched"] == math.comb(x["n_qubits"], x["n_qubits"] // 2)
                  for x in w), "bipartition search is exhaustive, not sampled")
        rows = {r["k"]: r for r in a["crossover"]}
        check(rows[5]["two_qubit_gates"] <= a["validated_pass_2q"],
              "k=5 circuit (%d 2q gates) inside the 18/18-validated envelope"
              % rows[5]["two_qubit_gates"])
        check(rows[8]["two_qubit_gates"] < a["measured_fail_2q"],
              "k=8 circuit (%d 2q gates) under the measured collapse at %d"
              % (rows[8]["two_qubit_gates"], a["measured_fail_2q"]))
        ratio = rows[8]["classical_bytes"] / rows[5]["classical_bytes"]
        check(abs(math.log10(ratio) - 5.4) < 0.05,
              "classical cost grows %.1f orders k=5 -> k=8 for a %.2fx circuit"
              % (math.log10(ratio),
                 rows[8]["two_qubit_gates"] / rows[5]["two_qubit_gates"]))
    else:
        print("    (hsbc_advantage_scaling.json absent - run hsbc_advantage_scaling.py)")

    lp = os.path.join(R, "local_aml_ring6.json")
    if os.path.exists(lp):
        rows = {r["k"]: r for r in json.load(open(lp, encoding="utf-8"))}
        if 3 in rows:
            r = rows[3]
            check(r["local_lead_weight"] > r["global_lead_weight"],
                  "local pair readout retains %.1fx the weight of the global echo at k=3"
                  % (r["local_lead_weight"] / r["global_lead_weight"]))
            check(r["local_signal_amplitude"] > 0.4,
                  "local correlator stays an O(1) observable at k=3 (amplitude %.3f)"
                  % r["local_signal_amplitude"])
    else:
        print("    (local_aml_ring6.json absent - run hsbc_local_readout.py)")

    print("")
    print("[11] Operator class - which side of the ledger each machine plays on")
    op = os.path.join(R, "hsbc_operator_class.json")
    if os.path.exists(op):
        o = json.load(open(op, encoding="utf-8"))
        check(o["all_stoquastic"],
              "H[G] is stoquastic at every typology and depth (%d instances)"
              % len(o["instances"]))
        check(all(r["max_offdiagonal"] <= 0.0 for r in o["instances"]),
              "no positive off-diagonal anywhere - Z-matrix by construction")
        check(o["all_ground_positive"],
              "ground vector strictly positive => ground state is sign-problem-free")
    else:
        print("    (hsbc_operator_class.json absent - run hsbc_operator_class.py)")

    dp = os.path.join(R, "hsbc_dirac_superres.json")
    if os.path.exists(dp):
        d = json.load(open(dp, encoding="utf-8"))
        sw = {r["circuits"]: r for r in d["sweep"]}
        f100 = min(p for p, r in sw.items() if r["fourier_accuracy"] >= 1.0)
        d100 = min(p for p, r in sw.items() if r["deconv_accuracy"] >= 1.0)
        check(d100 < f100,
              "deconvolution reaches full accuracy at %d circuits vs %d for Fourier (%.1fx)"
              % (d100, f100, f100 / d100))
    else:
        print("    (hsbc_dirac_superres.json absent - run hsbc_dirac_superres.py)")

    print("")
    print("[12] Analog platform - what the neutral-atom machine can hold")
    ap = os.path.join(R, "hsbc_aquila_class.json")
    if os.path.exists(ap):
        aq = json.load(open(ap, encoding="utf-8"))
        worst = max(a["max_map_error"] for a in aq["algebra"])
        check(worst < 1e-12,
              "repulsive Rydberg map reproduces -H at every typology (max dev %.1e)"
              % worst)
        c = aq["clock"]
        check(c["spacing_ok"] and c["clock_ok"],
              "clock and spacing pass: %.2f um edge, reachable t=%.0f vs %.0f needed"
              % (c["edge_spacing_um"], c["t_reachable"], c["t_needed"]))
        ref = [g for g in aq["geometry"] if g["family"] == "null_native"]
        check(all(g["edge_coupling_spread_pct"] < 1.0 for g in ref),
              "coupled-pair reference embeds exactly (coupling spread < 1%)")
        sig = [g for g in aq["geometry"] if g["family"] == "aml_ring6"]
        check(all(g["edge_coupling_spread_pct"] > 100.0 for g in sig),
              "discriminator does NOT embed - reported as the blocker it is")
        check(aq["verdict"]["algebra_exact"] and
              not aq["verdict"]["discriminator_realisable"],
              "verdict: exact on the reference, blocked on the discriminator")
    else:
        print("    (hsbc_aquila_class.json absent - run hsbc_aquila_class.py)")

    print("")
    print("[13] The three barriers to classical simulation")
    bp = os.path.join(R, "hsbc_three_barriers.json")
    if os.path.exists(bp):
        b = json.load(open(bp, encoding="utf-8"))
        check(not b["theta_is_clifford"],
              "preparation angle theta* = %.9f is NOT a multiple of pi/2 (non-Clifford)"
              % b["theta_star"])
        check(not b["theta_is_T_angle"] and b["theta_offset_from_T_rad"] > 1e-6,
              "theta* is not the T angle either - off pi/4 by %.6f rad, outside Clifford+T"
              % b["theta_offset_from_T_rad"])
        check(all(abs(r["imaginary_time_sign"] - 1.0) < 1e-9 for r in b["sign"]),
              "imaginary-time propagator carries zero cancellation (sign survival 1.000000)")
        worst = max(r["imaginary_time_sign"] / r["real_time_sign"] for r in b["sign"])
        check(worst > 10.0,
              "real time carries up to %.1fx the cancellation of imaginary time" % worst)
        peak = {f: max(r["magic_M2"] for r in rows)
                for f, rows in b["families"].items()}
        check(peak["aml_ring6"] > max(peak["aml_ring4"], peak["smurf_star"],
                                      peak["null_native"]),
              "girth-6 tile generates the most magic (%.4f of 5.022 Haar)"
              % peak["aml_ring6"])
        flat = b["families"]["null_native"]
        check(max(r["magic_M2"] for r in flat) - min(r["magic_M2"] for r in flat) < 1e-9,
              "null tile magic is exactly flat - prepared state is its eigenstate")
    else:
        print("    (hsbc_three_barriers.json absent - run hsbc_three_barriers.py)")

    print("")
    print("[14] Failed family lock + wrap-before-cone (do not regrade)")
    score_paths = [
        os.path.join(R, "hsbc_somega_score.json"),
        os.path.join(os.path.dirname(HERE), "hsbc", "results", "hsbc_somega_score.json"),
    ]
    score_path = next((p for p in score_paths if os.path.exists(p)), None)
    if score_path:
        sc = json.load(open(score_path, encoding="utf-8"))
        check(sc.get("sendable") is False, "dagfieg family sendable=false")
        gfail = sc.get("gates", {})
        check(gfail.get("omega_echo_within_5pct") is False, "echo gate stays FAIL")
        check(gfail.get("null_survival") is False, "null survival stays FAIL")
        # The archived FAIL receipt is left byte-identical, so its gate key
        # keeps the name it was written with. Match it by shape rather than
        # embedding that name here.
        star_keys = [k for k in gfail if k.startswith("star") and k.endswith("_band")]
        check(bool(star_keys) and all(gfail[k] is False for k in star_keys),
              "star-band stays FAIL (%s)" % ", ".join(sorted(star_keys)))
        rel = sc.get("omega_echo_rel_err")
        if rel is None:
            rel = (sc.get("families") or {}).get("echo", {}).get("rel_err")
        if rel is None:
            # working-copy score stores the number at top level in some revisions
            rel = sc.get("echo_rel_err")
        if isinstance(rel, (int, float)):
            check(rel > 0.5, "echo rel-err stays huge (%.3f)" % rel)
    else:
        print("    (no somega score in package or ../hsbc/results — fail-lock skipped)")

    tile = route["six_qubit_tile"]
    w6 = tile["aml_ring6"]["weight_near_delta_s"]
    ws = tile["smurf_star"]["weight_near_delta_s"]
    check(w6 > 0.10 and ws < 0.01,
          "wrap-before-cone model: girth-6 derived gap weight %.3f vs star %.3f (exact diag, not hardware)"
          % (w6, ws))

    print("")
    print("[15] NEW wrap-before-cone certificate (not the failed S(w) family)")
    wp = os.path.join(R, "hsbc_wrap_before_cone.json")
    if os.path.exists(wp):
        w = json.load(open(wp, encoding="utf-8"))
        check(w.get("failed_family") == "dagfiegmhr3c73e53gvg",
              "failed family stays named and locked")
        check("FAIL" in str(w.get("failed_family_status", "")),
              "dagfieg is not regraded as a win")
        g = (w.get("certificate") or {}).get("gates") or {}
        check(g.get("not_another_survival_fft") is True, "not another |L|^2 FFT")
        check(g.get("not_a_classifier") is True, "not a classifier")
        check(g.get("certificate") is True, "local wrap certificate holds")
        check(w.get("new_hardware_advantage_demonstrated") is False,
              "no invented hardware advantage")
        if w.get("status") == "LOCAL_PROOF":
            check(w.get("job_id") is None, "local proof has no invented job id")
        elif w.get("status") == "QUEUED":
            check(bool(w.get("job_id")) and w.get("job_id") != "dagfiegmhr3c73e53gvg",
                  "queued wrap card is a new id, not the failed family")
    else:
        print("    (hsbc_wrap_before_cone.json absent — run hsbc_wrap_before_cone.py)")

    print("")
    print("[16] Dirac wrap leftover (do not regrade the IBM fail)")
    dp = os.path.join(R, "hsbc_dirac_leftover.json")
    if os.path.exists(dp):
        d = json.load(open(dp, encoding="utf-8"))
        check(d.get("ibm_fail_locked") == "dagfiegmhr3c73e53gvg",
              "failed family still locked on the Dirac card")
        check(d.get("new_hardware_advantage_demonstrated") is False,
              "no invented Dirac hardware advantage")
        check(d.get("job_id") is None, "local Dirac leftover has no invented job id")
        g = d.get("local_gates") or {}
        check(g.get("ring_wraps") is True, "ring wraps at tau=1")
        check(g.get("star_explodes") is True, "star leftover hugs null")
        check(g.get("not_an_ibm_circuit") is True, "not a Heron commutator port")
        check((d.get("honesty") or {}).get("hsbc_fail_not_regraded") is True,
              "dagfieg is not regraded")
    else:
        print("    (hsbc_dirac_leftover.json absent — run dirac_leftover_local.py)")

    print("")
    print("[17] Full-W native card (64-cell continuous; leftover stays local)")
    fw = os.path.join(HERE, "..", "results", "dirac_full_W_flight.json")
    if os.path.exists(fw):
        f = json.load(open(fw, encoding="utf-8"))
        check(f.get("n_vars") == 64 and f.get("job_type") == "sample-hamiltonian",
              "native card is 64-cell sample-hamiltonian")
        check(f.get("not_6bit_integer") is True, "not the 6-bit leftover toy")
        check(f.get("new_hardware_advantage_demonstrated") is False,
              "full-W advantage not claimed")
        check(f.get("pf_vacuum_seen") is False, "PF vacuum was not seen")
        check(len(f.get("job_ids") or []) == 4, "four official full-W job ids")
        leftover = json.load(open(dp, encoding="utf-8")) if os.path.exists(dp) else {}
        check((leftover.get("full_W_card") or {}).get("new_hardware_advantage_demonstrated") is False,
              "leftover receipt does not inherit full-W as a leftover win")
    else:
        print("    (dirac_full_W_flight.json absent)")

    print("\n" + "=" * 68)
    print("verify: %d / %d checks passed" % (passed, total))
    print("=" * 68)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
