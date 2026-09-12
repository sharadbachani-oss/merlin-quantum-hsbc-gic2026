# -*- coding: utf-8 -*-
"""
hsbc_hardware_protocol.py — today's-hardware protocol for payment-graph S(ω).

Emits the circuits / analog schedules a device would run. Does not submit.
Does not invent counts. Expected signatures are taken from the model gate
(`hsbc_spectral_route.py`) and from first-principles closed forms.

IBM Heron (today):
  6-qubit exact first-principles prep (3 CX) on a native-pair tile, then first-order
  Trotter of the extra payment-graph ZZ terms that fit on heavy-hex, then
  inverse prep. Survival probability is the Loschmidt echo. FFT → S(ω).
  Tiled replicas (up to 26 tiles / 156 q) are independent payment windows.
  interference echo (Δs = 0.486254) is the in-job instrument check — a derived line
  the device must hit before any AML spectrum is graded.

QuEra Aquila (today):
  Exact pair → Rydberg map (V = 2κ E_u, Δ = κ E_u, Ω = 2 E_u) at 12
  pairs / 24 atoms / 4 cells — the FOV+C6 bound, not the 256-atom budget.
  Extra payment edges become local detuning, not extra packing. Analog
  quench inside the 4 µs window; atom-resolved populations → echo.

What is NOT claimed:
  Prior IBM 128q feature-map / generative jobs and Aquila 256-atom
  feature encodings are a different protocol (annex). They are not S(ω).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

SQRT5 = math.sqrt(5.0)
KAPPA = 3.0 / (3.0 - SQRT5)
DELTA = math.sqrt((KAPPA / 2.0) ** 2 + 4.0) - KAPPA / 2.0
CHI = math.atan(DELTA / 2.0)
THETA_STAR = math.pi / 2.0 - 2.0 * CHI
DELTA_S = 0.486254
PAIRS = ((0, 3), (1, 4), (2, 5))


def qasm_header(n: int) -> list[str]:
    return [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n}];",
        f"creg c[{n}];",
    ]


def exact_prep(lines: list[str], offset: int = 0) -> None:
    """Per-pair RY(θ*) · CX · H⊗H — fidelity 1 on the native operator."""
    for a, b in PAIRS:
        lines.append(f"ry({THETA_STAR:.16f}) q[{offset + a}];")
        lines.append(f"cx q[{offset + a}],q[{offset + b}];")
        lines.append(f"h q[{offset + a}];")
        lines.append(f"h q[{offset + b}];")


def exact_unprep(lines: list[str], offset: int = 0) -> None:
    """Inverse of exact_prep (H is involutive; CX then RY(−θ*))."""
    for a, b in reversed(PAIRS):
        lines.append(f"h q[{offset + a}];")
        lines.append(f"h q[{offset + b}];")
        lines.append(f"cx q[{offset + a}],q[{offset + b}];")
        lines.append(f"ry({-THETA_STAR:.16f}) q[{offset + a}];")


# Extra-edge typologies on the SAME 6 qubits (hardware tile).
# Native pairs are always (0,3),(1,4),(2,5). Extra edges are the quench.
FAMILY_EDGES = {
    "null": [],
    "ring4": [(0, 1), (1, 2), (2, 3)],
    "ring6": [(0, 1), (1, 2), (2, 4), (4, 5), (5, 3)],
    "star": [(0, 1), (0, 2), (0, 4)],
}

# Product-state interference echo probe (first-principles best_probe_product). Inverse is RY(−θ).
OMEGA_PROBE_RY = (
    0.5235987755982988,
    2.0943951023931953,
    2.0943951023931953,
    0.5235987755982988,
    1.5707963267948966,
    1.5707963267948966,
)

FROZEN_BANDS = {
    "echo_closed_form": DELTA_S,
    "echo_rel_tol": 0.05,
    "delta_closed_form": DELTA,
    "null_survival_min": 0.85,
    "ring6_delta_s_weight_min": 0.10,
    "star_delta_s_weight_max": 0.01,
    "delta_s_window": [DELTA_S - 0.12, DELTA_S + 0.12],
    "delta_window": [DELTA - 0.20, DELTA + 0.20],
    "shots": 8192,
    "times": [0.0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2],
}


def trotter_payment_edges(lines: list[str], extra_edges, dt: float, offset: int = 0) -> None:
    """One first-order Trotter slice of extra-edge κ D (published extra-edge slice)."""
    for a, b in extra_edges:
        lines.append(f"rzz({-KAPPA * dt:.16f}) q[{offset + a}],q[{offset + b}];")


def trotter_native_pairs(lines: list[str], dt: float, offset: int = 0) -> None:
    """RZZ(−κ dt) on the three native pairs — the D part of H_native."""
    for a, b in PAIRS:
        lines.append(f"rzz({-KAPPA * dt:.16f}) q[{offset + a}],q[{offset + b}];")


def trotter_drive(lines: list[str], dt: float, offset: int = 0) -> None:
    """RX(−2 dt) on every qubit — the −A = −Σ X term of H[G]."""
    for q in range(6):
        lines.append(f"rx({-2.0 * dt:.16f}) q[{offset + q}];")


def trotter_HG(lines: list[str], extra_edges, dt: float, offset: int = 0) -> None:
    """First-order Trotter of H[G] = κ D_G − A_G (native pairs + extra + drive)."""
    trotter_native_pairs(lines, dt, offset)
    trotter_payment_edges(lines, extra_edges, dt, offset)
    trotter_drive(lines, dt, offset)


def ibm_echo_family(times, extra_edges, steps_per_unit=2, evolve="extra"):
    """Loschmidt family: prep → Trotter(t) → unprep → measure.

    evolve:
      extra  — published extra-edge slice (representative QASM)
      HG     — full first-order H[G] (native RZZ + extra RZZ + RX)
    """
    circuits = []
    n_extra = len(list(extra_edges))
    for t in times:
        steps = max(1, int(round(steps_per_unit * t))) if t > 0 else 0
        dt = (t / steps) if steps else 0.0
        lines = qasm_header(6)
        exact_prep(lines)
        for _ in range(steps):
            if evolve == "HG":
                trotter_HG(lines, extra_edges, dt)
            else:
                trotter_payment_edges(lines, extra_edges, dt)
        exact_unprep(lines)
        for i in range(6):
            lines.append(f"measure q[{i}] -> c[{i}];")
        n_rzz = steps * (3 + n_extra) if evolve == "HG" else steps * n_extra
        circuits.append({
            "t": float(t),
            "steps": steps,
            "evolve": evolve,
            "n_2q_upper_bound": 3 + n_rzz + 3,
            "qasm": "\n".join(lines) + "\n",
        })
    return circuits


def echo_family(times, steps_per_unit=2):
    """In-job interference echo tile: product probe → native H[G] Trotter → inverse probe.

    This is the 6-qubit instrument check (peak within 5% of Δs = 0.486254).
    It is not the Walsh-compiled production qc.ibm.echo job
    d95c8revtlqs73fv8es0 (prior instrument lock, different protocol).
    """
    circuits = []
    for t in times:
        steps = max(1, int(round(steps_per_unit * t))) if t > 0 else 0
        dt = (t / steps) if steps else 0.0
        lines = qasm_header(6)
        for q, th in enumerate(OMEGA_PROBE_RY):
            lines.append(f"ry({th:.16f}) q[{q}];")
        for _ in range(steps):
            trotter_HG(lines, [], dt)
        for q, th in enumerate(OMEGA_PROBE_RY):
            lines.append(f"ry({-th:.16f}) q[{q}];")
        for i in range(6):
            lines.append(f"measure q[{i}] -> c[{i}];")
        circuits.append({
            "t": float(t),
            "steps": steps,
            "evolve": "HG_probe",
            "n_2q_upper_bound": steps * 3,
            "qasm": "\n".join(lines) + "\n",
        })
    return circuits


def heron_somega_family(times=None, evolve="HG"):
    """The sendability family: null + ring-4 + ring-6 + star + interference echo."""
    times = list(times if times is not None else FROZEN_BANDS["times"])
    out = {}
    for name, edges in FAMILY_EDGES.items():
        out[name] = ibm_echo_family(times, edges, evolve=evolve)
    out["echo"] = echo_family(times)
    names = []
    for fam, circs in out.items():
        for c in circs:
            names.append(f"{fam}_t{c['t']:.1f}")
    return {
        "families": out,
        "names": names,
        "times": times,
        "n_circuits": len(names),
        "shots": FROZEN_BANDS["shots"],
        "frozen_bands": FROZEN_BANDS,
    }


def echo_check():
    """In-job instrument: tiled interference echo must hit Δs = 0.486254 before grading."""
    return {
        "observable": "Delta_s",
        "closed_form": DELTA_S,
        "protocol": "qc.ibm.echo / tiled Heron echo (first-principles)",
        "pass_band": "peak within 5% of 0.486254 on a calibration tile",
        "role": "device is an instrument only if it first reads its own line",
        "prior_first-principles_job_not_this_family": (
            "ibm_kingston d95c8revtlqs73fv8es0 measured 0.4883 vs 0.486254 "
            "(0.43%). That is the first-principles Walsh interference echo lock, not the "
            "payment-graph S(ω) family."
        ),
    }


def aquila_card():
    return {
        "device": "QuEra Aquila (Braket analog)",
        "exact_map": {
            "pairs_per_shot": 12,
            "atoms": 24,
            "cells": 4,
            "V": "2 κ E_u",
            "Delta": "κ E_u",
            "rabi_Omega": "2 E_u",
            "map_error": "1e-9 (first-principles qc.aquila)",
            "crosstalk_bound": "9.96e-4",
            "fov_note": (
                "Binding constraint is 75×76 µm FOV + Rb C6, not the "
                "256-atom budget. Packing 252 atoms destroys the exact map."
            ),
        },
        "payment_encoding": (
            "Native pairs realize H = κ D − A. Extra payment-graph edges "
            "that do not fit the exact pair geometry are local detunings "
            "on the corresponding atoms (transaction shock), not extra "
            "packed atoms."
        ),
        "readout": "atom-resolved populations over a 4 µs analog window → L(t) → S(ω)",
        "expected_signature": {
            "native_cell": "pair gap δ = 0.839229 in the analog spectrum",
            "aml_ring_detuning": "excess low-ω weight vs null-detuning interleaves",
            "null_control": "zero extra detuning, same geometry, same shot budget",
        },
        "prior_256_atom_jobs": (
            "Eight earlier Aquila jobs (99.3% loading) encoded classical "
            "fraud features as detunings for an ML feature map. That is "
            "not this protocol and is not cited as S(ω) evidence."
        ),
    }


def expected_signatures():
    """Pulled from the model-gate JSON if present; otherwise closed forms."""
    path = os.path.join(OUT_DIR, "hsbc_spectral_route.json")
    if os.path.exists(path):
        g = json.load(open(path, encoding="utf-8"))
        adv = g.get("advantage", {})
        return {
            "source": "hsbc_spectral_route.json (model gate, exact diag, no shots)",
            "native_line_Hz_natural_units": DELTA,
            "aml4_peak_over_legit": adv.get("peak_height_ratio_aml4_over_legit"),
            "continuation_tv": adv.get("continuation_tv"),
            "continuation_G_residual": adv.get("continuation_G_residual"),
            "schmidt_chi_cycle_over_path": adv.get("schmidt_chi_cycle_over_path"),
            "hardware_claim": "none — signatures are the pass/fail bands for a future job",
        }
    return {
        "source": "closed forms only (run hsbc_spectral_route.py to attach model-gate bands)",
        "native_line": DELTA,
        "hardware_claim": "none",
    }


def main():
    # AML extra edges on a 6-qubit tile: the planted 4-cycle (0-1-2-3-0)
    # that is NOT a native pair. Native pairs are (0,3),(1,4),(2,5).
    extra_aml4 = FAMILY_EDGES["ring4"]
    times = list(FROZEN_BANDS["times"])
    family_aml = ibm_echo_family(times, extra_aml4, evolve="extra")
    family = heron_somega_family(times, evolve="HG")

    # published representative QASM stays the extra-edge slice at t=4.8
    qasm_path = os.path.join(HERE, "hsbc_ibm_somega_t4.8.qasm")
    with open(qasm_path, "w", encoding="utf-8") as f:
        f.write(family_aml[3]["qasm"])

    fam_dir = os.path.join(OUT_DIR, "somega_family")
    os.makedirs(fam_dir, exist_ok=True)
    for fam, circs in family["families"].items():
        for c in circs:
            fn = f"{fam}_t{c['t']:.1f}.qasm"
            with open(os.path.join(fam_dir, fn), "w", encoding="utf-8") as f:
                f.write(c["qasm"])

    payload = {
        "title": "HSBC hardware protocol — S(ω) of H[G] on today's devices",
        "status": "PROTOCOL + EXPECTED SIGNATURE. No shots in this file.",
        "ibm_heron": {
            "device_class": "Heron r2 (ibm_fez / ibm_kingston / ibm_marrakesh)",
            "tile": "6 qubits, native pairs on a heavy-hex path",
            "prep": "RY(theta*) CX H H per pair — 3 CX, fidelity 1",
            "observable": "Loschmidt survival after inverse prep; FFT → S(ω)",
            "in_job_controls": [
                "null extra-edge family (native operator only)",
                "replica pair on one tile",
                "interference echo calibration tile (Δs = 0.486254)",
            ],
            "flight_family": [
                "null", "ring4", "ring6", "star", "echo",
            ],
            "times": times,
            "n_circuits_per_family": len(times),
            "n_circuits_flight": family["n_circuits"],
            "shots_recommended": 8192,
            "aml4_extra_edges": [list(e) for e in extra_aml4],
            "family_extra_edges": {
                k: [list(e) for e in v] for k, v in FAMILY_EDGES.items()
            },
            "n_2q_at_t4.8_aml": family_aml[3]["n_2q_upper_bound"],
            "n_2q_at_t4.8_aml_HG": family["families"]["ring4"][3]["n_2q_upper_bound"],
            "echo_check": echo_check(),
            "frozen_bands": FROZEN_BANDS,
            "qasm_example": os.path.basename(qasm_path),
            "qasm_family_dir": "results/somega_family",
            "one_command": (
                "python hsbc_ibm_somega_flight.py   "
                "# retrieve-or-search; never submits"
            ),
            "matching_job": None,
        },
        "aquila": aquila_card(),
        "expected_signatures": expected_signatures(),
        "what_would_pass_a_hardware_gate": [
            "interference echo tile hits Δs = 0.486254 within the frozen band",
            "native-null family recovers the δ = 0.839229 line",
            "AML extra-edge family shows excess low-ω weight vs null, same shots",
            "scrambled-edge control does not",
            "girth-6 shows Δs weight the star does not (model-gate 18.7% vs ~0.02%)",
        ],
        "what_is_not_evidence": [
            "Dirac-3 fitted 3-body landscape (+5.8 pp AUPRC) — annex",
            "128-qubit feature map (+0.0023 AUPRC) — null, annex",
            "256-atom Aquila feature encoding — different protocol",
            "first-principles Walsh interference echo d95c8revtlqs73fv8es0 — instrument lock, not this family",
        ],
    }
    outp = os.path.join(OUT_DIR, "hsbc_protocol.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("wrote", outp)
    print("wrote", qasm_path)
    print("wrote", fam_dir, family["n_circuits"], "circuits")
    print("IBM AML extra-slice 2q@t=4.8:", family_aml[3]["n_2q_upper_bound"])
    print("IBM AML full-H 2q@t=4.8:", family["families"]["ring4"][3]["n_2q_upper_bound"])
    print("status: PROTOCOL ONLY — no shots")
    return payload


if __name__ == "__main__":
    main()
