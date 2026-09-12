# -*- coding: utf-8 -*-
"""Does the readout survive layering?  Global echo vs local pair response.

The global echo <g|e^{-iHt}|g> suffers the orthogonality catastrophe: the
prepared state's overlap with any single eigenvector of the k-layer
payment graph falls off with k, so the leading line carries less and less
weight and the shot cost explodes.  That is a real limit and it is the
one that decides whether this route scales.

A LOCAL observable does not have that problem.  We measure one layer's
coupled pair inside the full k-layer network:

    C(t) = <g| Z_a Z_b (t) |g>   on the layer-0 pair (0, 3k)

Its weight is set by one pair, not by the whole register, while its
spectrum still reflects the whole network inside the lightcone.  This
script measures the leading line and, decisively, ITS WEIGHT, for both
readouts as a function of layering depth.
"""
import json, os, sys, time
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("lw", os.path.join(HERE, "hsbc_layering_wall.py"))
lw = importlib.util.module_from_spec(_s); _s.loader.exec_module(lw)

def run(name, k, T=32.0, dt=0.2):
    n, nat, ex, pl = lw.layered(name, k)
    d_all = lw.diag_D(n, nat + ex + pl)
    _, g = lw.ground_state(n, nat)
    a, b = 0, 3 * k                                   # layer-0 coupled pair
    idx = np.arange(1 << n, dtype=np.int64)
    zz = (1 - 2 * ((idx >> a) & 1)) * (1 - 2 * ((idx >> b) & 1))
    steps = int(round(T / dt))
    psi = g.copy()
    Lg, Cl = [complex(np.vdot(g, psi))], [float(np.vdot(psi, zz * psi).real)]
    for _ in range(steps):
        psi = lw.lanczos_step(psi, n, d_all, dt)
        Lg.append(complex(np.vdot(g, psi)))
        Cl.append(float(np.vdot(psi, zz * psi).real))
    ts = np.arange(len(Lg)) * dt

    def lead(sig, complex_sig):
        x = np.asarray(sig, dtype=np.complex128)
        if not complex_sig:
            x = x - x.mean()                          # connected part
        y = np.fft.fftshift(np.fft.fft(x * np.hanning(len(x)), n=8 * len(x)))
        om = np.fft.fftshift(np.fft.fftfreq(len(y), d=dt)) * 2 * np.pi
        S = np.abs(y) ** 2; S = S / S.sum()
        m = om > 0.05
        o, s = om[m], S[m]
        i = int(np.argmax(s))
        return float(o[i]), float(s[i])

    og, wg = lead(Lg, True)
    ol, wl = lead(Cl, False)
    return {"family": name, "k": k, "n_qubits": n,
            "global_lead_omega": og, "global_lead_weight": wg,
            "local_lead_omega": ol, "local_lead_weight": wl,
            "local_signal_amplitude": float(np.ptp(Cl))}

if __name__ == "__main__":
    fam = sys.argv[1] if len(sys.argv) > 1 else "aml_ring6"
    ks = [int(x) for x in sys.argv[2:]] or [1, 2, 3]
    rows = []
    for k in ks:
        t0 = time.time(); r = run(fam, k); r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"  {fam} k={k} n={r['n_qubits']:2d} | GLOBAL w1={r['global_lead_omega']:.4f} "
              f"weight={r['global_lead_weight']:.5f} | LOCAL w1={r['local_lead_omega']:.4f} "
              f"weight={r['local_lead_weight']:.5f} amp={r['local_signal_amplitude']:.4f}"
              f"  ({r['seconds']}s)", flush=True)
    json.dump(rows, open(os.path.join(HERE, "results", f"local_{fam}.json"), "w"), indent=1)
