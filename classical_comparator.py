"""Classical comparator arm for the payment-network retarded response.

WHAT THIS IS. The other side of the advantage comparison, on the same operator,
the same observable and a matched accuracy target. The classical method is the
matrix-product route, the strongest available for a real-time response. The
accuracy target is 99.9% state fidelity, with bond dimension taken as the
maximum over all cuts, and two orderings are searched with the better one
reported so the comparison uses the classical method's best case.

THE INDUSTRIAL AXIS. Layering depth k is the number of laundering rings routed
through one shared placement tier. It is the launderer's cheapest control and it
is the axis on which motif mining degrades, because each added layer multiplies
the benign look-alikes.

WHY THIS OPERATOR MAY BE HARD CLASSICALLY, structurally and independently.
Its ground state is stoquastic, so imaginary-time sampling of the ground state
carries no sign problem. The retarded response is real-time, where the
cancellation returns. Separately, the derived pairing couples site i directly
to i + 3k, which is the long-range structure that makes matrix-product
orderings poor.

SCOPE. A measured cost for this instance family, observable, accuracy target and
the orderings searched. A credible classical baseline and a censored benchmark
observation, not a complexity proof. A truncation test is specific to its state,
ordering and tolerance.

QUANTUM SIDE. The exact compiler costs 8 logical CX per evolved pair, constant
in evolution time. Physical routing is a separate device-specific cost and is
not included.
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    "lw", os.path.join(HERE, "hsbc_layering_wall.py"))
lw = importlib.util.module_from_spec(_s)
_s.loader.exec_module(lw)

FIDELITY = 0.999
FAMILY = "aml_ring6"          # the signal-carrying typology


def orderings(n, k):
    """Orderings a matrix-product method would plausibly choose."""
    seq = list(range(n))
    pair_adj = []
    for i in range(3 * k):
        pair_adj += [i, i + 3 * k]
    layer = []
    for l in range(k):
        layer += [l + j * k for j in range(6)]
    return {"sequence": seq, "pair_adjacent": pair_adj, "layer_grouped": layer}


def min_bond_dimension(psi, n, order, fidelity=FIDELITY):
    v = psi.reshape([2] * n)
    v = np.transpose(v, [n - 1 - q for q in order])
    need = 1
    for cut in range(1, n):
        m = v.reshape(1 << cut, -1)
        s = np.linalg.svd(m, compute_uv=False)
        p = s ** 2
        p = p / p.sum()
        keep = int(np.searchsorted(np.cumsum(p), fidelity) + 1)
        need = max(need, min(keep, p.size))
    return need


def run(ks=(1, 2, 3), T=6.0, npts=9):
    rows = []
    for k in ks:
        n, nat, ex, pl = lw.layered(FAMILY, k)
        edges = nat + ex + pl
        d_all = lw.diag_D(n, edges)
        _, g = lw.ground_state(n, nat)
        times = np.linspace(0.0, T, npts)
        dt = times[1] - times[0]
        psi = g.astype(np.complex128)
        best = {key: 1 for key in orderings(n, k)}
        for _ in range(npts - 1):
            psi = lw.lanczos_step(psi, n, d_all, dt)
            for key, o in orderings(n, k).items():
                best[key] = max(best[key], min_bond_dimension(psi, n, o))
        cap = 1 << (n // 2)
        chi_star = min(best.values())
        n_pairs = 3 * k
        n_evolved = len(ex) + len(pl)
        quantum_cx = 8 * n_evolved + n_pairs          # 8 CX per evolved pair + prep
        rows.append(dict(k=k, n_qubits=n, edges=len(edges), chi_cap=cap,
                         chi_by_ordering=best, chi_best_ordering=chi_star,
                         classical_cost_proxy=float(chi_star ** 2 * n),
                         quantum_logical_cx=quantum_cx))
        print("  k=%d n=%2d edges=%2d  chi=%s  best=%d of cap %d   "
              "chi^2*n=%.3e   quantum CX=%d"
              % (k, n, len(edges), best, chi_star, cap, chi_star ** 2 * n, quantum_cx),
              flush=True)
    return rows


if __name__ == "__main__":
    print("classical comparator: bond dimension for the payment-network response")
    print("  typology %s, fidelity target %.3f, three orderings searched"
          % (FAMILY, FIDELITY))
    rows = run()
    chis = [r["chi_best_ordering"] for r in rows]
    caps = [r["chi_cap"] for r in rows]
    cg = [chis[i] / chis[i - 1] for i in range(1, len(chis))]
    capg = [caps[i] / caps[i - 1] for i in range(1, len(caps))]
    qg = [rows[i]["quantum_logical_cx"] / rows[i - 1]["quantum_logical_cx"]
          for i in range(1, len(rows))]
    print("\n  chi growth per added layer:      %s  (cap grows %s)"
          % (["%.2fx" % x for x in cg], ["%.0fx" % x for x in capg]))
    print("  quantum logical CX growth:       %s" % (["%.2fx" % x for x in qg]))
    out = dict(card="classical comparator arm, payment-network retarded response",
               family=FAMILY, fidelity_target=FIDELITY, rows=rows,
               chi_growth_per_layer=cg, cap_growth=capg, quantum_cx_growth=qg,
               scope="measured cost for this instance family, observable, accuracy "
                     "target and orderings searched; credible classical baseline "
                     "and censored benchmark observation, not a complexity proof",
               quantum_cost_rule="8 logical CX per evolved pair plus one CX per "
                                 "pair for preparation, constant in evolution "
                                 "time; physical routing separate")
    json.dump(out, open(os.path.join(HERE, "results", "classical_comparator.json"), "w"),
              indent=1)
    print("  wrote results/classical_comparator.json")
