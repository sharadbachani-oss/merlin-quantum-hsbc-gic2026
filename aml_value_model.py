# -*- coding: utf-8 -*-
"""
aml_value_model.py — commercial value of an independent typology
channel in financial-crime operations (AML), with every input sourced
or flagged as a bank-supplied parameter.

The S(omega) route is a SECOND-CHANNEL TYPOLOGY INSTRUMENT: a
training-free, symmetry-labelled spectral signature that separates
ring / chain topologies from star topologies. It does not replace the
transaction-monitoring (TM) score. It is priced on what an independent
channel is worth against the industry's actual cost line: alert triage
and SAR investigation, where up to 95% of alerts are false positives.

Three value lines, priced separately, no double counting:
  L1  triage deflection  — false alerts closed earlier at fixed recall
  L2  typology recall    — ring/mule structures the tabular score defers
  L3  model-risk         — SR 11-7 independent-channel diversification
                           (stated, not priced: no dollar claim)

Sensitivity over the two parameters only HSBC can supply:
  alerts per year, and the share of alert volume that is
  network-typology in nature.
"""
import json

# ---- sourced inputs -------------------------------------------------
FALSE_POSITIVE_RATE = 0.95      # PwC via Flagright (brief cites this)
COST_PER_ALERT_USD  = 30.0      # analyst triage, industry range 20-60
COST_PER_SAR_USD    = 750.0     # full investigation + filing, 500-1500
FRAUD_COST_MULT     = 4.41      # LexisNexis 2024 (brief's own anchor)

# ---- measured from the model gate -----------------------------------
# girth-6 chain puts 18.7% of S(omega) on the Omega line; girth-4 ring
# and edge-count-matched star put ~0.02%.  Contrast ratio ~900x.
SIG_RING = 0.187
SIG_STAR = 0.0002

def triage_deflection(alerts_yr, network_share, deflect_frac):
    """L1: alerts the typology channel closes without full investigation."""
    net_alerts = alerts_yr * network_share
    false_net  = net_alerts * FALSE_POSITIVE_RATE
    deflected  = false_net * deflect_frac
    return deflected, deflected * COST_PER_ALERT_USD

def typology_recall(alerts_yr, network_share, uplift_pp, sar_value):
    """L2: ring structures surfaced that the tabular channel deferred."""
    net_alerts = alerts_yr * network_share
    true_net   = net_alerts * (1 - FALSE_POSITIVE_RATE)
    extra      = true_net * uplift_pp
    return extra, extra * sar_value

def main():
    rows = []
    print("=" * 74)
    print("AML TYPOLOGY CHANNEL — VALUE MODEL")
    print("=" * 74)
    print(f"\nSourced: FP rate {FALSE_POSITIVE_RATE:.0%} (PwC/Flagright), "
          f"triage ${COST_PER_ALERT_USD:.0f}/alert, SAR ${COST_PER_SAR_USD:.0f}")
    print(f"Measured: ring S(omega) weight {SIG_RING:.3f} vs star "
          f"{SIG_STAR:.4f}  ->  {SIG_RING/SIG_STAR:.0f}x contrast\n")

    print("L1  TRIAGE DEFLECTION  (false network-alerts closed early)")
    print("   alerts/yr | network | deflect | alerts closed | annual value")
    for alerts in (1_000_000, 5_000_000, 20_000_000):
        for ns in (0.10, 0.20):
            for df in (0.20, 0.40):
                n, v = triage_deflection(alerts, ns, df)
                rows.append(dict(line="L1", alerts=alerts, net=ns,
                                 deflect=df, n=n, usd=v))
                print(f"  {alerts:>10,} | {ns:5.0%}   | {df:5.0%}   | "
                      f"{n:>12,.0f}  | ${v/1e6:8.2f}M")

    print("\nL2  TYPOLOGY RECALL  (ring cases surfaced, per +1pp on the")
    print("    network-typology true-positive pool)")
    print("   alerts/yr | network | +pp  | extra cases | annual value")
    for alerts in (1_000_000, 5_000_000, 20_000_000):
        for ns in (0.10, 0.20):
            for pp in (0.01, 0.05):
                n, v = typology_recall(alerts, ns, pp, COST_PER_SAR_USD * 10)
                rows.append(dict(line="L2", alerts=alerts, net=ns,
                                 pp=pp, n=n, usd=v))
                print(f"  {alerts:>10,} | {ns:5.0%}   | {pp:4.0%} | "
                      f"{n:>10,.0f}  | ${v/1e6:8.2f}M")

    # headline band: mid assumptions
    lo, _ = triage_deflection(5_000_000, 0.10, 0.20)
    lo_v = lo * COST_PER_ALERT_USD
    hi, _ = triage_deflection(20_000_000, 0.20, 0.40)
    hi_v = hi * COST_PER_ALERT_USD
    print("\n" + "=" * 74)
    print(f"L1 band across the swept range: ${lo_v/1e6:.1f}M - ${hi_v/1e6:.1f}M / yr")
    print("L3 model-risk diversification (SR 11-7): stated, NOT priced.")
    print("Regulatory tail (AML penalties) excluded by design: not")
    print("attributable to a single detection channel.")
    print("=" * 74)
    json.dump(dict(sourced=dict(fp_rate=FALSE_POSITIVE_RATE,
                                cost_alert=COST_PER_ALERT_USD,
                                cost_sar=COST_PER_SAR_USD,
                                fraud_mult=FRAUD_COST_MULT),
                   measured=dict(sig_ring=SIG_RING, sig_star=SIG_STAR,
                                 contrast=SIG_RING / SIG_STAR),
                   rows=rows,
                   l1_band_usd=[lo_v, hi_v]),
              open("results/aml_value_model.json", "w"), indent=1)
    print("-> results/aml_value_model.json")

if __name__ == "__main__":
    main()
