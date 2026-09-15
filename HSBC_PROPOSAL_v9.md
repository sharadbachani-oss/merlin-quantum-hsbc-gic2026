# Quantum-enhanced credit-card fraud detection: the collective response of a coordinated entity-graph neighbourhood — mechanism measured at 14 qubits, advantage decided at 36–54 qubits on Braket

**Global Quantum + AI Challenge 2026 — HSBC Enterprise Challenge · Phase 1 Concept Proposal · Team Merlin Digital (GIC 2026 dual-track finalist — Mitsubishi/AIST materials track) · v10.0 · 2026-09-14**

**Public repository: https://github.com/sharadbachani-oss/merlin-quantum-hsbc-gic2026**

---

## 1. Problem framing

Card-not-present fraud that scales is a property of a group, not of a transaction: coordinated rings share cards, addresses, devices and e-mail domains and bunch in time and amount, and what a per-transaction scorer cannot represent is how such a group answers a probe **together**. We have measured that collective answer. In a 14-qubit exact test of the entity-graph response oracle (24 neighbourhoods of six nodes, neighbour–neighbour couplings switched on from none to all fifteen pairs; `mechanism_collective.json`), the neighbours' response to a probe on the target falls monotonically as they couple to each other (0.098 → 0.057 at t = 1.0), and it is not a sum of pairwise effects: the pairwise-additive prediction misses it by 72% of its magnitude at t = 1.0 and gets the sign wrong at t = 2.0 (−0.019 predicted against +0.069 measured). A ring screens its members collectively; no additive or per-transaction model carries that quantity.

Our tuned temporal baseline on IEEE-CIS (LightGBM, identity-merged; 472,432 train / 118,108 test) reports the statement's §4.1 metric set in full — AUC-ROC 0.9118, AUPRC 0.5532, and at the F1-optimal threshold 0.7244: F1 0.5462, precision 0.6118, recall 0.4934, confusion matrix TN 112,771 / FP 1,273 / FN 2,059 / TP 2,005 (`g1_baseline_merged.json`) — but inside the operating gray zone (2,362 transactions at 26% fraud density, the alerts an analyst actually reviews) its discrimination falls to AUC 0.61. The collective response is the signal left standing in that band.

**The wall is ours, measured.** A real-time, multi-node retarded response on a coupled arrangement is the one formulation here classical simulation cannot follow. The field cites that wall; we built the strongest classical attack we could and took it there ourselves. On our own 64-rung lattice our exact sparse-Pauli referee reproduces every flown cell out to k = 6 (five cells within 1.4σ on one damping constant, four within 0.6σ), and past k = 8 that same attack leaves the machine: 31 of 64 rung operators exceed 2×10⁸ Pauli terms, snake-MPS needs χ ≈ 2⁴⁶, PEPS diverges at grading accuracy (`A1_FINAL_VERDICT.md`, `k6_census.json`, `GATEB_PART2_TN_ESTIMATE.md`). Three independent representations of that adversary fail in the same window; the class loses controlled convergence at 50–100 qubits — a crossing point we can name because we drove our own attack into it.

**The crossing is costed.** A 20–30-node entity neighbourhood is 40–60 qubits at one qubit pair per node, past that wall and inside the Braket registers the statement itself names: 18 nodes / 36 qubits is IonQ Forte Enterprise's all-to-all register (35 algorithmic qubits) today, 20–27 nodes / 40–54 qubits is IQM Emerald, and only the 30-node / 60-qubit top of the range needs Ankaa-2's 84 qubits at lower fidelity or the next trapped-ion generation; TN1 (50 q) bounds the noiseless check. Preparation is exact and shallow, observables closed-form; the answer costs 12,000 circuits and 12 M shots. Phase 2 decides its value in the issuer's own unit — frauds caught at a fixed analyst alert budget — by the paired gray-zone label test on cohort G1 over five temporal folds: **pass is a paired AUPRC lift over the same tree given the oracle's raw inputs, 95% CI above zero, at fixed alert workload; below that bar we publish the null with its interval.** IonQ via Braket, IBM Heron replicating, protocol frozen before the run (§5).

**The route was selected by measurement, not by fashion.** The statement's secondary objective asks where quantum methods measurably improve on tuned classical baselines and under what conditions. We answer with twenty-one controlled formulations, every one carrying the same receipt class, negative or positive, registered in Appendix A. Twenty eliminations — each against its own exact classical floor — leave the relational route above standing: floor passed, mechanism measured, label-relevance at scale the one open question. Receipted eliminations are why we can name the one live route rather than guess at it.

The computation framework behind these receipts is a dual-track finalist in the Global Quantum + AI Challenge 2026 — Mitsubishi/AIST materials and QCi — so the method has already been assessed and advanced by an independent technical panel.

## 2. Technical approach

**Paradigm.** Hybrid classical–quantum. (a) Photonic entropy computing (QCi Dirac-3) sampling a degree-3 fraud-pattern landscape for minority-class augmentation; (b) gate-model real-time dynamics on trapped-ion hardware via Amazon Braket for the entity-graph response feature; (c) XGBoost / LightGBM scoring and SHAP attribution throughout.

**Execution route.** Phase-1 evidence was gathered on the allocations the programme already holds (IBM Heron `ibm_fez` / `ibm_kingston`, QCi Dirac-3). Phase-2 execution is on **Amazon Braket**, as the statement asks: development and noise prototyping on SV1 (34 q), TN1 (50 q) and DM1 (17 q, noise modelling); response circuits on IonQ Forte Enterprise (36 physical / 35 algorithmic qubits, all-to-all) and IQM Emerald (54 q); IonQ's built-in debiasing and sharpening as the error-mitigation layer; IBM Heron off-Braket for replication on identical circuits.

**Label-scarce augmentation (measured).** Eight most separating ULB features thresholded at fraud-side quantiles give 24 binary literals; a degree-3 log-odds landscape over them (24 one-body, 263 pairwise, 1,306 three-body terms) is fitted on the training-window fraud population and sampled on Dirac-3 (counts in §3). Each pattern is decoded onto the feature space by anchoring on the labelled frauds available to the classifier. The ladder in §4 varies the labelled frauds the classifier and decoder see (5, 10, 20, all), holds the generator fixed and disclosed as warm, and grades the device against eight control arms — including classical Gibbs of the identical degree-3 model and exact Boltzmann sampling of it by enumeration of all 16,777,216 states, the floor we built to beat our own device.

**Real-time entity-graph response (the admissible route).** Nodes are transactions; edges join transactions sharing card1, addr1 or P_emaildomain within a trailing window, built from training-period data only and, at test time, from strictly earlier transactions. Each gray-zone transaction's k-hop neighbourhood is encoded as a coupled arrangement carrying the transactions, not just its shape: one qubit pair per node, fields from standardised log-amount, target–neighbour couplings from shared-entity counts decayed by inter-arrival time. Product ground state, signed local probe on the target (two branches), Trotterised evolution, observables read at a fixed tick grid; the half-difference of the branches is the retarded response i<[B, A(t)]>/2 — a handful of numbers per transaction appended to the same classifier, with SHAP on them as the statement's feature-attribution output.

**Why the response should see fraud rings.** The collective screening measured in §1 is the mechanism, and its deviation from additivity grows with evolution time — where the floor locates the configuration information. Six nodes is calibration scale, classically reachable by construction (§4); the mechanism pays past it, at the PoC's scale.

**Hardware discipline.** Equal-structure parameterised templates (transpile once, rebind per neighbourhood); interleaved null references and replica pairs (replica agreement 0.0039 on our 128-qubit card); measurement-error mitigation and the platform's debiasing.

## 3. Feasibility and resource requirements

| resource | status |
|---|---|
| Data | IEEE-CIS (primary), ULB (secondary, label-scarce benchmark), Sparkov (stress test) — public, in hand with the temporal pipeline |
| Classical baselines | LightGBM identity-merged temporal split on IEEE-CIS (`g1_baseline_merged.json`); XGBoost on ULB |
| Augmentation ladder | `fewshot_ladder.py`, laptop, ≈2 h: 4 label counts × 9 arms × 40 seeds, exact 2²⁴ enumeration |
| Photonic sampler | QCi Dirac-3, unmetered allocation; 40 jobs already receipted |
| Response feature hardware | IonQ Forte Enterprise (36 q, all-to-all) and IQM Emerald (54 q) on Amazon Braket — 36–54 qubits, i.e. 18–27 nodes, the sizes those registers hold; depth ≤ 8 Trotter steps; 2,000 neighbourhoods × 2 probe signs × 3 times = 12,000 circuits, 12 M shots, before controls and replicas; SV1/TN1/DM1 for development and noise prototyping; IBM Heron for replication |
| Software | Braket SDK, Qiskit, XGBoost/LightGBM, SHAP; every number regenerates from archived counts |
| Compute | single workstation |

**Quantum execution sample counts (statement §4.2).** Every transaction subsample is drawn by stratified sampling preserving its source population's fraud/non-fraud ratio. **ibm_fez gray-zone feature-extraction card: 96 transactions**, stratified on the band's 26.0% fraud rate (`hsbc_ibm_band.py`: `n_f = int(round(N_TXN * yb.mean()))`; `g4_band_analysis.json`, n = 96), carried by 206 circuits × 8,192 shots. **ibm_fez training-free generative card: 524,288 device samples** from 65 circuits × 8,192 shots, 4,000 decoded rows appended (`qgen_result.json`). **QCi Dirac-3: 40 jobs × 20 requested → 559 returned solutions, 82 distinct 24-bit patterns** (`ulb_raw.json`); landscape fitted on the **394** training-window frauds of ULB's time-sorted 80/20 split, 2,000 decoded rows appended per arm, the 56,962-transaction test window untouched. **QuEra Aquila via Amazon Braket: 28 transactions across seven jobs** at 99% atom loading (`hsbc_aquila_class.json`) — the Braket execution to date. **Quantum-kernel race (Braket-validated simulator): 15,895 train / 8,000 test** (`hsbc_qkernel.json`). **Exact 14-qubit response-oracle label test: 10,362 neighbourhoods** — a stratified training subsample of 8,000 with the fraud ratio preserved plus the frozen 2,362-transaction gray zone — five temporal folds; at this size the oracle is simulated exactly on CPU, so it states stratification for the feature test but is not quantum execution (`hsbc_field_response_metrics.json`). **Phase-2 budget: 2,000 neighbourhoods**, stratified on the gray-zone fraud rate, subset size and stratification frozen before the run (§5).

Assumptions: shared-entity edges carry relational signal; a 20–30-node neighbourhood is sufficient; the degree-3 landscape is a warm generator and the ladder isolates what the classifier's labels contribute. Constraints: quantum execution is offline batch, as the statement scopes, and no device sits in the 100–300 ms authorisation path (§4 gives the cache key).

## 4. Expected impact

![Figure 1 — The tuned temporal baseline on IEEE-CIS: full-set precision–recall against the gray-zone band (left), and the score distributions with the band marked (right).](C:/quantum ai 2026/figs_v7/hsbc_grayzone.png)

**The conditions map, measured (Appendix A).** IEEE-CIS gray zone, 128-qubit feature-extraction card (§3 counts): quantum lift +0.0023 AUPRC, random-feature arm −0.028, AUC with the quantum block below classical-only. 128-qubit training-free generative card: identical to its shuffled control to four decimals. ULB full-label augmentation: device 0.808 AUPRC, below plain jitter 0.818. Graph-shape response oracle: withdrawn on its floor and null on labels (0.194 vs 0.198). Field-carrying response oracle at 14 qubits: +0.0009 AUPRC, 95% CI [−0.005, +0.007]. All six primary metrics, every arm, cohort G2, five folds, F1/precision/recall at a fixed top-10% alert budget (`hsbc_field_response_metrics.json`): response 0.093 AUPRC / 0.561 AUC / 0.122 F1 / 0.102 precision / 0.152 recall, inside its raw-input control (0.092 / 0.578 / 0.127 / 0.106 / 0.158) on every one. Quantum kernel race, bar locked before any quantum Gram was seen: 0.181 vs 0.204. **Reading:** at per-transaction scale with abundant labels the boosted tree is saturated — the statement's "under what conditions", answered with an interval on every cell.

**Label-scarce augmentation, reproduced against its exact floor (ULB, time-sorted 80/20 split, 40 seeds per cell, frauds caught at a fixed 75-alert budget on the 56,962-transaction test window; `fewshot_ladder.json`).**

| labelled frauds → | 5 | 10 | 20 |
|---|---|---|---|
| no augmentation | 0.602 ± 0.103 (45.7) | 0.677 ± 0.054 (50.5) | 0.730 ± 0.027 (53.5) |
| SMOTE | **0.604 ± 0.102** (46.4) | see receipt | **0.761 ± 0.025** (55.6) |
| jitter | 0.600 ± 0.150 (45.8) | see receipt | 0.739 ± 0.031 (54.0) |
| random oversampling | 0.528 ± 0.135 (41.3) | see receipt | 0.692 ± 0.059 (50.9) |
| pairwise Ising, Gibbs | 0.539 ± 0.112 (42.3) | = device | 0.678 ± 0.045 (50.7) |
| degree-3 model, classical Gibbs | 0.539 ± 0.112 (42.3) | = device | 0.681 ± 0.042 (51.0) |
| degree-3 model, exact 2²⁴ enumeration | 0.539 ± 0.112 (42.3) | = device | 0.679 ± 0.042 (51.3) |
| **degree-3 model, Dirac-3 samples (fair decode)** | 0.539 ± 0.112 (42.3) | −0.097 vs SMOTE, CI [−0.116, −0.077] | 0.679 ± 0.044 (51.1) |
| Dirac-3 samples, archived warm decode (394 anchors) | 0.707 ± 0.020 (52.0) | see receipt | 0.725 ± 0.022 (52.7) |
| paired, Dirac − exact enumeration | 0.000 (identical) | 0.000 (identical) | −0.0007, CI [−0.005, +0.004], p = 0.87 |
| paired, Dirac − SMOTE | −0.065, CI [−0.101, −0.030] | −0.097, CI [−0.116, −0.077] | −0.082, CI [−0.095, −0.070] |

AUPRC mean ± sd over 40 seeds; frauds caught at the 75-alert budget in parentheses; the full-label rung is the archived receipt (`results/ulb_result.json`: jitter 0.818, device 0.808).

**Against the statement's published ULB reference set (§4.1).** Our full-label ULB rung reports AUPRC 0.799 (none) to 0.818 (jitter), F1 0.833–0.849, AUC-ROC 0.9896 (`results/ulb_result.json`), against the statement's references: AUPRC 0.871 (RF+SMOTE) and 0.867 (XGBoost+SMOTE), F1 0.8636 (CatBoost) and 0.9407 (hybrid XRAI), AUC-ROC 0.9887 (stacking ensemble). **Comparison methodology:** those figures come from random or k-fold splits of the shuffled 284,807 rows, so train and test draw from the same two days; ours is a single time-sorted 80/20 split, the last 56,962 transactions held out, strictly later than anything the model or any resampler saw. The gap runs as the split predicts: AUPRC ~0.05–0.07 below the SMOTE references, AUC-ROC level with the stacking ensemble.

**Reading.** At five and ten labelled frauds the four pattern arms above are byte-identical, because a decoder anchored on ≤ 15 frauds averages all of them whatever the pattern says; that augmentation sits 0.065 and 0.097 below SMOTE, intervals far from zero. The archived few-shot result (0.607) reproduces only as the warm-decode arm, which maps every sample onto the 394 training-window frauds: the gain was label information entering through the decoder, not the sampler. At twenty labels the decoder can finally distinguish patterns and the verdict holds — paired device − exact enumeration −0.0007, CI [−0.005, +0.004], every fair arm 0.08 below SMOTE, the warm-decode arm now trailing no augmentation. The exact Boltzmann distribution places 24% of its mass on the 82 device patterns (top 8.2%): the photonic sampler is verified against exact classical truth and reads the landscape's mode faithfully. What the measurement retires is the landscape, not the device — and that is the statement's §4.2 question answered on this branch: we mapped the condition under which a photonic sampler helps minority-class augmentation here, bounded it with an interval at every label count tested, and closed the branch on our own evidence.

**Quantitative PoC targets.** Primary (real-time route): the crossing costed below. Primary (label-scarce): the §5 ladder rule, repeated on IEEE-CIS folds and cohort G1 with the sampler on Braket. Secondary (statement §4.1/§5.3): AUC-ROC, AUPRC, F1, precision, recall and confusion matrix at the operating threshold; recall at fixed alert budget; SHAP attribution; consistency across fraud types and temporal partitions; simulator-vs-hardware agreement; qubit count, depth, shots and inference latency.

### Quantum advantage — the wall measured, the instrument calibrated, the crossing costed

**The field's direction, and its wall.** HSBC's own 2025 result with IBM — projected quantum feature maps from a 109-qubit Heron consumed by XGBoost, up to 34% relative AUC gain on bond-RFQ fill prediction — is the workflow shape the statement invites; its authors state the mechanism is not understood, the gain is dataset-specific and decays with the train–test gap, and noise itself acted as the regulariser. On fraud data the controlled benchmark (arXiv:2608.15718) finds no robust quantum-kernel advantage, and the spectral analysis of quantum kernels (arXiv:2606.20402) explains why: a feature map whose operator-Schmidt spectrum decays fast is a small tensor network, and classical. Our twenty-one controlled formulations (Appendix A) are that wall measured on IEEE-CIS and ULB; the route below is what survived them.

**The instrument, calibrated.** Fourteen qubits is the scale at which exact classical truth exists to check the oracle against, and all three readings agree with it. The mechanism — collective screening measured exactly (§1). The floor — we set a quadratic adversary on our own feature set; it reproduced every graph-shape feature (R² 0.79–0.998), so that oracle went, and the field-carrying oracle beat the same adversary and isolates real configuration information at the longest evolution times. The labels — at six nodes the response sits inside its own raw-input control (+0.0009, CI [−0.005, +0.007]) and the cheap scalar reads the same neighbourhood better (AUC 1.00 vs 0.73): the wall's prediction for a classically reachable quantity, confirmed on our own instrument. An instrument that reads correctly wherever exact truth exists is the one to trust past k = 8, where no check exists. That agreement is what de-risks Phase 2, and nothing else in Appendix A carries it.

**The crossing, costed.** The real-time response is the class the statement names by inviting "quantum computation for a specific component: feature transformation, re-ranking", and the class our own referee loses past k = 8 (§1). One experiment decides it: gray-zone AUPRC, classical-only against classical+response, on 2,000 stratified neighbourhoods of 18–27 nodes at 36–54 qubits via Braket, 12,000 circuits and 12 M shots, paired over five temporal folds at fixed alert workload against the §5 bar.

**Scalability to industrial relevance.** Augmentation is training-time: one landscape fit per feature set, one sampler run per campaign. The response feature is per neighbourhood, cached on its complete defining state — neighbourhood, fields, couplings, window — or a validated approximation; entity identity alone is not a valid key, since amount and timing enter the operator. Device calls therefore grow with distinct gray-zone keys, not with transaction volume.

**Business value, in the issuer's units.** With the statement's own figure that each $1 of fraud costs the institution $4.41 all-in, every additional gray-zone fraud caught is worth 4.41 × its amount; the PoC reports the count and the issuer's volumes convert it. At five labels the fair augmentation arms catch 42–46 of 75 test-window frauds against SMOTE's 46.4, so the campaign value attaches to the response feature.

The physics is established, the instrument is calibrated against exact classical truth, and the crossing point is measured; the remaining variable is hardware access at the scale the crossing needs — 36–54 qubits on Braket — which is exactly what a Phase-2 PoC sprint supplies.

## 5. Validation plan

Declared before every run, both components, both outcomes committed. **Augmentation:** the ladder protocol is fixed in the script header (label counts, seeds, arms, decode anchoring on the classifier's labels only, metrics, paired bootstrap and Wilcoxon statistics, and the rule that a gain counts only where the device beats both the best classical resampling arm and exact sampling of the same model, CI above zero); Phase 2 repeats it on IEEE-CIS folds and cohort G1 with the sampler on Braket, adding simulator-vs-hardware distributions. **Response feature:** graph construction, neighbourhood cap, coupling map, probe, step grid, stratified subset size, folds, metrics and bar fixed before hardware; uncoupled, shuffled-neighbourhood, random-feature and shuffled-quantum-feature arms in every batch; replication on a second device; classical comparators on the same observable (exact light-cone evolution, tensor-network evolution at reported bond dimension, cheap relational features at equal budget) — the adversary that measured the wall, aimed at the crossing. Pass = the §4 bars met on held-out temporal folds with hardware noise and shot uncertainty propagated through the classifier; fail = the null published with its interval.

## 6. Hybrid / cross-domain integration

Quantum work sits in the training layer (augmentation) and the feature layer (response) of an unchanged issuer pipeline: entity graph → neighbourhood → response circuit (offline, batch) → features cached by neighbourhood state → LightGBM score → SHAP explanation → alert; landscape fit → sampler → decoded minority samples → retrain. Governance: features and samples are explicit numbers with a stated definition, attribution is standard SHAP, and the model-risk process governing the current ensemble governs the augmented one.

## 7. Team capability

Merlin Quantum is the quantum division of Merlin Digital (50+ technology FTE): Suhail Bachani (Founder & CEO, PI), Dr. Hiro Bachani PhD (Program Director), Rohit Bachani (co-founder), Mitul Sawlani and Mohamed Jafrun (engineering), Zeena Furtado (finance & operations), Roshan Bhairwani (financial services & deep tech, London), Dr. Ana Baroni MSc (domain specialist). Programme hardware record: 259 receipted QPU jobs, 12.3 million shots, the largest at 128 qubits and 2,720 two-qubit gates; equal-structure templating, null-differenced estimators, replica-pair statistics and the adversarial referee behind the §1 wall were all developed on this track.

## 8. Scope, with treatment

(i) Our own fair-decode control retires the archived few-shot gain (0.607 at five labels): the device samples equal exact classical sampling of the same model and trail SMOTE at every label count. (ii) A 24-variable sampler enumerates in four minutes on a laptop, so no quantum attribution is available there. (iii) At full labels plain jitter (0.818) beats the device samples (0.808). (iv) At 14 qubits the response sits inside its own raw-input control (+0.0009, CI through zero) — the calibration point, where a six-node quantity is classically reachable; the PoC tests the scale our own floor points to. (v) ULB is the statement's secondary dataset; IEEE-CIS transfer of the ladder is a PoC deliverable, not a result. (vi) Our IEEE-CIS baseline (AUC 0.912) sits below the competition winner (0.946), and our ULB numbers sit below the published ULB set on a harder temporal split (§4); lifts are measured only against our own tuned scorer.

---

### Appendix A — Hardware job register and receipts

| measurement | machine | job id | receipt |
|---|---|---|---|
| ULB degree-3 landscape sampling, 40 jobs, 82 distinct patterns | QCi Dirac-3 | `6a9679b408442f441bbb6afe` … `6a967f7708442f441bbb6b26` (40 ids in file) | `ulb_raw.json`, `ulb_model.json` |
| Few-shot augmentation ladder, 4 label counts × 9 arms × 40 seeds, exact 2²⁴ floor | CPU | — | `fewshot_ladder.py`, `fewshot_ladder.json` |
| Collective-mechanism test: 24 neighbourhoods × coupling ladder 0–15 pairs, pairwise-additive prediction, burst vs spread (exploratory) | CPU, 14-qubit exact | — | `mechanism_collective.py`, `mechanism_collective.json` |
| 128-qubit feature-extraction band card: 206 circuits × 8,192 shots, 2,720 two-qubit gates; four-arm analysis | ibm_fez | `daaucejvpcac73dd232g` | `g4_band_analysis.json`, `ibm_band_state.json` |
| 128-qubit training-free generative card: 65 circuits × 8,192 shots, 524,288 samples | ibm_fez | `dab6therrl7c7386flo0` | `qgen_result.json`, `qgen_state.json` |
| Data-encoded quantum Boltzmann machine: data, null control and replica | ibm_fez | `daba21jvpcac73ddh9h0` | `qbm_model.json`, `qbm_result.json` |
| Full-label ULB augmentation, six arms | CPU + Dirac-3 samples | as above | `results/ulb_result.json` |
| QuEra Aquila neutral-atom lane via Amazon Braket, 7 jobs, 28 transactions, 99% atom loading | QuEra Aquila (Braket) | — | `hsbc_aquila_class.json` |
| Quantum kernel race, bar locked before any quantum Gram | Braket-validated blockade simulator | — | `hsbc_qkernel.json`, `hsbc_classical.json` |
| Alert-policy optimisation gate, greedy = exact at equal catch | CPU, exact enumeration | — | `policy_gate.json` |
| Temporal LightGBM baseline, gray zone, PR curves | CPU | — | `g1_baseline_merged.json`, `baseline_scores_lgbm.npy`, `baseline_ytest.npy` |
| Graph-shape response oracle vs labels: 10,362 neighbourhoods, five arms, five temporal folds | CPU | — | `hsbc_response_test_workstation.json`, `floor_response_feature.json` |
| Field-carrying response oracle: floor (600 synthetic neighbourhoods) and label test (10,362 neighbourhoods, six arms, five folds) | CPU | — | `floor_field_response.json`, `hsbc_field_response.json`, `hsbc_field_response.py` |
| Classical-hardness boundary **measured** for the response class on the 64-rung lattice: referee exact to k = 6, dead past k = 8 | ibm_kingston + referee | `d9rdfb1dsedc73agh5ng` | `A1_FINAL_VERDICT.md`, `k6_census.json`, `GATEB_PART2_TN_ESTIMATE.md` |

### Appendix B — Why the response oracle is the admissible formulation, and why the augmentation result is stated against enumeration

Kernels, feature maps, training-free samplers, Boltzmann machines and alert-policy optimisation are static or shallow tasks whose outputs classical methods reproduce. A sampler over 24 binary variables is in that class by construction: its exact distribution enumerates 16.8 million states in four minutes on a laptop, which is why the ladder is graded against that enumeration and not a proxy. The response oracle is different in kind — a real-time, multi-node, deep quantity on a coupled arrangement whose sign structure no local gauge removes: the class our own referee holds exactly to k = 6 and loses past k = 8. The PoC tests whether that quantity carries fraud signal.

### Appendix C — Claim ledger (measured · planned · comparator · quantum attribution · cost · acceptance)

| claim | status | classical comparator | quantum attribution | total cost charged | acceptance threshold |
|---|---|---|---|---|---|
| Temporal LightGBM baseline AUC 0.912 / AUPRC 0.553; gray zone frozen | measured | IEEE-CIS winner 0.946 (different protocol) | none | CPU | — |
| Conditions map: 21 formulations run with control arms, no lift at abundant labels | measured | random / shuffled arms | none | 2 IBM jobs, 40+ Dirac jobs | reported as null |
| Label-scarce augmentation with device samples | measured, retired: device = exact enumeration = Gibbs at 5 and 10 labels; −0.065 / −0.097 AUPRC vs SMOTE (CI excludes 0) | SMOTE, jitter, random oversampling, exact 2²⁴ sampling, Gibbs | none | 2 h CPU, 40 Dirac jobs | reported as the negative it is |
| Field-carrying response feature | built; floor passed; calibrated at 14q against exact simulation, +0.0009 AUPRC vs raw-input control, CI [−0.005, +0.007] | raw inputs, relational, shuffled, random | exact oracle; IonQ on Braket in PoC | 2.9 h CPU | same paired lift at 36–54q, CI > 0 |
| Graph-shape-only response feature | withdrawn on its floor (R² 0.79–0.998); label test null | relational, shuffled, random | none | 2.2 h CPU | as the floor predicted |
