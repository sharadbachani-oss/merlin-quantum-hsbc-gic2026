# Quantum-enhanced credit-card fraud detection: a real-time entity-graph response feature for the issuer's scoring model

**Global Quantum + AI Challenge 2026 — HSBC Enterprise Challenge · Phase 1 Concept Proposal · Team Merlin Digital (GIC 2026 dual-track finalist — Mitsubishi/AIST materials track) · v7.0 · 2026-09-12**

---

## 1. Problem framing

Card-not-present fraud scoring is a 100–300 ms decision on a transaction that is one node in a graph of shared cards, addresses, devices and e-mail domains. The classical state of the art — gradient-boosted ensembles on engineered features (IEEE-CIS winner AUC-ROC 0.9459) — is a per-transaction model: relational structure enters only through hand-built aggregates, which is the feature-engineering bottleneck the statement names. Our own tuned temporal-split baseline on IEEE-CIS (LightGBM, identity-merged) reaches AUC-ROC 0.9118 / AUPRC 0.5532, and inside the operating gray zone (2,362 transactions at 26% fraud density, the alerts an analyst actually reviews) the baseline's discrimination falls to AUC 0.61: the classical model has exhausted the per-transaction signal exactly where the money is.

What the gray zone needs is a **relational, dynamical** quantity: how a perturbation at this transaction propagates through its past-only entity neighborhood. That is a real-time many-body response — the class of computation in which every 2025–26 verified quantum-advantage candidate sits, and in which our programme already operates at 128 qubits. We propose that quantity as one feature block for the same classical scorer, evaluated on the statement's metrics and hardware route, with the gray-zone AUPRC as the pre-registered bar.

We state the boundary honestly: our nineteen earlier formulations — kernels, feature maps, training-free generative samplers, Boltzmann machines, alert-policy optimisation, band re-ranking — returned no measurable lift over the tuned baseline (best +0.0023 AUPRC on the 128-qubit feature-map card, within replica noise). Each was a static or sampling task, which the physics says classical methods handle. The proposal below is the formulation the physics admits.

## 2. Technical approach

**Paradigm.** Hybrid classical–quantum: gate-model real-time dynamics on trapped-ion hardware via Amazon Braket (IonQ Forte/Tempo, all-to-all connectivity, 2q fidelity ≥ 99.9%) for a reduced, stratified subset; LightGBM for scoring; SHAP for attribution.

**Entity graph, leakage-safe.** Nodes = transactions; edges join transactions sharing card1, addr1 or P_emaildomain within a trailing window, built from training-period data only and, at test time, from strictly earlier transactions (temporal split, no future edges). For each scored transaction the k-hop neighborhood (k = 2, capped at 20–30 nodes) is extracted.

**Quantum component — the response oracle.** Each neighborhood is encoded as a coupled arrangement: one qubit pair per transaction node, the pair's internal operator fixed, inter-node couplings declared from the shared-entity edges (weight by entity type and recency). Prepare the product ground state; apply a signed local probe on the target node (opposite ±π/2 rotations, two branches); evolve for a stated number of Trotter steps; read the target and neighbor observables. The half-difference of the two branches is the **retarded response** i⟨[B, A(t)]⟩/2 between the target and its neighborhood at each step — a handful of numbers per transaction, sampled without reconstructing a wavefunction. For the uncoupled control the response is known in closed form, which anchors every circuit.

**Feature and scoring.** The response vector (target self-response, neighbor responses at 2–3 time points, their signs) is appended to the baseline feature set; the same LightGBM, same hyperparameter budget, is retrained. Feature attribution is SHAP on the combined model, so the quantum block's contribution per prediction is reported directly (statement §4.2 explainability).

**Hardware discipline.** Equal-structure parameterised templates (transpile once, rebind angles per neighborhood), interleaved null references and replica pairs (replica agreement 0.0039 on our 128-qubit card), measurement-error mitigation and the platform's debiasing; simulator-vs-hardware comparison on the same circuits (SV1/DM1 as the statement suggests).

## 3. Feasibility and resource requirements

| resource | status |
|---|---|
| Data | IEEE-CIS (primary), ULB and Sparkov (secondary) — public, already in hand with the temporal baseline pipeline |
| Classical baseline | LightGBM identity-merged temporal split; gray-zone band frozen (`g1_baseline_merged.json`) |
| Quantum subset | stratified sample of the gray zone, fraud ratio preserved, size stated (target 2,000–4,000 neighborhoods) |
| Hardware | IonQ via Amazon Braket, 20–60 qubits per circuit, depth ≤ 8 Trotter steps; ~4,000 circuits × 1,000 shots; Braket simulators for development; IBM Heron as second device for replication |
| Software | Braket SDK, Qiskit for the template, LightGBM, SHAP; all analysis regenerates from archived counts |
| Compute | single workstation for graph extraction and training |

**Scope, stated with its treatment.** (i) No detection gain is claimed today — nineteen formulations were null under four-arm controls and are reported as such. (ii) The response oracle is untested on fraud labels — the PoC's single pre-registered bar decides it. (iii) Neighborhood responses are computed offline — production latency is met by entity-level caching, not in-path calls. (iv) Our baseline (AUC 0.912) sits below the IEEE-CIS winner (0.946) — the gray-zone lift is measured against our own tuned scorer and the cheap relational features. (v) IEEE-CIS entity keys are anonymised — the graph is built from card1/addr1/e-mail only, temporal split, no future edges.

Assumptions: shared-entity edges carry relational signal; a 20–30-node neighborhood is sufficient. Constraints: per-transaction quantum execution is offline (batch scoring evaluation, as the statement scopes); production latency would be met by caching neighborhood responses at the entity level. Cost of repeated feature acquisition is charged in the ledger.

## 4. Expected impact

![Figure 1 — The tuned temporal baseline on IEEE-CIS: full-set precision–recall against the gray-zone band (left), and the score distributions with the band marked (right) — the band is where analyst workload sits and where per-transaction signal is exhausted.](C:/quantum ai 2026/figs_v7/hsbc_grayzone.png)

**Prior work already receipted.** Tuned temporal baseline AUC 0.9118 / AUPRC 0.5532 (IEEE-CIS); gray zone [0.630, 0.819] frozen, 2,362 transactions, 26% fraud, in-band AUC 0.614. Hardware: 128-qubit feature-extraction card on IBM Heron (206 circuits × 8,192 shots, 2,720 two-qubit gates per circuit, replica agreement 0.0039, `daaucejvpcac73dd232g`); 128-qubit training-free generative card (524,288 samples, `dab6therrl7c7386flo0`); ULB generative augmentation on QCi Dirac-3 (+1.3% AUPRC over an Ising-pair sampler). These establish the instrument and the discipline; none of them is claimed as a detection gain.

![Figure 2 — The control discipline the PoC inherits: gray-zone AUPRC of the classical scorer alone and with the 128-qubit band feature, against random-feature and shuffled-feature arms; the quantum lift is +0.002 and is reported as null.](C:/quantum ai 2026/figs_v7/hsbc_arms.png)

**Quantitative PoC targets.** Primary: AUPRC inside the gray zone, classical-only vs classical+response, paired over 5 temporal folds; bar = a lift whose 95% CI excludes zero at fixed alert workload. Secondary (statement §4.1/§5.3): AUC-ROC, F1, precision/recall at the operating threshold on the full held-out set; recall at fixed alert budget; calibration; SHAP attribution of the response block; consistency across fraud types and temporal partitions; simulator-vs-hardware agreement; qubit count, depth, shot budget, and inference latency of the cached-feature path.

### Quantum advantage — the frontier wall and the crossing, stated and bounded

**The field's direction, and its wall.** HSBC's own 2025 result with IBM — projected quantum feature maps from a 109-qubit Heron, consumed by XGBoost in rolling backtests, up to 34% relative AUC gain on bond-RFQ fill prediction — is the workflow shape the statement invites, and its authors state the mechanism is not understood, the gain is dataset-specific and decays with the train–test gap, and hardware noise itself acted as the regulariser. On fraud data the controlled benchmark (arXiv:2608.15718) finds no robust quantum-kernel advantage (effects < 0.013 ARI, kernel concentration with qubit count, the one significant cell vanishing under search-budget ablation), and the spectral analysis of quantum kernels (arXiv:2606.20402) explains why: a feature map whose operator-Schmidt spectrum decays fast is a small tensor network and classical. Our nineteen nulls are that wall measured on IEEE-CIS and ULB with four control arms.

**The crossing.** Keep the HSBC–IBM shape — offline quantum transform, classical scorer, SHAP — but replace the feature map with the object that has a slowly-decaying spectrum by construction: the **retarded response of a coupled, past-only entity-graph neighborhood**, a real-time, multi-node, deep quantity with a sign structure no local gauge removes. It is the class in which classical simulation loses controlled convergence at 50–100 qubits and the class of our own 128-qubit collective measurements, and the four-arm control design is the search-budget ablation the benchmark literature demands. **Bounded:** no detection lift is claimed today; the pre-registered gray-zone AUPRC bar with confidence interval on temporal folds is the single test, simulator-vs-hardware is reported, and a null is published with its interval.

**What success means for HSBC.** At fixed analyst workload, more fraud caught in the band where the current model is blind, with a per-alert explanation that names the relational path — and a characterisation, as the statement asks, of *under what conditions* the quantum feature helps (neighborhood density, entity type, recency).

## 5. Validation plan

Pre-registered before hardware: graph construction, neighborhood cap, coupling map, probe, step grid, subset size and stratification, folds, metrics, and the bar. Controls in every batch: uncoupled circuits (closed-form response), shuffled-neighborhood circuits (structure destroyed, statistics preserved), random-feature arm and shuffled-quantum-feature arm in the classifier — our earlier band card used exactly these arms and they are what turned a +0.002 into a null rather than a claim. Replication: same templates on a second device (IBM Heron). Classical comparators on the same response observable: exact evolution inside the light-cone at ≤ 24 qubits, tensor-network evolution at reported bond dimension at matched accuracy, and inexpensive relational features (motif counts, diffusion, spectral, temporal-graph) given the same hyperparameter budget — the feature must beat cheap relational features, not just the per-transaction baseline. Success = the primary bar met on held-out temporal folds with hardware noise and shot uncertainty propagated through the classifier. If the response feature adds no value, that result is reported with its confidence interval and the conditions tested.

## 6. Hybrid / cross-domain integration

Quantum work sits in the feature layer of an otherwise unchanged issuer pipeline: entity graph → neighborhood → response circuit (offline, batch) → response features cached per entity → LightGBM score → SHAP explanation → alert. Nothing in the 100–300 ms authorisation path calls a quantum device. Governance: features are explicit numbers with a physical definition, attribution is standard SHAP, and the same model-risk process that governs the current ensemble governs the augmented one.

## 7. Team capability

Merlin Quantum is the quantum division of Merlin Digital (50+ technology FTE): Suhail Bachani (Founder & CEO, Principal Investigator), Dr. Hiro Bachani PhD (Program Director), Rohit Bachani (co-founder), Mitul Sawlani (engineering, Purdue), Mohamed Jafrun (engineering), Zeena Furtado (finance & operations), Roshan Bhairwani (financial services & deep tech, London; ex-Lehman Brothers, currently Morgan Stanley — his Merlin work is outside that role), Dr. Ana Baroni MSc (domain specialist). GIC 2026 dual-track finalist. Programme hardware record: 259 receipted QPU jobs, 12.3 million shots, the largest at 128 qubits and 2,720 two-qubit gates; equal-structure templating, null-differenced estimators and replica-pair statistics developed on this track; every negative result carries the same receipt class as a positive one.

---

### Appendix A — Hardware job register and receipts

| measurement | machine | job id | receipt |
|---|---|---|---|
| 128-qubit feature-extraction band card: 206 circuits × 8,192 shots, 2,720 two-qubit gates; four-arm analysis | ibm_fez | `daaucejvpcac73dd232g` | `g4_band_analysis.json`, `ibm_band_state.json` |
| 128-qubit training-free generative card: 65 circuits × 8,192 shots, 524,288 samples | ibm_fez | `dab6therrl7c7386flo0` | `qgen_result.json`, `qgen_state.json` |
| ULB generative augmentation, Dirac degree-3 vs Ising pair vs SMOTE/jitter (40 jobs) | QCi Dirac-3 | `6a9679b408442f441bbb6afe`, `6a967a3a08442f441bbb6b00`, `6a967c5908442f441bbb6b01`, `6a967c6e08442f441bbb6b02`, `6a967c8a08442f441bbb6b03`, `6a967c9f08442f441bbb6b04`, `6a967cb408442f441bbb6b05`, `6a967cc908442f441bbb6b06`, `6a967cdf08442f441bbb6b07`, `6a967cf408442f441bbb6b08`, `6a967d0908442f441bbb6b09`, `6a967d1e08442f441bbb6b0a`, `6a967d3308442f441bbb6b0b`, `6a967d4908442f441bbb6b0c`, `6a967d5e08442f441bbb6b0d`, `6a967d7308442f441bbb6b0e`, `6a967d8808442f441bbb6b0f`, `6a967d9e08442f441bbb6b10`, `6a967db908442f441bbb6b11`, `6a967dce08442f441bbb6b12`, `6a967de408442f441bbb6b13`, `6a967df908442f441bbb6b14`, `6a967e0e08442f441bbb6b15`, `6a967e2308442f441bbb6b16`, `6a967e3808442f441bbb6b17`, `6a967e4e08442f441bbb6b18`, `6a967e6308442f441bbb6b19`, `6a967e7808442f441bbb6b1a`, `6a967e8d08442f441bbb6b1b`, `6a967ea308442f441bbb6b1c`, `6a967eb808442f441bbb6b1d`, `6a967ecd08442f441bbb6b1e`, `6a967ee208442f441bbb6b1f`, `6a967ef808442f441bbb6b20`, `6a967f0d08442f441bbb6b21`, `6a967f2208442f441bbb6b22`, `6a967f3708442f441bbb6b23`, `6a967f4d08442f441bbb6b24`, `6a967f6208442f441bbb6b25`, `6a967f7708442f441bbb6b26` | `ulb_raw.json`, `ulb_model.json` |
| 400-job diversity test: 344 jobs → 123 distinct samples (saturation) | QCi Dirac-3 | ids in file | `qgen_ulb400` receipts |
| Aquila encoding rehearsal, 12 atoms, R1–R4 | Braket local AHS simulator | — | `g2_rehearsal.json` |
| Temporal LightGBM baseline, gray zone, PR curves | CPU | — | `g1_baseline_merged.json`, `baseline_scores_lgbm.npy`, `baseline_ytest.npy` |
| Alert-policy optimisation gate (greedy = exact) | CPU | — | `policy_gate.json` |

### Appendix B — Why the earlier formulations were null, and why this one is different

Kernel and feature-map classifiers, training-free samplers, Boltzmann machines and alert-policy optimisation are static or shallow tasks: their outputs are ground-state or sampling properties that classical methods reproduce, which is what our four control arms measured. The response oracle is a real-time, multi-node, deep quantity on a coupled arrangement whose sign structure cannot be removed by any local gauge; that is the class in which classical simulation is known to lose controlled convergence at 50–100 qubits, and the class of our own 128-qubit collective-spectrum measurements. The PoC tests whether that quantity carries fraud signal; it does not assume it.
