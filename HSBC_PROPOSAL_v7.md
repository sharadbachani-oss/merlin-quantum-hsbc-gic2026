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

Assumptions: shared-entity edges carry relational signal; a 20–30-node neighborhood is sufficient. Constraints: per-transaction quantum execution is offline (batch scoring evaluation, as the statement scopes); production latency would be met by caching neighborhood responses at the entity level. Cost of repeated feature acquisition is charged in the ledger.

## 4. Expected impact

**Prior work already receipted.** Tuned temporal baseline AUC 0.9118 / AUPRC 0.5532 (IEEE-CIS); gray zone [0.630, 0.819] frozen, 2,362 transactions, 26% fraud, in-band AUC 0.614. Hardware: 128-qubit feature-extraction card on IBM Heron (206 circuits × 8,192 shots, 2,720 two-qubit gates per circuit, replica agreement 0.0039, `daaucejvpcac73dd232g`); 128-qubit training-free generative card (524,288 samples, `dab6therrl7c7386flo0`); ULB generative augmentation on QCi Dirac-3 (+1.3% AUPRC over an Ising-pair sampler). These establish the instrument and the discipline; none of them is claimed as a detection gain.

**Quantitative PoC targets.** Primary: AUPRC inside the gray zone, classical-only vs classical+response, paired over 5 temporal folds; bar = a lift whose 95% CI excludes zero at fixed alert workload. Secondary (statement §4.1/§5.3): AUC-ROC, F1, precision/recall at the operating threshold on the full held-out set; recall at fixed alert budget; calibration; SHAP attribution of the response block; consistency across fraud types and temporal partitions; simulator-vs-hardware agreement; qubit count, depth, shot budget, and inference latency of the cached-feature path.

**What success means for HSBC.** At fixed analyst workload, more fraud caught in the band where the current model is blind, with a per-alert explanation that names the relational path — and a characterisation, as the statement asks, of *under what conditions* the quantum feature helps (neighborhood density, entity type, recency).

## 5. Validation plan

Pre-registered before hardware: graph construction, neighborhood cap, coupling map, probe, step grid, subset size and stratification, folds, metrics, and the bar. Controls in every batch: uncoupled circuits (closed-form response), shuffled-neighborhood circuits (structure destroyed, statistics preserved), random-feature arm and shuffled-quantum-feature arm in the classifier — our earlier band card used exactly these arms and they are what turned a +0.002 into a null rather than a claim. Replication: same templates on a second device (IBM Heron). Classical comparators on the same response observable: exact evolution inside the light-cone at ≤ 24 qubits, tensor-network evolution at reported bond dimension at matched accuracy, and inexpensive relational features (motif counts, diffusion, spectral, temporal-graph) given the same hyperparameter budget — the feature must beat cheap relational features, not just the per-transaction baseline. Success = the primary bar met on held-out temporal folds with hardware noise and shot uncertainty propagated through the classifier. If the response feature adds no value, that result is reported with its confidence interval and the conditions tested.

## 6. Hybrid / cross-domain integration

Quantum work sits in the feature layer of an otherwise unchanged issuer pipeline: entity graph → neighborhood → response circuit (offline, batch) → response features cached per entity → LightGBM score → SHAP explanation → alert. Nothing in the 100–300 ms authorisation path calls a quantum device. Governance: features are explicit numbers with a physical definition, attribution is standard SHAP, and the same model-risk process that governs the current ensemble governs the augmented one.

## 7. Team capability

Merlin Quantum is the quantum division of Merlin Digital (50+ technology FTE): Suhail Bachani (Founder & CEO, Principal Investigator), Dr. Hiro Bachani PhD (Program Director), Rohit Bachani (co-founder), Mitul Sawlani (engineering, Purdue), Mohamed Jafrun (engineering), Zeena Furtado (finance & operations), Roshan Bhairwani (financial services & deep tech, London; ex-Lehman Brothers, currently Morgan Stanley — his Merlin work is outside that role), Dr. Ana Baroni MSc (domain specialist). GIC 2026 dual-track finalist. Programme hardware record: 259 receipted QPU jobs, 12.3 million shots, the largest at 128 qubits and 2,720 two-qubit gates; equal-structure templating, null-differenced estimators and replica-pair statistics developed on this track; every negative result carries the same receipt class as a positive one.

---

### Appendix A — Receipts

| item | file / job |
|---|---|
| Temporal baseline and gray zone | `g1_baseline_merged.json` |
| Band re-ranking card, four arms | `g4_band_analysis.json`, IBM `daaucejvpcac73dd232g` |
| Generative card | `qgen_result.json`, IBM `dab6therrl7c7386flo0` |
| ULB Dirac augmentation | `ulb_model.json`, `ulb_raw.json` |
| Alert-policy gate (greedy = exact, submodular) | `policy_gate.json` |

### Appendix B — Why the earlier formulations were null, and why this one is different

Kernel and feature-map classifiers, training-free samplers, Boltzmann machines and alert-policy optimisation are static or shallow tasks: their outputs are ground-state or sampling properties that classical methods reproduce, which is what our four control arms measured. The response oracle is a real-time, multi-node, deep quantity on a coupled arrangement whose sign structure cannot be removed by any local gauge; that is the class in which classical simulation is known to lose controlled convergence at 50–100 qubits, and the class of our own 128-qubit collective-spectrum measurements. The PoC tests whether that quantity carries fraud signal; it does not assume it.
