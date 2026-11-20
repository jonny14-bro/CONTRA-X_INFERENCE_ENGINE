# EAHMMF — Complete Research Project Progress Report

**Project:** Expert-Aware Hierarchical Memory Management Framework
**Research component:** PWEA — Per-Prompt Workload Expert Activation
**Report date:** 15 November 2026
**Current phase:** Telemetry collection, expert profiling and cross-category evaluation completed; hierarchical memory management implementation remains ahead.

---

# 1. Executive Summary

EAHMMF is a research project focused on improving Mixture-of-Experts (MoE) inference through expert-aware hierarchical memory management.

The central idea is to predict which experts are likely to be required during inference and distribute them across three memory tiers:

- **Hot:** GPU VRAM — experts expected to be needed frequently or soon.
- **Warm:** System RAM — experts with moderate expected demand.
- **Cold:** SSD — experts with lower expected demand.

The long-term objective is to reduce unnecessary expert transfers, make better use of limited GPU memory, and improve inference efficiency.

The project has progressed beyond its initial conceptual stage. A substantial telemetry and evaluation foundation has been developed.

The completed experimental work includes:

1. Establishing a fixed MoE model and experimental environment.
2. Developing and using an expert activation telemetry pipeline.
3. Collecting a five-category dataset containing 25 sessions and 175 prompts.
4. Recording 28,428,288 activation events.
5. Building prompt-level and session-level expert profiles.
6. Performing five-fold Leave-One-Category-Out (LOCO) evaluation.
7. Discovering and correcting an expert-ID alignment defect in the evaluator.
8. Running corrected deterministic evaluations.
9. Implementing supplementary tie-aware ranking metrics.
10. Verifying evaluation artifacts and recording automated test results.

The corrected results show that PWEA achieves higher overall NDCG@5 than the aggregate-frequency baseline at both prompt and session levels.

However, the current work does not yet demonstrate actual VRAM/RAM/SSD placement, live expert migration, or end-to-end inference latency improvements.

The next major milestone is to build and validate a leakage-safe future expert-demand predictor, followed by a memory-tier simulator.

---

# 2. Project Objective

The primary objective is:

**To develop a predictive, expert-aware memory management framework for MoE models that dynamically places experts across VRAM, RAM and SSD according to expected inference demand.**

The intended processing pipeline is:

1. Receive an inference workload.
2. Characterize the workload.
3. Predict expert demand.
4. Calculate expert priority.
5. Assign experts to Hot, Warm and Cold tiers.
6. Allocate or retain experts in the corresponding memory tiers.
7. Monitor actual expert usage during inference.
8. Compare predicted demand with observed demand.
9. Migrate experts when required.
10. Evaluate prediction accuracy, memory efficiency and inference performance.

The current PWEA research establishes the expert activation measurement and ranking-evaluation foundation for this architecture.

---

# 3. Current Experimental Environment

## Model configuration

| Property | Recorded value |
|---|---|
| Model | Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL |
| Model format | GGUF |
| Layers | 48 |
| Experts per layer | 128 |
| Engine | qwen3moe |
| Device selection | sim |
| Language | Python |
| Framework | none |
| Workload phase | inference |

Model SHA256:

`eb331a4eee8eb6b5a8eb25f44f96f45c71b8d10f553c0a456190dd590a7ef77d`

Engine SHA256:

`bff0bb35ff25bada35e2b97464e8e275c94cc166824be59856e685dfedbe0`

Engine ELF BuildID:

`56804e8be3792d0ec242dda19d29ad5e3564ff1e`

**Important:** The recorded device is `sim`. The current experiment therefore establishes activation-prediction behavior under the simulation backend. It does not establish real GPU memory placement or hardware latency improvements.

### Evidence

- `pwea/artifacts/loco_five_category_cohort_manifest_20261115.json`
- `pwea/artifacts/loco_existing_cohort_manifest.json`

---

# 4. Completed Stage 1 — Expert Activation Telemetry Collection

## Objective

Collect expert activation events while preserving the identity of each task, prompt, session and workload.

## Collection methodology

Five software-development task categories were selected:

1. Code generation
2. Debugging
3. Code explanation
4. Code transformation
5. Algorithmic problem solving

Each category contains five sessions. Each session contains seven prompts.

The collection process uses:

- A fresh process for each session.
- Unique session and workload identifiers.
- Seven prompt turns per session.
- `/reset` between prompts.
- A deterministic process-level seed.
- Separate telemetry output for each session.
- Explicit task-category metadata.

Telemetry export windows are generated after two prompts and at the final prompt.

Expected completed-prompt sequence:

`[2, 2, 2, 2, 2, 2, 1]`

Expected export-window sizes:

`[2, 2, 2, 1]`

## Collection results

| Category | Sessions | Prompts | Status |
|---|---:|---:|---|
| Code generation | 5 | 35 | Complete |
| Debugging | 5 | 35 | Complete |
| Code explanation | 5 | 35 | Complete |
| Code transformation | 5 | 35 | Complete |
| Algorithmic problem solving | 5 | 35 | Complete |
| **Total** | **25** | **175** | **Complete** |

Total activation events:

**28,428,288**

### Evidence

- `pwea/data/raw_sessions/`
- `pwea/collect_pilot_session.py`
- `pwea/artifacts/loco_existing_cohort_manifest.json`
- `pwea/artifacts/loco_five_category_cohort_manifest_20261115.json`
- Per-session pilot reports and configurations under `pwea/`

### Collection limitations

- Code-generation S1–S4 were recollected on 12 November 2026. S5 is from 11 November 2026.
- Code-explanation S01 has a different workload-ID naming pattern from S02–S05. Its identity was accepted after checking the telemetry workload ID.
- Twenty of 35 algorithmic-problem-solving outputs reached the 1024-token generation cap. These are generation-censored outputs and must remain disclosed as a dataset limitation.

**Stage status: COMPLETED.**

---

# 5. Completed Stage 2 — Dataset Integrity and Provenance Audit

## Objective

Verify that the collected data is structurally consistent and that its source artifacts match their recorded hashes.

## Audit results

| Property | Result |
|---|---:|
| Audit status | Passed |
| Categories | 5 |
| Sessions | 25 |
| Prompts | 175 |
| Sessions per category | 5 |
| Prompts per category | 35 |
| Activation events | 28,428,288 |
| Reported errors | 0 |
| Algorithmic generation-cap hits | 20 |

The audit checked:

- Pilot reports and configurations.
- Model and engine hashes.
- Model dimensions.
- Simulation device selection.
- Telemetry, stdout and stderr hashes.
- Prompt-suite hashes and expected line counts.
- Parser validation.
- Workload identifiers and category labels.
- Prompt identifiers.
- Telemetry windows.
- Dropped-event information.
- Activation counts.

### Primary evidence

`pwea/artifacts/loco_dataset_integrity_audit_20261115.json`

Audit SHA256:

`7cc42c90cf34c9308ad611b70dca2d74676db7fff2ac838279eb643f6ab43d51`

Frozen original manifest:

`pwea/artifacts/loco_existing_cohort_manifest.json`

SHA256:

`d4259cc8f1ba8982059c895e6c002f6068dce98f734ac7c8b0139129cfc7ab7c`

Five-category cohort manifest:

`pwea/artifacts/loco_five_category_cohort_manifest_20261115.json`

SHA256:

`246e6a50990a5482b5079e0018b2f9a1823f684abbf10012062054863a57a01c`

**Stage status: COMPLETED — audit passed with zero reported errors.**

---

# 6. Completed Stage 3 — PWEA Expert Profile Construction

## Objective

Construct normalized expert activation profiles from the collected telemetry.

The PWEA work uses prompt-level activation shares to represent the distribution of expert demand.

The profile builder creates per-layer expert records and orders experts by descending mean prompt activation share, using expert ID as a tie-breaker.

This sorting behavior is important because it exposed a defect in the original evaluation implementation.

### Evidence

- `pwea/build_profile.py`
- `pwea/artifacts/`
- Corrected evaluation outputs described later in this report.

**Stage status: IMPLEMENTED AND USED IN THE EVALUATION PIPELINE.**

This profile-construction stage should not be confused with a trained future-demand prediction model.

---

# 7. Completed Stage 4 — Five-Category LOCO Evaluation

## Objective

Evaluate whether expert profiles built from four task categories can predict expert demand in a fifth, unseen category.

The protocol is Leave-One-Category-Out (LOCO).

For each fold:

- Four categories are used for training.
- One category is held out.
- Training contains 20 sessions and 140 prompts.
- Testing contains five sessions and 35 prompts.

## Fold structure

| Held-out category | Training sessions | Training prompts | Test sessions | Test prompts |
|---|---:|---:|---:|---:|
| Code generation | 20 | 140 | 5 | 35 |
| Debugging | 20 | 140 | 5 | 35 |
| Code explanation | 20 | 140 | 5 | 35 |
| Code transformation | 20 | 140 | 5 | 35 |
| Algorithmic problem solving | 20 | 140 | 5 | 35 |

Training support was recorded as eligible in all five folds:

- 140 training prompts.
- 20 independent source workloads.
- Minimum prompt support: 20.
- Minimum independent workload support: 3.
- `support_status: supported`.
- `eligible_for_prediction: true`.

Prompt-level evaluation records contain 35 held-out prompts and five held-out sessions. They do not contain a separate `heldout_support` object.

### Evidence

- `pwea/configs/loco_five_category_corrected_20261115.json`
- `pwea/configs/loco_five_category_prompt_corrected_20261115.json`
- Corrected evaluation JSON files.

**Stage status: COMPLETED.**

---

# 8. Completed Stage 5 — Evaluator Defect Discovery and Correction

## Defect discovered

The original evaluator compared score arrays by list position.

However, the profile builder independently sorts expert records by activation share.

Therefore, position zero in one profile was not guaranteed to represent the same expert ID as position zero in another profile.

This could result in comparing different experts as if they were identical.

The issue affected:

- Prompt-level evaluation.
- Session-level evaluation.

## Correction implemented

The evaluators were modified to align score vectors by `expert_id` before calculating ranking metrics.

The implementation validates that expert IDs uniquely cover the expected range:

`0..n_expert-1`

Corrected prompt-level helper:

`scores_by_expert_id`

Corrected session-level helper:

`_scores_by_expert_id`

Regression tests were added for expert-ID alignment and duplicate-ID rejection.

## Evidence

Corrected source:

- `pwea/evaluate_prompt_transfer.py`
- `pwea/evaluate_category_transfer.py`

Backups:

- `pwea/evaluate_prompt_transfer.py.pre_expert_id_alignment_20261115.bak`
- `pwea/evaluate_category_transfer.py.pre_expert_id_alignment_20261115_171247.bak`

Historical uncorrected outputs:

- `pwea/artifacts/loco_five_category_evaluation_20261115.json`
- `pwea/artifacts/loco_five_category_prompt_evaluation_20261115.json`

These historical results are superseded and must not be used for scientific claims.

**Stage status: COMPLETED.**

This correction is an important part of the project's research validation history and must be preserved in future reports.

---

# 9. Completed Stage 6 — Corrected Deterministic Evaluation

## Evaluation metrics

The evaluation uses:

- Precision@5
- Recall@5
- NDCG@5
- Spearman rank correlation

Parameters:

- `k = 5`
- `relevant_k = 13`

PWEA is compared against an aggregate-frequency baseline.

## 9.1 Session-level overall results

| Metric | PWEA | Frequency baseline |
|---|---:|---:|
| Precision@5 | 0.765667 | 0.721333 |
| Recall@5 | 0.294487 | 0.277436 |
| NDCG@5 | 0.784309 | 0.736263 |
| Spearman | 0.800715 | 0.767987 |

## Session-level NDCG@5 by category

| Held-out category | PWEA | Baseline |
|---|---:|---:|
| Code generation | 0.762319 | 0.751933 |
| Debugging | 0.847193 | 0.762000 |
| Code explanation | 0.786321 | 0.745510 |
| Code transformation | 0.848945 | 0.739164 |
| Algorithmic problem solving | 0.676766 | 0.682706 |

PWEA performs better in four of five categories. The frequency baseline performs slightly better for algorithmic problem solving.

## 9.2 Prompt-level overall results

| Metric | PWEA | Frequency baseline |
|---|---:|---:|
| Precision@5 | 0.712310 | 0.675738 |
| Recall@5 | 0.273965 | 0.259899 |
| NDCG@5 | 0.734869 | 0.689970 |
| Spearman | 0.731843 | 0.692361 |

## Prompt-level NDCG@5 by category

| Held-out category | PWEA | Baseline |
|---|---:|---:|
| Code generation | 0.720937 | 0.710120 |
| Debugging | 0.803826 | 0.723767 |
| Code explanation | 0.724196 | 0.686674 |
| Code transformation | 0.788793 | 0.687580 |
| Algorithmic problem solving | 0.636593 | 0.641711 |

Again, PWEA performs better in four of five categories.

## Interpretation

The corrected results support the conclusion that PWEA provides stronger overall cross-category expert-ranking quality than the aggregate-frequency baseline on this dataset and under this protocol.

They do not establish statistical significance, universal superiority, or improved inference performance.

Prompt-level and session-level results are different aggregation units and must not be conflated.

### Evidence — session level

- `pwea/artifacts/loco_five_category_corrected_evaluation_20261115.json`
- `pwea/artifacts/loco_five_category_corrected_run_summary_20261115.json`
- `pwea/artifacts/loco_five_category_corrected_checksums_20261115.sha256`

### Evidence — prompt level

- `pwea/artifacts/loco_five_category_prompt_corrected_evaluation_20261115.json`
- `pwea/artifacts/loco_five_category_prompt_corrected_run_summary_20261115.json`
- `pwea/artifacts/loco_five_category_prompt_corrected_checksums_20261115.sha256`

**Stage status: COMPLETED.**

---

# 10. Completed Stage 7 — Tie-Aware Ranking Sensitivity

## Objective

Measure the sensitivity of ranking metrics to tied expert scores.

The tie-aware implementation:

1. Groups exactly equal scores.
2. Calculates expected top-k inclusion probabilities.
3. Calculates expected overlap between predicted and actual top-k sets.
4. Calculates tie-aware Precision and Recall.
5. Calculates tie-aware NDCG using average gains within predicted tie groups.

Exact floating-point equality is used to identify ties; no tolerance is applied.

## Prompt-level results

| Metric | PWEA deterministic | PWEA tie-aware |
|---|---:|---:|
| Precision@5 | 0.712310 | 0.713143 |
| Recall@5 | 0.273965 | 0.274286 |
| NDCG@5 | 0.734869 | 0.734869 |

| Metric | Baseline deterministic | Baseline tie-aware |
|---|---:|---:|
| Precision@5 | 0.675738 | 0.676096 |
| Recall@5 | 0.259899 | 0.260037 |
| NDCG@5 | 0.689970 | 0.689970 |

Prompt-level Precision and Recall change slightly under tie-aware evaluation. Aggregate NDCG remains unchanged in the reported results.

At session level, deterministic and tie-aware results are effectively identical, with only negligible floating-point differences.

### Evidence — session level

- `pwea/artifacts/loco_five_category_tie_aware_evaluation_20261115.json`
- `pwea/artifacts/loco_five_category_tie_aware_run_summary_20261115.json`
- `pwea/artifacts/loco_five_category_tie_aware_checksums_20261115.sha256`

### Evidence — prompt level

- `pwea/artifacts/loco_five_category_prompt_tie_aware_evaluation_20261115.json`
- `pwea/artifacts/loco_five_category_prompt_tie_aware_run_summary_20261115.json`
- `pwea/artifacts/loco_five_category_prompt_tie_aware_checksums_20261115.sha256`

**Stage status: COMPLETED.**

---

# 11. Completed Stage 8 — Automated Tests and Reproducibility Checks

Recorded validation results:

- 58 automated tests passed.
- Python compilation checks passed for the relevant evaluator modules.
- Corrected session-level checksum verification passed.
- Corrected prompt-level checksum verification passed.
- Tie-aware session-level checksum verification passed.
- Tie-aware prompt-level checksum verification passed.
- Dataset integrity audit passed.

The checksum manifests preserve the exact configurations, inputs, source-code hashes, evaluation outputs and run summaries.

### Evidence

- `pwea/tests/`
- Corrected and tie-aware checksum manifests.
- `pwea/artifacts/loco_dataset_integrity_audit_20261115.json`

A passing test suite and matching checksums establish implementation and artifact integrity checks. They do not establish the absence of experimental bias or runtime performance improvement.

**Stage status: COMPLETED.**

---

# 12. Overall Completed Stage Summary

| Stage | Work | Status |
|---|---|---|
| 1 | Model and experimental environment | Completed |
| 2 | Expert activation telemetry collection | Completed |
| 3 | Dataset integrity and provenance audit | Passed |
| 4 | PWEA profile construction | Implemented |
| 5 | Five-category LOCO design | Completed |
| 6 | Expert-ID evaluator correction | Completed |
| 7 | Corrected deterministic evaluation | Completed |
| 8 | Tie-aware sensitivity analysis | Completed |
| 9 | Automated tests and checksum verification | Completed |

The telemetry and evaluation foundation is now established.

---

# 13. Remaining Stage 1 — Formalize the Expert Prediction Target

Before training a predictor, specify exactly what it must predict.

Possible targets:

- Whether an expert will be activated in the next prompt.
- Expected activation share of each expert.
- Probability that an expert appears in the top-k set.
- Expected expert demand over a future inference window.

The prediction timing must be explicit.

For example, using telemetry from the current prompt to predict the same prompt's expert demand is not a valid pre-inference prediction.

### Required deliverables

- Formal target definition.
- Input feature schema.
- Prediction horizon.
- Train/validation/test split rules.
- Data leakage checks.
- Evaluation protocol.

**Status: NOT YET COMPLETED.**

---

# 14. Remaining Stage 2 — Train an Expert Prediction Model

Train a model to estimate future expert demand from workload features and historical telemetry.

Potential models:

- Random Forest.
- Gradient-boosted trees.
- Logistic regression for binary activation.
- Per-layer classification or regression models.
- Lightweight neural networks if justified.

Compare against simple frequency and recency baselines.

### Required deliverables

- Training dataset.
- Feature extraction pipeline.
- Training script.
- Saved model artifact.
- Validation metrics.
- Calibration analysis.
- Error analysis.
- Independent session-level evaluation.

An earlier EAHMMF experiment trained a RandomForestClassifier using a separate dataset containing 767,755 valid rows and 128 experts. That work is historical supporting evidence, not proof that the current five-category cohort is already integrated into a validated runtime predictor.

**Status: FUTURE STAGE FOR THE CURRENT PWEA COHORT.**

---

# 15. Remaining Stage 3 — Expert Priority Scoring

Convert predicted expert demand into a priority score.

Potential factors:

- Predicted activation probability.
- Expected activation share.
- Recency of use.
- Historical frequency.
- Expert size.
- Loading cost.
- Transfer cost.
- Available VRAM and RAM.
- Safety margins.

The formula must be explicitly defined and validated.

### Required deliverables

- Mathematical priority formula.
- Parameter definitions.
- Unit tests.
- Ablation experiments.
- Comparison with LRU, LFU and frequency-only approaches.

**Status: NOT YET COMPLETED.**

---

# 16. Remaining Stage 4 — Hot/Warm/Cold Memory-Tier Policy

Implement the placement decision.

| Tier | Storage | Intended purpose |
|---|---|---|
| Hot | GPU VRAM | High-priority experts |
| Warm | System RAM | Medium-priority experts |
| Cold | SSD | Lower-priority experts |

The policy must enforce capacity constraints and avoid exceeding memory budgets.

### Required deliverables

- Tier capacity manager.
- Placement algorithm.
- Eviction algorithm.
- Expert residency tracking.
- Memory-budget validation.
- Allocation-failure handling.

**Status: NOT YET DEMONSTRATED.**

---

# 17. Remaining Stage 5 — Memory-Tier Simulator

Before integrating real hardware allocation, implement a simulator.

The simulator should model:

- VRAM capacity.
- RAM capacity.
- SSD capacity.
- Expert sizes.
- Expert demand.
- Cache hits and misses.
- Expert loading.
- Eviction.
- Migration.
- Transfer cost.
- Prediction errors.

Compare:

1. LRU.
2. LFU.
3. Aggregate-frequency placement.
4. PWEA-guided placement.
5. Learned-predictor placement.

### Required deliverables

- Memory-tier simulator.
- Policy implementations.
- Reproducible workloads.
- Capacity and eviction tests.
- Simulated transfer and miss statistics.

**Status: NOT YET COMPLETED.**

This is the recommended next implementation milestone after the predictor target and feature design are finalized.

---

# 18. Remaining Stage 6 — Real VRAM/RAM/SSD Integration

Integrate the memory policy into the actual MoE inference engine.

The system must be able to load, retain, evict and access experts from the intended tiers.

### Required deliverables

- VRAM allocation integration.
- RAM allocation integration.
- SSD-backed expert storage and loading.
- Synchronization and concurrency controls.
- Allocation failure recovery.
- Runtime correctness tests.

The present `sim` experiments do not demonstrate this functionality.

**Status: NOT YET COMPLETED.**

---

# 19. Remaining Stage 7 — Runtime Expert Migration

Implement migration between:

- SSD → RAM.
- RAM → VRAM.
- VRAM → RAM.
- RAM → SSD.

The migration system should measure:

- Transfer duration.
- Migration frequency.
- Inference stalls.
- Cache hit rate.
- Cost of incorrect predictions.
- Memory pressure.

### Required deliverables

- Migration scheduler.
- Migration queue.
- Residency state tracking.
- Asynchronous transfer where feasible.
- Failure recovery.
- Migration overhead measurements.

**Status: NOT YET COMPLETED.**

---

# 20. Remaining Stage 8 — Feedback and Adaptive Prediction

Compare predicted expert demand with actual routing.

Use prediction errors to update future estimates or placement priorities.

### Required deliverables

- Predicted-versus-actual telemetry.
- Drift detection.
- Online or periodic model updates.
- Adaptation safeguards.
- Static-versus-adaptive ablation.

**Status: NOT YET COMPLETED.**

---

# 21. Remaining Stage 9 — End-to-End Performance Benchmarking

Compare the complete EAHMMF system against relevant baselines.

Recommended comparisons:

1. Existing engine/default behavior.
2. LRU placement.
3. LFU placement.
4. Frequency-based placement.
5. PWEA-guided placement.
6. Learned-predictor-guided placement.

Measure:

- Time to first token.
- Inter-token latency.
- Total generation latency.
- Throughput.
- Peak and average VRAM.
- Peak and average RAM.
- SSD read volume.
- Expert load time.
- Migration time.
- Cache hit rate.
- Prediction precision and recall.
- Output correctness.

Run repeated trials and report variability. Separate model loading time from inference time.

**Status: NOT YET COMPLETED.**

---

# 22. Remaining Stage 10 — Generalization

Extend the research beyond the current five Python task categories and one model configuration.

Potential extensions:

- Other programming languages.
- Multiple frameworks.
- Non-coding workloads.
- Different prompt lengths.
- Different MoE model sizes.
- Other MoE architectures.
- Different hardware configurations.

Use independent workload splits and avoid training/test leakage.

**Status: FUTURE RESEARCH.**

---

# 23. Current Limitations and Threats to Validity

## Dataset size

Five sessions per category is an initial experimental cohort, not sufficient evidence for broad generalization.

## Generation cap

Twenty algorithmic-problem-solving outputs reached the 1024-token cap. These outputs are generation-censored.

## Collection dates

Code-generation S1–S4 were recollected on 12 November 2026; S5 is from 11 November 2026.

## Prompt correlation

Prompts within a session may be correlated. The 175 prompts should not automatically be treated as 175 independent workloads.

## Simulation backend

The current experiments use `sim`. Real GPU memory residency, SSD loading, transfer overhead and hardware inference latency have not been established.

## Single model

The current five-category cohort represents one model configuration.

## Statistical significance

The reported improvements are descriptive. No confidence intervals, hypothesis tests or independent replication study are established by the current evidence.

## Prediction versus placement

The present evaluation measures expert-ranking quality. It does not demonstrate successful memory placement or runtime improvement.

## Historical evaluator defect

The original uncorrected evaluation used mismatched expert identities. Those outputs must remain marked as superseded.

---

# 24. Recommended Immediate Roadmap

| Priority | Work item | Purpose |
|---|---|---|
| P0 | Freeze corrected PWEA results | Preserve the research baseline |
| P1 | Formalize prediction target and timing | Prevent ambiguity and leakage |
| P2 | Build prompt/session feature dataset | Prepare training data |
| P3 | Train expert-demand predictor | Establish predictive capability |
| P4 | Define expert priority scoring | Convert predictions into decisions |
| P5 | Build memory-tier simulator | Test placement policies safely |
| P6 | Integrate real VRAM/RAM/SSD | Validate actual placement |
| P7 | Implement migration and feedback | Complete runtime control loop |
| P8 | Benchmark against baselines | Demonstrate performance impact |
| P9 | Extend models and workloads | Test generalization |

---

# 25. Research Claims Supported by Current Evidence

The following claims are supported:

1. A 25-session, 175-prompt, five-category telemetry cohort has been collected.
2. The cohort contains 28,428,288 activation events.
3. The model has 48 layers and 128 experts per layer.
4. The integrity audit passed with zero reported errors.
5. Five-fold LOCO evaluation was completed at prompt and session levels.
6. An expert-ID alignment defect was identified and corrected.
7. Corrected PWEA results achieved higher overall NDCG@5 than the frequency baseline at both evaluation levels.
8. PWEA achieved higher NDCG@5 in four of five held-out categories.
9. Tie-aware sensitivity analysis was completed.
10. The recorded automated test suite passed 58 tests.

The following claims are **not yet supported**:

- EAHMMF has reduced inference latency.
- EAHMMF has optimized actual GPU VRAM usage.
- Experts are dynamically migrated between VRAM, RAM and SSD.
- The complete system outperforms existing memory managers.
- The results generalize to all MoE models or workloads.

---

# 26. Final Project Assessment

EAHMMF now has a meaningful experimental foundation: telemetry collection, dataset validation, expert-profile construction, corrected cross-category evaluation, tie-aware sensitivity analysis and reproducibility artifacts.

The most important next milestone is to turn the current expert-ranking evidence into a properly specified future-demand predictor.

After validating the predictor, the project should move to a memory-tier simulator, then real memory allocation and runtime migration.

This order separates prediction quality from placement quality and prevents claims about hardware performance before that performance has actually been measured.

**Current maturity:** Telemetry and cross-category expert-ranking evaluation foundation established. Predictive memory placement and runtime hierarchical memory management remain future implementation stages.

---

# Appendix A — Primary Evidence Inventory

## Dataset and integrity

- `pwea/artifacts/loco_dataset_integrity_audit_20261115.json`
- `pwea/artifacts/loco_existing_cohort_manifest.json`
- `pwea/artifacts/loco_five_category_cohort_manifest_20261115.json`
- `pwea/data/raw_sessions/`

## Corrected deterministic session-level evaluation

- `pwea/artifacts/loco_five_category_corrected_evaluation_20261115.json`
- `pwea/artifacts/loco_five_category_corrected_run_summary_20261115.json`
- `pwea/artifacts/loco_five_category_corrected_checksums_20261115.sha256`

## Corrected deterministic prompt-level evaluation

- `pwea/artifacts/loco_five_category_prompt_corrected_evaluation_20261115.json`
- `pwea/artifacts/loco_five_category_prompt_corrected_run_summary_20261115.json`
- `pwea/artifacts/loco_five_category_prompt_corrected_checksums_20261115.sha256`

## Tie-aware evaluation

- `pwea/artifacts/loco_five_category_tie_aware_evaluation_20261115.json`
- `pwea/artifacts/loco_five_category_tie_aware_run_summary_20261115.json`
- `pwea/artifacts/loco_five_category_tie_aware_checksums_20261115.sha256`
- `pwea/artifacts/loco_five_category_prompt_tie_aware_evaluation_20261115.json`
- `pwea/artifacts/loco_five_category_prompt_tie_aware_run_summary_20261115.json`
- `pwea/artifacts/loco_five_category_prompt_tie_aware_checksums_20261115.sha256`

## Configurations

- `pwea/configs/loco_five_category_corrected_20261115.json`
- `pwea/configs/loco_five_category_prompt_corrected_20261115.json`
- `pwea/configs/loco_five_category_tie_aware_20261115.json`
- `pwea/configs/loco_five_category_prompt_tie_aware_20261115.json`

## Implementation and tests

- `pwea/build_profile.py`
- `pwea/evaluate_profile.py`
- `pwea/evaluate_category_transfer.py`
- `pwea/evaluate_prompt_transfer.py`
- `pwea/tests/`

---

# Appendix B — Important Research Distinction

Always distinguish:

1. **Activation measurement:** what telemetry observed.
2. **Expert-demand prediction:** what the model predicts will be used.
3. **Memory-placement outcome:** whether experts are actually placed and migrated efficiently.

The present PWEA results establish the activation-measurement foundation and evaluate profile-based cross-category expert ranking. They do not establish end-to-end memory-placement performance.
