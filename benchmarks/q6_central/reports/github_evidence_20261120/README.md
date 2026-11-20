# Q6 Central Validation — Evidence Bundle

**Execution date:** 20 November 2026
**Project:** EAHMMF / CONTRAX
**Evidence type:** Q6 kernel validation and real-workload numerical error attribution

## 1. Purpose

This directory preserves a compact evidence package for the Q6 central
validation and the associated real-workload numerical error analysis.

It contains aggregate reports and provenance information. It does not
contain the full captured weight and activation inputs.

## 2. Central validation suite

Recorded run: `run_20261120_191240`

| Measurement | Result |
|---|---:|
| Tests | 9 |
| Passed | 9 |
| Failed | 0 |
| Skipped | 0 |
| Recorded duration | 380.144 ms |

The tests cover the integrated Q6 kernel, checked Q8G16 quantization,
finite and non-finite edge cases, subnormal values, rounding modes,
FTZ/DAZ behavior, stress validation and rounding diagnostics.

Recorded environment:

- CPU: Intel Core i5-13450HX
- Compiler: GCC 16.2.1
- C++ standard: C++17
- Kernel: Linux 7.2.11-200.fc44.x86_64
- Run mode: `--full`

These results establish the recorded test outcomes on this environment.
They do not establish performance or correctness across all hardware,
compilers, model architectures or workloads.

## 3. Real-workload numerical error attribution

Recorded run: `run_20261120_210022`

| Measurement | Result |
|---|---:|
| Captured records | 5,000 |
| Activation tokens | 13,288 |
| Output pairs analyzed | 20,712,064 |
| Threads | 8 |
| Zeroing NRMSE | 0.139392% |
| Rounding NRMSE | 0.434884% |
| Total NRMSE | 0.451601% |
| Closure maximum absolute error | 4.3791e-05 |
| Closure RMSE | 1.7344e-07 |
| Non-finite outputs | 0 |

The operation-level breakdown is in
`real_workload/attribution/operation_summary.tsv`.

The aggregate summary is in
`real_workload/attribution/summary.json`.

## 4. Input provenance

The real-workload run used:

- Weight input size: 11,495,159,808 bytes
- Activation input size: 375,792,640 bytes
- Descriptor SHA-256:
  `f2e418c87cb051ec10071fbf8d343e07c71d5f63edd1508dc0a0602efa482f39`

The large inputs are not included in this evidence bundle or intended
to be distributed as part of this compact repository evidence package.

Reproduction requires obtaining the exact inputs, verifying their
provenance and descriptor hash, and running:

```bash
benchmarks/q6_central/real_workload_benchmark.sh --check
benchmarks/q6_central/real_workload_benchmark.sh --all
```

## 5. Evidence files

- `q6_central_report.json` — central validation report.
- `real_workload/manifest.txt` — real-workload run identity and input metadata.
- `real_workload/attribution/summary.json` — aggregate attribution results.
- `real_workload/attribution/operation_summary.tsv` — operation-level results.

## 6. Reproducibility limitation

The real-workload manifest records the Git revision as unavailable.
Therefore, these results document the recorded working-tree execution;
they are not pinned to a specific source commit.

For a release-quality claim, repeat the validation against the exact
release commit and preserve the resulting source revision and input
identities.

## 7. Interpretation

The recorded suite passed all nine central validation tests. The
real-workload attribution measured a total NRMSE of 0.451601% for the
captured dataset.

These are numerical validation results, not proof of end-to-end
inference speedup, universal model compatibility, or performance
improvement across hardware.
