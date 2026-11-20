# R4 — CUDA Numerical Correctness Report

**Date:** 2026-11-11
**Project:** EAHMMF / ETF Telemetry
**Milestone:** R4 — CUDA numerical correctness
**Status:** Core correctness validation passed; historical discrepancy remains unreproduced.

---

## 1. Objective

R4 validates the numerical correctness of CUDA expert-matrix computation against the CPU reference implementation using the full Qwen3-Coder model.

This is distinct from R3, which validated CUDA-backed expert residency but did not enable CUDA expert-matrix computation.

The R4 tests exercise:
1. General CUDA expert-matmul kernel.
2. Optimized warp-based CUDA expert-matmul kernel.
3. Paired gate/up CUDA execution path.
4. Repeatability of the paired gate/up path.

## 2. Test environment and workload

- Model: Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf
- Model SHA-256: `eb331a4eee8eb6b5a8eb25f44f96f45c71b8d10f553c0a456190dd590a7ef77d`
- Executable SHA-256: `e593ce2ce2993e2d7ec5ab09ba73a8c50b227c0348056d494b2d58e19ec8d0bd`
- GPU: NVIDIA GeForce RTX 3050 A Laptop GPU
- GPU memory: 4094 MiB
- Driver: 615.78.08
- CUDA UMD: 13.4
- CUDA toolkit: 13.2.86
- CPU: Intel Core i5-13450HX
- Context: 512
- Threads: 8
- Generation length: 16 tokens
- Temperature: 0
- Expert execution tier: VRAM
- Device: CUDA
- RAM budget: 512 MiB
- VRAM budget: 512 MiB
- VRAM reserve: 1024 MiB
- Expert policy: LRU

Prompt:
`Write a Python function to calculate Fibonacci numbers.`

The full model includes 48 layers and 128 experts. The tested expert formats include Q4_K, Q5_K, Q6_K and Q8_0.

## 3. Correctness methodology

The CUDA expert computation was enabled with `--cuda-compute`, and CPU-reference comparison was enabled with `--cuda-shadow`.

The reported `cuda-shadow` summary provides:
- Number of comparisons (`calls`)
- Number of comparisons outside the configured tolerance (`bad`)
- Number of skipped comparisons (`skipped`)
- Maximum reported relative error (`max_rel_err`)

The configured shadow tolerance is 1e-4.

A successful test requires:
- Process exit code 0.
- Expected CUDA calls.
- Zero CPU fallbacks.
- Zero bad comparisons.
- Zero skipped comparisons.
- Successful acquisition accounting with zero failed acquisitions.

## 4. Results

| Execution path | CUDA calls | CPU fallbacks | Pair calls | Shadow calls | Bad | Skipped | Max relative error |
|---|---:|---:|---:|---:|---:|---:|---:|
| General CUDA kernel | 32,364 | 0 | 0 | 32,364 | 0 | 0 | 3.45e-06 |
| Warp CUDA kernel | 32,364 | 0 | 0 | 32,364 | 0 | 0 | 8.51e-07 |
| Paired gate/up | 32,364 | 0 | 10,788 | 32,364 | 0 | 0 | 8.51e-07 |
| Paired gate/up repeat | 32,364 | 0 | 10,788 | 32,364 | 0 | 0 | 8.51e-07 |

All four runs exited with status 0.

### Acquisition accounting

Each run completed 10,788 successful acquisitions and zero failed acquisitions. Serving accounting reported PASS.

The paired repeat independently reproduced:
- 32,364 CUDA calls.
- 10,788 paired calls.
- 32,364 shadow comparisons.
- Zero bad comparisons.
- Zero skipped comparisons.
- Maximum relative error of 8.51e-07.

## 5. Performance observations

| Execution path | Wall time | Generation speed |
|---|---:|---:|
| General CUDA kernel | 63.55 s | 0.51 tok/s |
| Warp CUDA kernel | 31.00 s | 0.91 tok/s |
| Paired gate/up | 30.57 s | 0.92 tok/s |
| Paired gate/up repeat | 29.87 s | 0.93 tok/s |

The general-to-warp and warp-to-paired timing differences are observations from individual runs, not statistically established performance claims.

The paired repeat confirms a similar generation rate to the first paired run. These runs were correctness tests, not a controlled performance study.

## 6. Historical discrepancy from the R3 report

The R3 report records an earlier full-model shadow comparison with:
- 186 mismatches out of 32,364 comparisons.
- Maximum reported relative error of 3.73.

The retained R3 benchmark commands did not enable `--cuda-compute` or `--cuda-shadow`. R3 therefore does not contain the raw execution evidence for that earlier numerical comparison.

A repository-wide search did not recover the original raw output or exact command/configuration associated with the 186 mismatches.

All four R4 execution runs passed with zero mismatches. This means the earlier discrepancy was not reproduced under the R4 configurations. Its original cause cannot be established from the retained evidence.

**The historical discrepancy is therefore classified as unreproduced and unexplained, not conclusively resolved.**

## 7. Evidence layout and integrity

Original R4 runs:
- `general_kernel/`
- `warp_kernel/`
- `paired_kernel/`

Repeat:
- `paired_kernel_repeat/`

Integrity manifests:
- `SHA256SUMS` — original 12 R4 evidence files.
- `paired_kernel_repeat/SHA256SUMS` — 7 repeat evidence files.

Both manifests passed checksum verification after the repeat was completed.

The report is separate from the original run logs. No R0, R2 or R3 evidence was modified.

## 8. Limitations

1. Validation covers one full model, one GPU, one machine and one prompt/configuration.
2. The generated output is short (16 tokens).
3. The tests validate the exercised quantized expert-matmul formats and paths; they do not establish correctness for every model architecture, quantization format, shape or GPU.
4. The four runs are not a statistically controlled performance benchmark.
5. The historical 186-mismatch experiment has no recovered raw log or exact configuration.
6. The reported maximum relative error is the implementation's shadow metric and should not be interpreted as a conventional per-element relative error without examining its calculation.

## 9. Conclusion

The general CUDA kernel, optimized warp kernel and paired gate/up path all passed full-model CPU-reference shadow validation under the tested workload.

Across four runs:
- 129,456 shadow comparisons were performed.
- Zero mismatches were reported.
- Zero comparisons were skipped.
- Zero CPU fallbacks were reported.
- All runs exited successfully.
- All runs passed acquisition accounting.

The paired gate/up path also passed an independent repeat.

**R4 core CUDA numerical correctness validation: PASS for the tested model, workload and implementation paths.**

The historical R3-reported discrepancy remains an explicit unresolved evidence limitation.

## 10. Next research stages

Suggested continuation:
- R5: broader workload and prompt validation.
- R6: ablation studies and multi-model evaluation.
- R7: packaging, reproducibility and documentation.

These are future stages and are not claimed as completed by R4.
