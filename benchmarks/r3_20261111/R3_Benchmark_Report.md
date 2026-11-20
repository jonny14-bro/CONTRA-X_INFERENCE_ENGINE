# R3 — Reproducible Expert Residency Benchmark

**Execution date:** 11 November 2026
**Measured trial window:** 2026-11-11T12:50:47+05:30 to 2026-11-11T12:58:53+05:30
**Status:** 15/15 measured trials passed strict validation

## 1. Objective

Compare LRU, Frequency 1024, and Frequency 4096 expert residency policies
under one fixed real-model workload. This evaluates CUDA-backed expert
residency and policy behavior, not CUDA expert-matrix computation.

## 2. Reproducibility identity

| Item | Recorded identity |
|---|---|
| Model | Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf |
| Model SHA256 | `eb331a4eee8eb6b5a8eb25f44f96f45c71b8d10f553c0a456190dd590a7ef77d` |
| Model size | 21,740,305,568 bytes |
| CUDA executable SHA256 | `e593ce2ce2993e2d7ec5ab09ba73a8c50b227c0348056d494b2d58e19ec8d0bd` |
| CPU executable SHA256 | `3746523539abedf2b487d744ecc866554bc3d33c62b0998da982d5868c2986b6` |
| Runner SHA256 | `99e2792b4ee00ead97806a34480187e3b54c70742c251bf5fb263a80b9836525` |
| CPU | 13th Gen Intel Core i5-13450HX |
| GPU | NVIDIA GeForce RTX 3050 A Laptop GPU, 4094 MiB |
| NVIDIA driver | 615.78.08 |
| Kernel | 7.2.11-200.fc44.x86_64 |
| CUDA toolkit | 13.2 (nvcc 13.2.86) |

Full environment capture: `metadata/environment.txt`.

## 3. Workload and configuration

- Prompt: `Write a Python function to calculate Fibonacci numbers.`
- Generation limit: 16 tokens; context: 512 tokens.
- Threads: 8; temperature: 0.
- Device: CUDA; expert tier: VRAM.
- RAM budget: 512 MiB.
- VRAM budget: 512 MiB; reserve: 1024 MiB.
- Policies: LRU, Frequency 1024, Frequency 4096.
- One successful warm-up per policy.
- Five measured trials per policy, randomized order by round.
- Random seed: `20261111`; 15-second interval between measured runs.
- No `--cuda-compute` option was used.

Commands and per-run records are retained in `results.jsonl` and trial logs.

## 4. Validation

- Successful CUDA warm-ups: 3/3.
- Measured trials: 15/15 returned code zero and validation PASS.
- Every measured trial generated exactly 16 tokens.
- All serving-accounting checks passed.
- Acquires = successful acquisitions = releases for every run.
- Failed expert acquisitions: zero.
- SSD + RAM + VRAM serving totals equal successful acquisitions.
- VRAM allocation/recovery error counters: zero.
- Residency event logs were present for all trials.

The initial CPU-only binary failure is retained under `failed_attempts/`.
It documents the missing-CUDA build and the reason for rebuilding.

## 5. Performance results

| Policy | n | Mean wall (s) | Median (s) | SD (s) | Min–max (s) | Mean generation (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 5 | 30.0791 | 30.1030 | 0.1028 | 29.9012–30.1605 | 0.9140 |
| Frequency 1024 | 5 | 14.3301 | 14.3472 | 0.1342 | 14.1127–14.4824 | 1.9600 |
| Frequency 4096 | 5 | 10.8210 | 10.8001 | 0.0700 | 10.7464–10.9215 | 2.6400 |

### Paired wall-time reduction by round

- Frequency 4096 vs LRU: 64.29%, 64.20%, 63.96%, 63.79%, 63.88% (mean 64.02%).
- Frequency 4096 vs Frequency 1024: 25.10%, 23.64%, 24.39%, 23.86%, 25.43% (mean 24.48%).

These are descriptive comparisons across five paired rounds; no statistical
significance claim is made.

## 6. Host resource measurements

| Policy | Mean peak RSS (GiB) | Mean user CPU (s) | Mean system CPU (s) |
|---|---:|---:|---:|
| LRU | 13.61 | 61.41 | 3.63 |
| Frequency 1024 | 13.61 | 45.74 | 2.66 |
| Frequency 4096 | 13.60 | 41.79 | 2.55 |

Peak host RSS was approximately 13.6 GiB across all policies.

## 7. Final serving-tier totals

| Policy | SSD served | RAM served | VRAM served | Successful acquires | Failed acquires |
|---|---:|---:|---:|---:|---:|
| LRU | 0 | 0 | 53,940 | 53,940 | 0 |
| Frequency 1024 | 18,875 | 14,780 | 20,285 | 53,940 | 0 |
| Frequency 4096 | 29,065 | 10,880 | 13,995 | 53,940 | 0 |

## 8. Migration totals

Each cell reports `successful migrations / failed migration attempts`.
A migration failure is not a failed expert acquisition.

| Policy | RAM→SSD | RAM→VRAM | SSD→RAM | VRAM→RAM | VRAM→SSD |
|---|---:|---:|---:|---:|---:|
| LRU | 52,370 / 0 | 53,940 / 0 | 53,940 / 0 | 53,175 / 0 | 0 / 0 |
| Frequency 1024 | 19,105 / 0 | 15,235 / 14,780 | 25,455 / 18,875 | 9,715 / 4,685 | 4,685 / 0 |
| Frequency 4096 | 10,960 / 0 | 6,790 / 10,880 | 13,535 / 29,065 | 5,015 / 955 | 955 / 0 |

## 9. GPU snapshots

The runner queries `nvidia-smi` in this order:
`temperature.gpu,power.draw,utilization.gpu,memory.used`.
The tuple fields are temperature (°C), power (W), utilization (%) and
memory used (MiB). These are point-in-time readings before and after each
run, not continuous telemetry; they do not establish peak or average values.

## 10. Interpretation

Frequency 4096 had the lowest mean wall time and highest mean generation
throughput in this experiment. Its mean wall time was 64.02% lower than
LRU and 24.48% lower than Frequency 1024.

The result is specific to this model, machine, configuration, prompt and
short generation workload. It should not be generalized to other models,
hardware, prompt distributions or longer workloads without further tests.

## 11. Limitations and next stages

1. CUDA-backed residency was tested; CUDA expert-matrix computation was not.
2. Earlier full-model shadow comparison showed a mismatch of 186 / 32,364
   and maximum relative error 3.73. Real-model numerical correctness remains open.
3. Five trials per policy are an initial controlled comparison, not a broad
   performance characterization.
4. GPU readings are snapshots rather than continuous measurements.
5. Other prompts, generation lengths, models and hardware remain to be tested.

Next: R4 CUDA numerical correctness; R5 broader workloads and repetitions;
R6 ablations and multi-model evaluation; R7 packaging and documentation.

## 12. Evidence index

- `results.jsonl`: raw structured run records.
- `trials/`: stdout, stderr, timing and residency logs for all 15 trials.
- `warmup/`: successful CUDA warm-up logs.
- `failed_attempts/`: retained CPU-only failure evidence.
- `metadata/environment.txt`: environment and executable identities.
- `metadata/cuda_smoke_test.log`: CUDA residency smoke test.
- `metadata/trials_console.log`: trial runner console output.
- `run_benchmarks.py`: benchmark runner.
- `metadata/runner_sha256.txt`: runner hash record.

## 13. Scope of checksum manifest

`SHA256SUMS` covers this report and primary benchmark evidence. Backup files
and Python bytecode are retained but excluded from the manifest.
