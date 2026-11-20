# EAHMMF Inference Engine --- v1.1.0

> **Research-oriented Qwen3-MoE inference engine with expert-aware
> memory management, CUDA-assisted expert computation, and optimized
> quantized CPU matrix multiplication.**

**Release:** `v1.1.0` (proposed)\
**Status:** Release-candidate documentation; final regression and
release checks remain.\
**Predecessor:** [EAHMMF Inference Prototype
v0.1.0](https://github.com/jonny14-bro/EAHMMF_inference_prototype_V-0.1.0)

This repository documents the next stage in the evolution of the
original EAHMMF inference prototype. The original repository remains a
separate historical milestone; this project does not replace it.

## Overview

EAHMMF Inference Engine is an experimental C++ inference engine for
Qwen3-MoE GGUF models. Its development combines:

1.  **Expert-aware inference and memory management** --- expert
    residency and integration of EAHMMF research components.
2.  **Optimized numerical execution** --- optimized CPU quantized matrix
    multiplication while retaining CUDA execution paths and a
    conservative FP32 fallback.

The v1.1.0 development milestone introduces **Q8G16 V2 as the
first-choice optimized CPU path** for eligible Q6_K matrix
multiplications. Q8K remains available as a secondary implementation,
and the original FP32 implementation remains the general fallback.

The design does not force every tensor through one optimized kernel. It
selects a supported implementation for each operation while preserving a
fallback for unsupported formats, shapes, hardware capabilities, or
invalid quantization inputs.

### Key capabilities

-   Qwen3-MoE GGUF model loading and inference.
-   MoE expert routing and expert feed-forward execution.
-   CUDA-assisted expert matrix multiplication / resident expert
    execution path.
-   Q8G16 V2 SIMD activation quantization and AVX-VNNI dot-product path
    for eligible Q6_K operations.
-   Q8K secondary CPU path.
-   Original FP32 dequantization and matrix multiplication fallback.
-   Checked activation quantization, including handling for non-finite
    values and floating-point environment differences.
-   EAHMMF/PWEA/ETF research-component integration.
-   Replay-based accuracy and performance experiments, plus short
    real-model inference smoke tests.

## Evolution from v0.1.0

The original [EAHMMF inference prototype
v0.1.0](https://github.com/jonny14-bro/EAHMMF_inference_prototype_V-0.1.0)
established the foundational inference and expert-aware
memory-management work.

The v1.1.0 line is a separate repository so that the development history
and architectural evolution can be inspected independently.

  -----------------------------------------------------------------------
  Area                    v0.1.0 prototype        v1.1.0 development
  ----------------------- ----------------------- -----------------------
  Inference foundation    Initial Qwen3-MoE       Continued engine
                          inference prototype     development

  Expert execution        Initial CUDA-assisted   CUDA path retained
                          execution               alongside optimized CPU
                                                  paths

  CPU Q6_K computation    Earlier baseline        Q8G16 V2 first, Q8K
                          implementation          secondary

  Dispatch                Earlier execution logic Capability- and
                                                  shape-aware dispatch

  Fallback                Baseline implementation Original FP32 path
                                                  retained

  Numerical testing       Initial prototype       Replay comparisons,
                          validation              quantizer edge cases
                                                  and runtime dispatch
                                                  checks

  Research integration    EAHMMF foundation       Continued
                                                  EAHMMF/PWEA/ETF
                                                  integration
  -----------------------------------------------------------------------

This describes the development direction and tested additions; it does
not imply every subsystem was rewritten or every research component is
active in every inference mode.

## Architecture

### 1. High-level inference and expert-execution flow

``` mermaid
flowchart TD
    A[Prompt / Input Tokens] --> B[Tokenizer]
    B --> C[Qwen3-MoE Transformer]
    C --> D[MoE Router]
    D --> E[Top-k Expert Selection]
    E --> F[Expert FFN]
    F --> G{Execution path}
    G -->|CUDA path available| H[CUDA Expert MatMul / Resident Execution]
    G -->|CPU execution| I[CPU MatMul Dispatcher]
    I --> J{Eligible Q6_K shape and CPU?}
    J -->|Yes| K[Q8G16 V2 SIMD + AVX-VNNI]
    J -->|No| L{Q8K eligible?}
    L -->|Yes| M[Q8K + AVX-VNNI]
    L -->|No| N[Original FP32 fallback]
    K --> O[Expert Output]
    M --> O
    N --> O
    H --> O
    O --> P[Continue Transformer Layers]
    P --> Q[Logits / Sampling]
    Q --> R[Generated Tokens]
```

The CPU dispatcher is shape-, format- and capability-aware. Q8G16 V2 is
the first optimized CPU choice for eligible Q6_K operations. Q8K is the
secondary implementation when Q8G16 is unavailable or deliberately
disabled and Q8K remains eligible. If neither optimized path can safely
execute, the original FP32 implementation is retained.

**Important:** the current Q8K dot-product implementation also requires
AVX-VNNI. It is not yet a universal fallback for CPUs without AVX-VNNI.
Such CPUs use the original FP32 path unless a non-VNNI Q8K kernel is
added.

### 2. CPU dispatch sequence

``` mermaid
flowchart TD
    A[Q6_K CPU matmul request] --> B{Q8G16 eligible?}
    B -->|Yes| C[Checked Q8G16 quantization]
    C --> D{All activation blocks valid?}
    D -->|Yes| E[Q8G16 V2 dot kernel]
    D -->|No| H[Original FP32 fallback]
    B -->|No| F{Q8K eligible and not rejected?}
    F -->|Yes| G[Q8K dot kernel]
    F -->|No| H
    E --> I[Return output]
    G --> I
    H --> I
```

Eligibility includes Q6_K weight type, supported dimensions and block
layout, and CPU instruction-set requirements. Q8G16 uses AVX2 SIMD
quantization and an AVX-VNNI dot kernel. Q8K also uses AVX-VNNI.

The checked Q8G16 quantizer rejects invalid activation blocks. On
rejection, the dispatcher deliberately skips Q8K and uses the FP32
fallback.

### 3. CUDA and CPU execution

The engine retains its CUDA expert-computation path. When CUDA expert
computation succeeds, it returns the result; CPU matrix multiplication
is used when the CUDA path is not taken or does not complete the
operation.

The CPU dispatcher then selects Q8G16, Q8K, or FP32 according to
eligibility and validation.

This README does not claim every model operation runs on the GPU or
every expert is permanently resident in VRAM. Actual residency and
execution depend on the configured runtime path, available memory,
tensor support and runtime state.

## CPU quantized matrix multiplication

### Q8G16 representation

The groupwise activation representation uses 16 independent groups of 16
values per 256-element activation block.

``` cpp
struct Q8G16 {
    int8_t  qs[256];
    float   d[16];
    int16_t bsums[16];
};
```

Each group has its own scale and subgroup sum. Quantized values are
clamped to `[-127, 127]`. The current structure occupies **352 bytes per
256-element activation block**.

### Q8G16 V2 execution

The V2 dot implementation:

1.  Quantizes activations into Q8G16 using the AVX2 SIMD quantizer.
2.  Decodes Q6_K weights into a predecoded representation.
3.  Computes integer dot products using AVX-VNNI.
4.  Applies the group-sum correction.
5.  Converts and accumulates group results using the V2 floating-point
    accumulation path.

The checked quantization wrapper controls the floating-point rounding
environment and rejects non-finite or unsupported activation inputs. It
restores the caller's floating-point environment after execution.

### Q8K secondary path

Q8K remains available as a secondary optimized CPU implementation. It is
retained for controlled comparison and as a secondary dispatch option
where its eligibility requirements are met.

The current Q8K dot implementation requires AVX-VNNI. It therefore does
not provide a non-VNNI CPU fallback.

### FP32 fallback

The original dequantization and FP32 dot-product path remains in the
engine. It is used for unsupported weight formats, unsupported
dimensions, unavailable CPU capabilities, and checked Q8G16 quantization
rejection.

Keeping this path is intentional: optimized kernels must not be used
outside their supported conditions.

## EAHMMF, PWEA and ETF integration

The engine continues the research direction established by the original
EAHMMF prototype.

The intended research workflow is:

``` mermaid
flowchart TD
    A[Inference Prompt] --> B[Task Summary / Planning]
    B --> C[PWEA Expert Prediction]
    C --> D[EAHMMF Memory Placement]
    D --> E[Inference Execution]
    E --> F[ETF Telemetry]
    F --> G[Runtime Measurements / Feedback]
    G --> C
```

-   **EAHMMF --- Expert-Aware Hierarchical Memory Management
    Framework:** manages expert placement across memory tiers such as
    VRAM, RAM and SSD.
-   **PWEA:** the prediction-related component used in the
    expert-placement workflow.
-   **ETF --- Expert Telemetry Framework:** collects expert-activation
    and runtime observations for analysis and feedback.

The research workflow and CPU kernel dispatcher are related but distinct
parts of the system. The Q8G16 dispatch smoke tests documented below
were run with `--no-etf-collection`; one captured baseline log also
reported prediction disabled. Those tests validate CPU dispatch and
inference, **not** the complete
prompt-to-prediction-to-placement-to-telemetry feedback loop.

Do not interpret the existence of these integrations as proof that
predictive placement and telemetry collection were active during every
benchmark.

## Benchmark results

The results below come from local experiments on the hardware described
in [Hardware and software
environment](#hardware-and-software-environment). They are research
measurements, not independently reproduced third-party benchmarks.

### 1. Checked Q8G16 replay benchmark

The checked replay benchmark compared the existing Q8K path with the
checked Q8G16 V2 + SIMD path. The reported aggregate is the sum of
per-record medians; a separate median-of-run aggregate is also shown.

  Metric                                Q8K   Q8G16 V2 + SIMD
  --------------------------- ------------- -----------------
  Sum of per-record medians     4.950583 ms       3.937160 ms
  Median per-run aggregate      5.007642 ms       4.001904 ms

**Measured improvement:** 20.47% using the sum of per-record medians;
20.08% using the median per-run aggregate.

Test conditions: 8 threads, 2 warmups, 7 trials, repeated runs. These
are replay measurements over captured activation/weight records. They
are **not** end-to-end model inference latency.

### 2. Activation approximation accuracy on six replay records

NRMSE compares each quantized activation representation against the FP32
activation reference. Lower is better.

  Operation      Q8K NRMSE   Q8G16 NRMSE   Relative reduction
  ------------ ----------- ------------- --------------------
  `moe_gate`     0.821213%     0.400375%                51.2%
  `moe_up`       0.916685%     0.443675%                51.6%
  `moe_down`     1.716142%     0.732849%                57.3%
  `attn_q`       1.406302%     0.427661%                69.6%
  `attn_k`       1.902267%     0.575626%                69.7%
  `attn_v`       4.165959%     1.255601%                69.9%

Across these six captured records, Q8G16 reduced activation NRMSE by
approximately 51--70% relative to Q8K.

These results describe the tested records and are not a guarantee for
every possible activation distribution or model.

### 3. Q8G16 V2 versus V1 numerical comparison

Across the six replay operations, the V2 versus V1 output comparison
reported:

-   NRMSE approximately `0.000010%` to `0.000015%`.
-   Maximum absolute difference approximately `1.19e-6` to `2.86e-6`.
-   No non-finite outputs in the tested records.

The very small difference is consistent with the different
floating-point accumulation order used by V2. It does not establish
bitwise identity.

### 4. SIMD quantizer equivalence

The scalar and AVX2 SIMD Q8G16 quantizers were compared over **1,224
activation blocks** from the six replay records.

-   Byte mismatches: **0**
-   Scalar/SIMD output equality: **PASS** for the tested blocks.

### 5. Replay performance context

An earlier unchecked replay comparison reported a 25.88% aggregate
improvement. The checked benchmark is the more conservative and relevant
result for this README, so the headline figure is the **20.08% checked
aggregate improvement**.

The replay benchmark compares specific implementations and records. It
should not be presented as a 20.08% improvement in full-model tokens per
second.

## Numerical validation

The Q8G16 checked quantizer and related tests covered:

  -----------------------------------------------------------------------
  Test                                Recorded result
  ----------------------------------- -----------------------------------
  Scalar vs SIMD replay quantization  1,224 blocks, zero byte mismatches

  Rounding modes                      4 modes × 10,000 cases; zero
                                      mismatches, zero restoration errors

  FTZ/DAZ combinations                Scalar/SIMD equality and
                                      floating-point environment
                                      restoration passed

  Finite edge inputs                  Zero, tiny normal, large finite,
                                      mixed-sign and near-zero cases
                                      passed

  Non-finite inputs                   NaN and ±Inf rejected

  Subnormal-only input                Rejected when scale computation
                                      became invalid

  Rejected output handling            Output zeroed by checked wrapper
  -----------------------------------------------------------------------

These tests provide evidence for the checked quantizer's behavior under
the tested conditions. They are not exhaustive formal verification of
all floating-point inputs, compiler versions or processor
implementations.

## Runtime validation

Short real-model inference tests used:

-   Model: `Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf`
-   48 transformer layers
-   128 experts, top-8 routing
-   Prompt: `Explain how SIMD improves quantized matrix multiplication.`
-   Generation: 8 tokens
-   Context: 512
-   Batch: 1
-   Threads: 8
-   Temperature: 0
-   ETF collection disabled for the documented smoke tests

### Automatic default dispatch

With no Q8G16 or Q8K disable override and with tracing enabled:

  Dispatch trace      Count
  ---------------- --------
  Q8G16 V2           15,003
  Q8K                     0
  FP32               53,214

The FP32 calls include unsupported formats such as Q5_K, Q8_0 and F32.
Q8G16 V2 was selected for eligible Q6_K operations.

Inference completed:

-   Prompt: 39 tokens in 3.88 s (10.1 tokens/s).
-   Generation: 8 tokens in 0.96 s (8.30 tokens/s).

### Forced Q8K path

With `Q8G16_DISABLE=1`:

-   Q8K dispatches: 15,003.
-   Q8G16 V2 dispatches: 0.
-   Inference completed.

Recorded short run: prompt 39 tokens in 3.90 s (10.0 tokens/s);
generation 8 tokens in 0.97 s (8.21 tokens/s).

### Forced FP32 path

With both `Q8G16_DISABLE=1` and `Q8K_DISABLE=1`:

-   FP32 dispatch trace observed for Q6_K operations.
-   Inference completed.

Recorded short run: prompt 39 tokens in 5.61 s (6.9 tokens/s);
generation 8 tokens in 1.75 s (4.58 tokens/s).

### Interpretation

These runs establish that the three CPU dispatch branches execute and
that short inference completed in each tested configuration. They are
**smoke tests**, not controlled A/B benchmarks. Their timings should not
be used to claim a precise full-model speedup.

The trace counts are per matmul dispatch event, not token counts or
expert counts.

## Hardware and software environment

The principal development and benchmark environment recorded for these
experiments:

  Component      Configuration
  -------------- ---------------------------------------------------
  CPU            Intel Core i5-13450HX
  CPU features   AVX2 and AVX-VNNI; no AVX512-VNNI
  GPU            NVIDIA GeForce RTX 3050 A Laptop GPU (4 GB class)
  OS             Fedora Linux
  Compiler       GCC 16.2.1
  CUDA toolkit   CUDA 13.2.86
  Build flags    `-O3 -march=native -fopenmp -std=c++17`
  Model          Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf

Performance and dispatch behavior can differ on other CPUs, GPUs,
drivers, compilers, memory configurations and model files.

## Build and run

The commands below reflect the recorded development environment. Adjust
CUDA paths and compiler settings for your system.

### CUDA-enabled build

``` bash
make clean
make CUDA=1 CUDA_HOME=/usr/local/cuda qwen3moe -j1
```

### Run a short inference

``` bash
./qwen3moe \
  -m /path/to/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf \
  -p "Explain how SIMD improves quantized matrix multiplication." \
  -n 8 -c 512 -b 1 -t 8 \
  --temp 0 --seed 42 --no-etf-collection
```

The exact command-line options depend on the current executable build.
Consult the source and help output for the authoritative option list.

## Configuration and dispatch controls

The following environment variables are diagnostic controls:

  -----------------------------------------------------------------------
  Variable                            Effect
  ----------------------------------- -----------------------------------
  `Q8G16_TRACE=1`                     Enables Q8G16/dispatch diagnostic
                                      traces

  `Q8G16_DISABLE=1`                   Disables Q8G16 so the dispatcher
                                      can exercise the secondary path

  `Q8K_DISABLE=1`                     Disables Q8K so the dispatcher can
                                      exercise the FP32 fallback

  `Q8G16_SHADOW=1`                    Enables sampled Q8G16 shadow
                                      comparison against the FP32
                                      reference
  -----------------------------------------------------------------------

For normal operation, leave the disable variables unset. Q8G16 is the
first-choice optimized CPU path when the operation and CPU are eligible.

Shadow and trace modes add diagnostic work and should not be used to
measure production performance.

## Current limitations

1.  **CPU feature dependency:** Q8G16's current dot-product path
    requires AVX-VNNI. Q8K also requires AVX-VNNI. A scalar or non-VNNI
    Q8K implementation would be needed to make Q8K a fallback on CPUs
    without AVX-VNNI.
2.  **Shape and format eligibility:** optimized CPU dispatch applies
    only to supported Q6_K tensor shapes and layouts. Other formats and
    unsupported shapes use the FP32 path.
3.  **Replay scope:** the reported replay benchmarks cover six captured
    operations and do not represent all layers, prompts or activation
    distributions.
4.  **Runtime benchmark scope:** the documented real-model runs are
    short smoke tests, not statistically controlled performance studies.
5.  **Shadow scope:** sampled shadow validation compares selected rows
    and tokens; it is not exhaustive full-tensor equivalence testing.
6.  **Research integration:** EAHMMF, PWEA and ETF have distinct runtime
    modes. The Q8 dispatch smoke tests used `--no-etf-collection`, so
    they do not validate the entire predictive-placement and telemetry
    feedback workflow.
7.  **Hardware scope:** the current performance measurements were
    collected on one development machine and should not be generalized
    to all hardware.
8.  **Release validation:** the complete regression suite, clean
    CPU-only build, clean CUDA build and EAHMMF stress tests must be run
    against the exact release commit before publishing the final v1.1.0
    tag.

## Release checklist

Before publishing the `v1.1.0` GitHub release:

-   [ ] Run the complete existing test and regression suite.
-   [ ] Verify a clean CUDA-enabled build.
-   [ ] Verify a clean CPU-only build.
-   [ ] Re-run Q8G16, Q8K and forced FP32 dispatch smoke tests.
-   [ ] Re-run numerical edge-case tests.
-   [ ] Re-run EAHMMF memory-management and migration invariants.
-   [ ] Confirm no diagnostic override is exported in the normal launch
    environment.
-   [ ] Review third-party licenses and model licensing separately.
-   [ ] Add the final repository URL and exact release commit.
-   [ ] Tag the verified commit as `v1.1.0`.

## Acknowledgements and lineage

This project continues the work documented in the original [EAHMMF
Inference Prototype
v0.1.0](https://github.com/jonny14-bro/EAHMMF_inference_prototype_V-0.1.0).

The two repositories are intended to remain independently available so
readers can examine the prototype and the subsequent engineering
evolution.

**EAHMMF Inference Engine v1.1.0** is a research and engineering
project. Benchmark results are tied to the stated test setup and should
be reproduced before being used for external performance claims.
