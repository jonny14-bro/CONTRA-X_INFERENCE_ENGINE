# Q6 Real Workload Benchmark

The real captured Q6 workload benchmarks are intentionally separated
from the normal quick, stress and full test modes.

Run from the repository root:

- `benchmarks/q6_central/real_workload_benchmark.sh --check`
- `benchmarks/q6_central/real_workload_benchmark.sh --replay`
- `benchmarks/q6_central/real_workload_benchmark.sh --attribution`
- `benchmarks/q6_central/real_workload_benchmark.sh --all`

`--check` validates dataset presence and compiles the three diagnostics.
It does not execute the large replay.

`--replay` runs the Q8K three-way and Q8G16 five-way comparisons.

`--attribution` runs the dataset-wide zeroing/rounding/total weighted
error diagnostic. The diagnostic currently writes to a fixed output
directory; the runner archives the previous output and copies each
new result into its timestamped run directory.

`--all` executes replay followed by attribution.

Thread count can be changed with `Q6_BENCH_THREADS`, defaulting to 8.

Timestamped reports and raw logs are stored under:
`benchmarks/q6_central/reports/real_workload/`

The existing `benchmark_suite --quick`, `--stress`, `--full`, and
`--report-only` modes remain unchanged.
