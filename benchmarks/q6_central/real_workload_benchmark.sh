#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(pwd)"
CENTRAL="benchmarks/q6_central"
INTEGRATION="benchmarks/q6_integration_20261118"
DATASET="benchmarks/real_threeway_validation_20261120"
BIN="$CENTRAL/bin"
REPORTS="$CENTRAL/reports/real_workload"
BUILDLOG="$REPORTS/build_logs"

WEIGHTS="benchmarks/real_activation_weight_replay_20261118/weights.bin"
ACTIVATIONS="benchmarks/real_activations_q8k_full_20261118/activations.f32"
DESCRIPTORS="$DATASET/q6_only_descriptors.tsv"

THREADS="${Q6_BENCH_THREADS:-8}"
CXX="${CXX:-g++}"

usage() {
    cat <<'EOF'
Q6 Real Workload Benchmark

Usage:
  benchmarks/q6_central/real_workload_benchmark.sh --check
  benchmarks/q6_central/real_workload_benchmark.sh --replay
  benchmarks/q6_central/real_workload_benchmark.sh --attribution
  benchmarks/q6_central/real_workload_benchmark.sh --all

Modes:
  --check        Validate data and compile diagnostic executables.
  --replay       Run Q8K three-way and Q8G16 five-way replay.
  --attribution  Run dataset-wide weighted-error attribution.
  --all          Run replay followed by weighted-error attribution.

Environment:
  Q6_BENCH_THREADS=8
  CXX=g++

The real-workload modes are separate from --quick, --stress and --full.
EOF
}

MODE="${1:---help}"

if [[ "$MODE" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "$MODE" != "--check" &&
      "$MODE" != "--replay" &&
      "$MODE" != "--attribution" &&
      "$MODE" != "--all" ]]; then
    usage
    exit 2
fi

if [[ ! "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Q6_BENCH_THREADS must be a positive integer."
    exit 2
fi

for f in "$WEIGHTS" "$ACTIVATIONS" "$DESCRIPTORS"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing dataset file: $f"
        exit 1
    fi
done

RECORDS="$(wc -l < "$DESCRIPTORS")"

if [[ "$RECORDS" -ne 5000 ]]; then
    echo "ERROR: Expected 5000 Q6 descriptors, found $RECORDS."
    exit 1
fi

echo "===== DATASET VALIDATION ====="
echo "Records: $RECORDS"
stat -c 'Weights:     %s bytes' "$WEIGHTS"
stat -c 'Activations: %s bytes' "$ACTIVATIONS"
stat -c 'Descriptors: %s bytes' "$DESCRIPTORS"

PWEA=(
    pwea/native/src_pwea_unicode.cpp
    pwea/native/src_pwea_features.cpp
    pwea/native/src_pwea_model.cpp
    pwea/native/src_pwea_predictor.cpp
)

for f in "${PWEA[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing dependency: $f"
        exit 1
    fi
done

mkdir -p "$BIN" "$BUILDLOG"

FLAGS=(
    -O3
    -march=native
    -fopenmp
    -std=c++17
    -Wall
    -Wextra
    -Wno-unused-parameter
    -I.
    -Ipwea/native/include
)

BUILD_ID="$(date +%Y%m%d_%H%M%S)"

build_one() {
    local name="$1"
    local source="$2"
    local log="$BUILDLOG/${BUILD_ID}_${name}.log"

    echo
    echo "===== BUILDING $name ====="

    if ! "$CXX" "${FLAGS[@]}" \
        -o "$BIN/$name" \
        "$source" "${PWEA[@]}" > "$log" 2>&1; then
        cat "$log"
        echo "ERROR: Build failed: $name"
        return 1
    fi

    echo "BUILD PASS: $name"
    echo "Binary: $BIN/$name"
    echo "Log: $log"
}

build_one compare_q6_threeway \
    "$INTEGRATION/compare_q6_threeway.cpp"

build_one compare_q6_fiveway \
    "$INTEGRATION/compare_q6_fiveway.cpp"

build_one diagnose_q8g16_dataset_weighted_error \
    "$INTEGRATION/diagnose_q8g16_dataset_weighted_error.cpp"

echo
echo "===== BUILD VALIDATION ====="

for name in \
    compare_q6_threeway \
    compare_q6_fiveway \
    diagnose_q8g16_dataset_weighted_error
do
    if [[ ! -x "$BIN/$name" ]]; then
        echo "ERROR: Executable unavailable: $name"
        exit 1
    fi
    echo "PASS: $name"
done

if [[ "$MODE" == "--check" ]]; then
    echo
    echo "CHECK COMPLETE"
    echo "All three real-workload diagnostic executables compiled."
    echo "No dataset replay was executed."
    exit 0
fi

RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPORTS/$RUN_ID"

mkdir -p "$RUN_DIR"

{
    echo "Q6 REAL WORKLOAD BENCHMARK"
    echo "Run: $RUN_ID"
    echo "Mode: $MODE"
    echo "Date: $(date --iso-8601=seconds)"
    echo "Threads: $THREADS"
    echo "Compiler: $("$CXX" --version | head -n 1)"
    echo "Kernel: $(uname -r)"
    echo "Architecture: $(uname -m)"
    echo "Git revision: $(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "Git status: $(git status --short 2>/dev/null | wc -l) modified/untracked entries"
    echo "Records: $RECORDS"
    echo "Weights bytes: $(stat -c %s "$WEIGHTS")"
    echo "Activations bytes: $(stat -c %s "$ACTIVATIONS")"
    echo "Descriptors SHA256: $(sha256sum "$DESCRIPTORS" | cut -d' ' -f1)"
    echo
    echo "EXECUTABLE HASHES"
    sha256sum "$BIN/compare_q6_threeway"
    sha256sum "$BIN/compare_q6_fiveway"
    sha256sum "$BIN/diagnose_q8g16_dataset_weighted_error"
} > "$RUN_DIR/manifest.txt"

run_replay() {
    echo
    echo "===== Q8K THREE-WAY REPLAY ====="

    if ! "$BIN/compare_q6_threeway" \
        "$WEIGHTS" "$ACTIVATIONS" "$DESCRIPTORS" "$THREADS" \
        > "$RUN_DIR/q8k_threeway.log" 2>&1; then
        cat "$RUN_DIR/q8k_threeway.log"
        echo "FAIL: Q8K three-way replay"
        return 1
    fi

    tail -n 25 "$RUN_DIR/q8k_threeway.log"
    echo "PASS: Q8K three-way replay"

    echo
    echo "===== Q8G16 FIVE-WAY REPLAY ====="

    if ! "$BIN/compare_q6_fiveway" \
        "$WEIGHTS" "$ACTIVATIONS" "$DESCRIPTORS" "$THREADS" \
        > "$RUN_DIR/q8g16_fiveway.log" 2>&1; then
        cat "$RUN_DIR/q8g16_fiveway.log"
        echo "FAIL: Q8G16 five-way replay"
        return 1
    fi

    tail -n 25 "$RUN_DIR/q8g16_fiveway.log"
    echo "PASS: Q8G16 five-way replay"
}

run_attribution() {
    local fixed="$DATASET/weighted_error_dataset_20261120"

    echo
    echo "===== WEIGHTED-ERROR ATTRIBUTION ====="

    # Preserve the previous result before the diagnostic writes to
    # its currently hard-coded output directory.
    if [[ -d "$fixed" ]]; then
        cp -a "$fixed" "$RUN_DIR/attribution_previous"
        echo "Previous attribution report archived."
    fi

    if ! "$BIN/diagnose_q8g16_dataset_weighted_error" \
        "$WEIGHTS" "$ACTIVATIONS" "$DESCRIPTORS" "$THREADS" \
        > "$RUN_DIR/attribution.log" 2>&1; then
        cat "$RUN_DIR/attribution.log"
        echo "FAIL: Weighted-error attribution"
        return 1
    fi

    if [[ ! -s "$fixed/summary.json" ]]; then
        echo "ERROR: Attribution summary.json missing or empty."
        return 1
    fi

    cp -a "$fixed" "$RUN_DIR/attribution"

    tail -n 35 "$RUN_DIR/attribution.log"
    echo "PASS: Weighted-error attribution"
    echo "Artifacts: $RUN_DIR/attribution"
}

case "$MODE" in
    --replay)
        run_replay
        ;;
    --attribution)
        run_attribution
        ;;
    --all)
        run_replay
        run_attribution
        ;;
esac

{
    echo
    echo "Completed: $(date --iso-8601=seconds)"
    echo "Status: PASS"
    echo "Report directory: $RUN_DIR"
} >> "$RUN_DIR/manifest.txt"

echo
echo "===== REAL WORKLOAD BENCHMARK COMPLETE ====="
echo "Mode: $MODE"
echo "Report: $RUN_DIR"
echo "Manifest: $RUN_DIR/manifest.txt"
