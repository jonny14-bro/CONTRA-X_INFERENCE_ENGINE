#!/usr/bin/env bash

set -u

MODEL="${CONTRAX_MODEL:-${HOME}/Downloads/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf}"
PROMPT="Explain how mixture-of-experts routing works and why expert placement in GPU memory can improve inference performance."

ROOT="benchmarks/placement_effectiveness"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/repeated_$RUN_ID"

mkdir -p "$OUT"

echo "Benchmark directory: $OUT"

run_case() {
    local mode="$1"
    local label="$2"
    local dir="$OUT/$label"

    mkdir -p "$dir"

    echo
    echo "=================================================="
    echo "RUN: $label | MODE: $mode"
    echo "=================================================="

    local -a common=(
        -m "$MODEL"
        --device cuda
        --cuda-compute
        --expert-tier ssd
        --expert-policy lru
        --ram-budget-mb 1024
        --vram-budget-mb 512
        --vram-reserve-mb 512
        --etf-log "$dir/etf.jsonl"
        --expert-log "$dir/experts.log"
        --expert-stats
        --expert-stats-per-prompt
        --expert-stats-per-prompt-file "$dir/stats.jsonl"
        --seed 20261116
        --temp 0
        -p "$PROMPT"
        -n 16
        --raw
    )

    if [[ "$mode" == "pwea" ]]; then
        ./qwen3moe "${common[@]}" \
            --pwea-placement \
            --pwea-prediction-only \
            --pwea-max-migrations 16 \
            2>&1 | tee "$dir/run.log"
    else
        ./qwen3moe "${common[@]}" \
            2>&1 | tee "$dir/run.log"
    fi

    local rc=${PIPESTATUS[0]}
    echo "$rc" > "$dir/exit_status.txt"

    echo "$label EXIT STATUS: $rc"

    if [[ "$rc" -ne 0 ]]; then
        echo "ERROR: benchmark stopped because $label failed."
        return "$rc"
    fi

    return 0
}

echo
echo "========== WARM-UP =========="

run_case ssd warmup_ssd || exit 1
run_case pwea warmup_pwea || exit 1

echo
echo "========== FIVE PAIRED ROUNDS =========="

for round in 1 2 3 4 5; do
    echo
    echo "######## ROUND $round ########"

    if (( round % 2 == 1 )); then
        run_case ssd "round_${round}_ssd" || exit 1
        run_case pwea "round_${round}_pwea" || exit 1
    else
        run_case pwea "round_${round}_pwea" || exit 1
        run_case ssd "round_${round}_ssd" || exit 1
    fi
done

echo
echo "========== BENCHMARK COMPLETE =========="
echo "Results: $OUT"
echo "Warm-ups: 2"
echo "Measured runs: 10"
echo "Successful completion of all runs."
