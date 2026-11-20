#!/usr/bin/env bash
set -u

DIR="benchmarks/q6_integration_20261118"
OUT="benchmarks/q6_python_quality_20261120"
MODEL="$HOME/Downloads/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf"

mkdir -p "$OUT"

PROMPT='Write a concise complete Python 3 task manager. Define a Task class with title, priority and completed fields. Implement add_task, complete_task, remove_task and display_pending functions. Validate inputs, sort pending tasks by priority, and include a short main demonstration. Use only the standard library. Output executable Python code only.'

run_one() {
    local name="$1"
    local binary="$2"
    local rep="$3"
    local tag="${name}_rep${rep}"

    echo "===== $tag ====="

    /usr/bin/time -v -o "$OUT/$tag.time" \
        "$binary" \
        -m "$MODEL" \
        -c 512 -b 4 -t 8 -n 256 \
        -p "$PROMPT" \
        --temp 0 --top-k 1 --seed 42 \
        --device sim --expert-tier ssd \
        --no-etf-collection \
        > "$OUT/$tag.stdout" \
        2> "$OUT/$tag.stderr"

    local rc=$?
    echo "Exit: $rc"
    grep -E '\[prompt ' "$OUT/$tag.stderr" || true
    grep -E 'Elapsed \(wall clock\)|Maximum resident|Major \(requiring I/O\)|Exit status' \
        "$OUT/$tag.time" || true
    sha256sum "$OUT/$tag.stdout"
    return 0
}

# Alternating order; page cache is already warm from previous experiment.
run_one fallback "$DIR/qwen3moe_q6_fallback" 1
run_one vnni     "$DIR/qwen3moe_q6_vnni" 1
run_one vnni     "$DIR/qwen3moe_q6_vnni" 2
run_one fallback "$DIR/qwen3moe_q6_fallback" 2

echo
echo "===== OUTPUT SIZES ====="
wc -c "$OUT"/*.stdout
