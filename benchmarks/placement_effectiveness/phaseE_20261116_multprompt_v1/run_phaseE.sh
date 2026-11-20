#!/usr/bin/env bash

set -uo pipefail

ROOT="benchmarks/placement_effectiveness/phaseE_20261116_multprompt_v1"
MODEL="${CONTRAX_MODEL:-${HOME}/Downloads/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf}"
EXPECTED_MODEL_SHA="eb331a4eee8eb6b5a8eb25f44f96f45c71b8d10f553c0a456190dd590a7ef77d"
EXPECTED_MANIFEST_SHA="fff9e6e45f5d901528287f6059637f29df2c77437cbbeaee1584ca7289f16976"

RUNS="$ROOT/runs"
RUN_LOG="$ROOT/runner.log"

if [[ -e "$RUNS" || -e "$RUN_LOG" ]]; then
    echo "STOP: runs/ or runner.log already exists."
    exit 1
fi

if [[ ! -x ./qwen3moe ]]; then
    echo "STOP: ./qwen3moe is missing or not executable."
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "STOP: model file missing."
    exit 1
fi

actual_manifest_sha="$(sha256sum "$ROOT/workload_manifest.json" | awk '{print $1}')"
actual_model_sha="$(sha256sum "$MODEL" | awk '{print $1}')"

if [[ "$actual_manifest_sha" != "$EXPECTED_MANIFEST_SHA" ]]; then
    echo "STOP: workload manifest hash mismatch."
    exit 1
fi

if [[ "$actual_model_sha" != "$EXPECTED_MODEL_SHA" ]]; then
    echo "STOP: model hash mismatch."
    exit 1
fi

if [[ "$(find "$ROOT/prompts" -maxdepth 1 -name 'prompt_*.txt' | wc -l)" -ne 12 ]]; then
    echo "STOP: expected exactly 12 prompt files."
    exit 1
fi

mkdir -p "$RUNS"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "PHASE E MULTI-PROMPT BENCHMARK"
echo "Started: $(date --iso-8601=seconds)"
echo "Manifest SHA256: $actual_manifest_sha"
echo "Model SHA256: $actual_model_sha"
echo "Executable SHA256: $(sha256sum ./qwen3moe | awk '{print $1}')"
echo "Prompts: 12 | Categories: 4 | Repetitions: 3"
echo "Measured runs: 72 | Warmups: 2"

{
    echo "kernel: $(uname -r)"
    echo "driver:"
    nvidia-smi
    echo "disk:"
    df -h "$ROOT"
} > "$ROOT/environment.txt"

run_case() {
    local mode="$1"
    local label="$2"
    local prompt_file="$3"
    local dir="$4"

    mkdir -p "$dir"

    local prompt
    prompt="$(cat "$prompt_file")"

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
        -p "$prompt"
        -n 16
        --raw
    )

    echo
    echo "RUN $label MODE=$mode"
    echo "Prompt file: $prompt_file"
    echo "Prompt SHA256: $(sha256sum "$prompt_file" | awk '{print $1}')"

    if [[ "$mode" == "pwea" ]]; then
        ./qwen3moe "${common[@]}" \
            --pwea-placement \
            --pwea-prediction-only \
            --pwea-max-migrations 16 \
            > "$dir/run.log" 2>&1
    else
        ./qwen3moe "${common[@]}" \
            > "$dir/run.log" 2>&1
    fi

    local rc=$?
    echo "$rc" > "$dir/exit_status.txt"

    grep -E \
        '^\[prompt |^\[ETF\]|^\[expert residency\]|^  serving:|^cuda-compute:|^\[PWEA->EAHMMF\] evaluated=' \
        "$dir/run.log" || true

    if [[ "$rc" -ne 0 ]]; then
        echo "FAILED: $label mode=$mode exit=$rc"
        tail -n 60 "$dir/run.log"
        return "$rc"
    fi

    if [[ ! -s "$dir/stats.jsonl" || ! -s "$dir/etf.jsonl" ]]; then
        echo "FAILED: missing stats or ETF output for $label"
        return 1
    fi

    python - "$dir/stats.jsonl" "$dir/etf.jsonl" <<'PY'
import json
import sys
from pathlib import Path

stats = [
    json.loads(x)
    for x in Path(sys.argv[1]).read_text().splitlines()
    if x.strip()
]
etf = [
    json.loads(x)
    for x in Path(sys.argv[2]).read_text().splitlines()
    if x.strip()
]

if len(stats) != 1 or not stats[0].get("completed"):
    raise SystemExit("FAIL: stats record missing or incomplete")

if len(etf) != 1:
    raise SystemExit("FAIL: expected exactly one ETF record")

record = etf[0]
if record.get("window", {}).get("dropped_events") != 0:
    raise SystemExit("FAIL: ETF dropped events detected")

if not record.get("activations"):
    raise SystemExit("FAIL: ETF activation list is empty")

print(
    "RUN VALIDATION: PASS | "
    f"completed={stats[0]['completed']} | "
    f"ETF events={len(record['activations'])} | "
    f"dropped={record['window']['dropped_events']}"
)
PY
    local validation_rc=$?

    if [[ "$validation_rc" -ne 0 ]]; then
        echo "FAILED: output validation for $label"
        return "$validation_rc"
    fi

    echo "PASS: $label mode=$mode"
    return 0
}

echo
echo "========== WARM-UP RUNS =========="

run_case ssd warmup_ssd \
    "$ROOT/prompts/prompt_01.txt" "$RUNS/warmup/ssd" || exit 1

run_case pwea warmup_pwea \
    "$ROOT/prompts/prompt_01.txt" "$RUNS/warmup/pwea" || exit 1

echo
echo "========== MEASURED PAIRED RUNS =========="

for prompt_index in $(seq 1 12); do
    printf -v prompt_id "%02d" "$prompt_index"
    prompt_file="$ROOT/prompts/prompt_${prompt_id}.txt"

    for repetition in 1 2 3; do
        printf -v rep_id "%02d" "$repetition"
        pair_dir="$RUNS/prompt_${prompt_id}/rep_${rep_id}"

        echo
        echo "######## PROMPT $prompt_id REPETITION $rep_id ########"

        # Alternate first-run order across prompt/repetition pairs.
        if (( (prompt_index + repetition) % 2 == 0 )); then
            run_case ssd "p${prompt_id}_r${rep_id}_ssd" \
                "$prompt_file" "$pair_dir/ssd" || exit 1

            run_case pwea "p${prompt_id}_r${rep_id}_pwea" \
                "$prompt_file" "$pair_dir/pwea" || exit 1
        else
            run_case pwea "p${prompt_id}_r${rep_id}_pwea" \
                "$prompt_file" "$pair_dir/pwea" || exit 1

            run_case ssd "p${prompt_id}_r${rep_id}_ssd" \
                "$prompt_file" "$pair_dir/ssd" || exit 1
        fi
    done
done

echo
echo "========== PHASE E BENCHMARK COMPLETE =========="
echo "Completed: $(date --iso-8601=seconds)"
echo "Expected measured runs: 72"
echo "Expected warmups: 2"
echo "Results: $RUNS"
