#!/usr/bin/env bash

set -uo pipefail

ROOT="benchmarks/placement_effectiveness/cpu_compute_ablation_20261117"
SOURCE_ROOT="benchmarks/placement_effectiveness/phaseE_20261116_multprompt_v1"
MODEL="${CONTRAX_MODEL:-${HOME}/Downloads/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf}"

EXPECTED_MODEL_SHA="eb331a4eee8eb6b5a8eb25f44f96f45c71b8d10f553c0a456190dd590a7ef77d"
EXPECTED_MANIFEST_SHA="fff9e6e45f5d901528287f6059637f29df2c77437cbbeaee1584ca7289f16976"

RUNS="$ROOT/runs"
RUN_LOG="$ROOT/runner.log"
MANIFEST="$SOURCE_ROOT/workload_manifest.json"

if [[ -e "$RUNS" || -e "$RUN_LOG" ]]; then
    echo "STOP: runs/ or runner.log already exists."
    echo "Do not overwrite existing experiment evidence."
    exit 1
fi

if [[ ! -x ./qwen3moe ]]; then
    echo "STOP: ./qwen3moe missing or not executable."
    exit 1
fi

if [[ ! -f "$MODEL" || ! -f "$MANIFEST" ]]; then
    echo "STOP: model or manifest missing."
    exit 1
fi

actual_model_sha="$(sha256sum "$MODEL" | awk '{print $1}')"
actual_manifest_sha="$(sha256sum "$MANIFEST" | awk '{print $1}')"

if [[ "$actual_model_sha" != "$EXPECTED_MODEL_SHA" ]]; then
    echo "STOP: model SHA256 mismatch."
    exit 1
fi

if [[ "$actual_manifest_sha" != "$EXPECTED_MANIFEST_SHA" ]]; then
    echo "STOP: manifest SHA256 mismatch."
    exit 1
fi

python - "$SOURCE_ROOT" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
prompts = manifest.get("prompts", [])

if len(prompts) != 12:
    raise SystemExit(f"STOP: expected 12 prompts, found {len(prompts)}")

if manifest.get("prompt_count") != 12:
    raise SystemExit("STOP: manifest prompt_count is not 12")

for index, item in enumerate(prompts, 1):
    path = source / "prompts" / f"prompt_{index:02d}.txt"

    if not path.is_file():
        raise SystemExit(f"STOP: missing {path}")

    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = item.get("prompt_sha256")

    if actual != expected:
        raise SystemExit(
            f"STOP: prompt {index:02d} SHA256 mismatch: "
            f"{actual} != {expected}"
        )

print("Manifest validation: PASS")
print("Prompt count: 12")
print("All 12 prompt SHA256 values: PASS")
PY

if [[ $? -ne 0 ]]; then
    exit 1
fi

mkdir -p "$RUNS"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "CPU-ONLY PLACEMENT ABLATION"
echo "Started: $(date --iso-8601=seconds)"
echo "Model SHA256: $actual_model_sha"
echo "Manifest SHA256: $actual_manifest_sha"
echo "Executable SHA256: $(sha256sum ./qwen3moe | awk '{print $1}')"
echo "Expert compute: CPU ONLY"
echo "CUDA device: ENABLED FOR MEMORY PLACEMENT"
echo "CUDA expert matmul: DISABLED"
echo "Prompts: 12 | Repetitions: 3 | Warmups: 2"
echo "Measured runs: 72"
echo "PWEA max migrations: 16"

{
    echo "experiment=cpu_compute_ablation_20261117"
    echo "expert_compute=CPU_ONLY"
    echo "cuda_device=enabled_for_memory_placement"
    echo "cuda_expert_matmul=disabled"
    echo "model_sha256=$actual_model_sha"
    echo "manifest_sha256=$actual_manifest_sha"
    echo "executable_sha256=$(sha256sum ./qwen3moe | awk '{print $1}')"
    echo "kernel=$(uname -r)"
    echo "date=$(date --iso-8601=seconds)"
    echo
    nvidia-smi || true
    echo
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
    echo "Expert compute mode: CPU_ONLY"

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
        '^\[prompt |^\[ETF\]|^\[expert residency\]|^  serving:|^\[PWEA->EAHMMF\] evaluated=' \
        "$dir/run.log" || true

    if [[ "$rc" -ne 0 ]]; then
        echo "FAILED: $label exit=$rc"
        tail -n 60 "$dir/run.log"
        return "$rc"
    fi

    if [[ ! -s "$dir/stats.jsonl" || ! -s "$dir/etf.jsonl" ]]; then
        echo "FAILED: missing stats or ETF output."
        return 1
    fi

    if grep -Eq 'cuda-compute:|gpu_calls=|pair_calls=' "$dir/run.log"; then
        echo "FAILED: unexpected CUDA expert-compute telemetry."
        return 1
    fi

    python - "$mode" "$dir/stats.jsonl" "$dir/etf.jsonl" <<'PY'
import json
import re
import sys
from pathlib import Path

mode = sys.argv[1]
stats_rows = [
    json.loads(x)
    for x in Path(sys.argv[2]).read_text().splitlines()
    if x.strip()
]
etf_rows = [
    json.loads(x)
    for x in Path(sys.argv[3]).read_text().splitlines()
    if x.strip()
]

if len(stats_rows) != 1 or not stats_rows[0].get("completed"):
    raise SystemExit("FAIL: stats missing or incomplete")

if len(etf_rows) != 1:
    raise SystemExit("FAIL: expected exactly one ETF record")

stats = stats_rows[0]
etf = etf_rows[0]

if etf.get("window", {}).get("dropped_events") != 0:
    raise SystemExit("FAIL: ETF dropped events detected")

if not etf.get("activations"):
    raise SystemExit("FAIL: ETF activation list empty")

s2 = stats.get("S2", {})
counters = s2.get("counters", {})
acquires = counters.get("acquires", 0)
successful = counters.get("successful_acquires", 0)
failed = counters.get("failed_acquires", 0)
served = counters.get("served_by_tier_ssd_ram_vram", [])

if failed != 0 or successful != acquires:
    raise SystemExit("FAIL: expert acquisition accounting")

if len(served) != 3 or sum(served) != acquires:
    raise SystemExit("FAIL: expert serving accounting")

s1 = stats.get("S1", {}).get("counters", {})
migrations = s1.get("migration_count_by_path", [])
migration_failures = s1.get("migration_failures_by_path", [])

if len(migrations) != 5 or len(migration_failures) != 5:
    raise SystemExit("FAIL: unexpected migration counter schema")

if mode == "ssd":
    if sum(migrations) != 0:
        raise SystemExit("FAIL: SSD baseline unexpectedly migrated experts")
    if sum(migration_failures) != 0:
        raise SystemExit("FAIL: SSD baseline migration failures detected")
else:
    # Read the unique promotion count from the PWEA execution summary.
    # Path counters can count both SSD->RAM and RAM->VRAM transitions.
    log_path = Path(sys.argv[2]).parent / "run.log"
    log = log_path.read_text(errors="replace")

    matches = re.findall(
        r"attempted=(\d+)\s+migrated=(\d+)\s+failed=(\d+)",
        log
    )

    if len(matches) != 1:
        raise SystemExit(
            f"FAIL: expected one PWEA execution summary, found {len(matches)}"
        )

    attempted, migrated, failed = map(int, matches[0])

    if attempted > 16:
        raise SystemExit("FAIL: PWEA exceeded the 16-attempt limit")
    if migrated != attempted:
        raise SystemExit(
            f"FAIL: attempted={attempted}, but migrated={migrated}"
        )
    if failed != 0:
        raise SystemExit(f"FAIL: PWEA reported {failed} migration failures")

    if sum(migration_failures) != 0:
        raise SystemExit("FAIL: migration failure counters are nonzero")

    # Check post-placement residency and post-inference residency.
    for stage in ("S1", "S2"):
        experts = stats.get(stage, {}).get("experts", {})
        total = sum(experts.get(k, 0) for k in ("ssd", "ram", "vram"))

        if total != 6144:
            raise SystemExit(
                f"FAIL: {stage} residency totals {total}, expected 6144"
            )
        if experts.get("vram", 0) != migrated:
            raise SystemExit(
                f"FAIL: {stage} VRAM residency does not match "
                f"successful PWEA promotions ({migrated})"
            )
        if experts.get("transitional", 0) != 0:
            raise SystemExit(
                f"FAIL: {stage} has transitional experts remaining"
            )

timing = stats.get("timing_seconds", {})
for key in ("inference", "total"):
    if key not in timing or timing[key] <= 0:
        raise SystemExit(f"FAIL: invalid {key} timing")

print(
    "RUN VALIDATION: PASS | "
    f"mode={mode} | acquires={acquires} | "
    f"served={served} | ETF={len(etf['activations'])} | "
    f"migrations={sum(migrations)}"
)
PY

    local validation_rc=$?

    if [[ "$validation_rc" -ne 0 ]]; then
        echo "FAILED: structured validation for $label"
        return "$validation_rc"
    fi

    echo "PASS: $label mode=$mode"
    return 0
}

echo
echo "========== WARM-UP RUNS =========="

run_case ssd warmup_ssd \
    "$SOURCE_ROOT/prompts/prompt_01.txt" \
    "$RUNS/warmup/ssd" || exit 1

run_case pwea warmup_pwea \
    "$SOURCE_ROOT/prompts/prompt_01.txt" \
    "$RUNS/warmup/pwea" || exit 1

echo
echo "========== MEASURED PAIRED RUNS =========="

for prompt_index in $(seq 1 12); do
    printf -v prompt_id "%02d" "$prompt_index"
    prompt_file="$SOURCE_ROOT/prompts/prompt_${prompt_id}.txt"

    for repetition in 1 2 3; do
        printf -v rep_id "%02d" "$repetition"
        pair_dir="$RUNS/prompt_${prompt_id}/rep_${rep_id}"

        echo
        echo "######## PROMPT $prompt_id REPETITION $rep_id ########"

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
echo "========== RUN COUNT VALIDATION =========="

stats_count="$(find "$RUNS" -name stats.jsonl -type f | wc -l)"

echo "Stats files: $stats_count"
if [[ "$stats_count" -ne 74 ]]; then
    echo "STOP: expected 74 stats files (72 measured + 2 warmups)."
    exit 1
fi

echo
echo "CPU-ONLY ABLATION COMPLETE"
echo "Completed: $(date --iso-8601=seconds)"
echo "Measured pairs: 36"
echo "Warmups: 2"
echo "Results: $RUNS"
