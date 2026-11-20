import json
import re
import statistics
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(
    "benchmarks/placement_effectiveness/"
    "phaseE_20261116_multprompt_v1"
)
RUNS = ROOT / "runs"
MANIFEST = ROOT / "workload_manifest.json"

def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

def mean(values):
    return statistics.mean(values) if values else 0.0

def median(values):
    return statistics.median(values) if values else 0.0

def pct_change(base, new):
    return ((base - new) / base * 100.0) if base else 0.0

def parse_cuda(log_path):
    text = log_path.read_text(errors="replace")
    matches = re.findall(
        r"gpu_calls=(\d+)\s+cpu_fallbacks=(\d+)"
        r"\s+pair_calls=(\d+)",
        text
    )
    if not matches:
        return None
    return tuple(map(int, matches[-1]))

def routing_comparison(a, b):
    aa = a.get("activations", [])
    bb = b.get("activations", [])

    if len(aa) != len(bb):
        return {
            "same_length": False,
            "same_routes": False,
            "max_weight_diff": None,
        }

    same_routes = True
    max_diff = 0.0

    for x, y in zip(aa, bb):
        if (
            x.get("layer") != y.get("layer")
            or x.get("token_position") != y.get("token_position")
            or x.get("expert_id") != y.get("expert_id")
        ):
            same_routes = False

        wx = float(x.get("routing_weight", 0.0))
        wy = float(y.get("routing_weight", 0.0))
        max_diff = max(max_diff, abs(wx - wy))

    return {
        "same_length": True,
        "same_routes": same_routes,
        "max_weight_diff": max_diff,
    }

manifest = json.loads(MANIFEST.read_text())
prompts = manifest["prompts"]

if len(prompts) != 12:
    raise SystemExit(f"Expected 12 manifest prompts, got {len(prompts)}")

records = []
errors = []

for index, prompt in enumerate(prompts, start=1):
    prompt_dir = RUNS / f"prompt_{index:02d}"
    category = prompt["category"]
    session = prompt["session_id"]

    for rep in range(1, 4):
        pair_dir = prompt_dir / f"rep_{rep:02d}"
        pair = {}

        for mode in ("ssd", "pwea"):
            path = pair_dir / mode

            try:
                stats_rows = read_jsonl(path / "stats.jsonl")
                etf_rows = read_jsonl(path / "etf.jsonl")
                stats = stats_rows[0]
                etf = etf_rows[0]

                if len(stats_rows) != 1:
                    raise ValueError("stats record count != 1")
                if len(etf_rows) != 1:
                    raise ValueError("ETF record count != 1")
                if not stats.get("completed"):
                    raise ValueError("prompt not completed")
                if etf.get("window", {}).get("dropped_events") != 0:
                    raise ValueError("ETF dropped events")
                if not etf.get("activations"):
                    raise ValueError("no ETF activations")

                cuda = parse_cuda(path / "run.log")

                pair[mode] = {
                    "stats": stats,
                    "etf": etf,
                    "cuda": cuda,
                }

            except Exception as exc:
                errors.append(
                    f"prompt_{index:02d}/rep_{rep:02d}/{mode}: {exc}"
                )

        if len(pair) != 2:
            continue

        ssd = pair["ssd"]
        pwea = pair["pwea"]
        st = ssd["stats"]
        pt = pwea["stats"]

        sr = st["S2"]["counters"]
        pr = pt["S2"]["counters"]

        route = routing_comparison(ssd["etf"], pwea["etf"])

        records.append({
            "prompt": index,
            "category": category,
            "session": session,
            "rep": rep,

            "ssd_inference": st["timing_seconds"]["inference"],
            "pwea_inference": pt["timing_seconds"]["inference"],
            "ssd_total": st["timing_seconds"]["total"],
            "pwea_total": pt["timing_seconds"]["total"],
            "prediction": pt["timing_seconds"]["prediction"],
            "placement": pt["timing_seconds"]["placement"],

            "inference_improvement_pct": pct_change(
                st["timing_seconds"]["inference"],
                pt["timing_seconds"]["inference"]
            ),
            "total_improvement_pct": pct_change(
                st["timing_seconds"]["total"],
                pt["timing_seconds"]["total"]
            ),

            "ssd_acquires": sr["acquires"],
            "pwea_acquires": pr["acquires"],
            "ssd_failed": sr["failed_acquires"],
            "pwea_failed": pr["failed_acquires"],

            "ssd_served": sr["served_by_tier_ssd_ram_vram"],
            "pwea_served": pr["served_by_tier_ssd_ram_vram"],

            "pwea_final_experts": pt["S2"]["experts"],
            "pwea_migrations": pt["S1"]["counters"][
                "migration_count_by_path"
            ],
            "pwea_migration_failures": pt["S1"]["counters"][
                "migration_failures_by_path"
            ],
            "pwea_migration_bytes": pt["S1"]["counters"][
                "migration_bytes_by_path"
            ],
            "pwea_migration_duration_us": pt["S1"]["counters"][
                "migration_duration_us_by_path"
            ],

            "ssd_cuda": ssd["cuda"],
            "pwea_cuda": pwea["cuda"],

            "ssd_etf_events": len(ssd["etf"]["activations"]),
            "pwea_etf_events": len(pwea["etf"]["activations"]),
            "ssd_dropped": ssd["etf"]["window"]["dropped_events"],
            "pwea_dropped": pwea["etf"]["window"]["dropped_events"],
            "routing": route,
        })

print("\n========== PHASE E VALIDATION ==========")
print("Expected pairs: 36")
print("Valid pairs:", len(records))
print("Validation errors:", len(errors))

for error in errors:
    print("ERROR:", error)

if len(records) != 36 or errors:
    raise SystemExit("STOP: benchmark integrity validation failed")

# Run-level paired summaries.
for field, label in (
    ("inference_improvement_pct", "Inference improvement"),
    ("total_improvement_pct", "End-to-end improvement"),
):
    values = [r[field] for r in records]
    print(f"\n{label}:")
    print(f"  Mean:   {mean(values):.4f}%")
    print(f"  Median: {median(values):.4f}%")
    print(f"  Min:    {min(values):.4f}%")
    print(f"  Max:    {max(values):.4f}%")
    print(f"  PWEA faster: {sum(v > 0 for v in values)}/36")

print("\n========== CATEGORY RESULTS ==========")
categories = sorted(set(r["category"] for r in records))

for category in categories:
    subset = [r for r in records if r["category"] == category]
    print(f"\n{category} ({len(subset)} paired runs)")
    print(
        "  Inference improvement:",
        f"{mean([r['inference_improvement_pct'] for r in subset]):.4f}%"
    )
    print(
        "  Total improvement:",
        f"{mean([r['total_improvement_pct'] for r in subset]):.4f}%"
    )
    print(
        "  PWEA faster:",
        f"{sum(r['inference_improvement_pct'] > 0 for r in subset)}"
        f"/{len(subset)}"
    )

# Prompt-level means: each prompt gets equal weight.
print("\n========== PROMPT-LEVEL RESULTS ==========")
prompt_results = []

for index in range(1, 13):
    subset = [r for r in records if r["prompt"] == index]
    if len(subset) != 3:
        raise SystemExit(f"Prompt {index}: expected 3 pairs")

    item = {
        "prompt": index,
        "category": subset[0]["category"],
        "inference_improvement_pct": mean(
            [r["inference_improvement_pct"] for r in subset]
        ),
        "total_improvement_pct": mean(
            [r["total_improvement_pct"] for r in subset]
        ),
    }
    prompt_results.append(item)

    print(
        f"Prompt {index:02d} | {item['category']} | "
        f"inference {item['inference_improvement_pct']:+.4f}% | "
        f"total {item['total_improvement_pct']:+.4f}%"
    )

print("\nEqual-weight prompt mean inference improvement:",
      f"{mean([x['inference_improvement_pct'] for x in prompt_results]):.4f}%")
print("Equal-weight prompt mean total improvement:",
      f"{mean([x['total_improvement_pct'] for x in prompt_results]):.4f}%")

# Placement and execution.
print("\n========== PLACEMENT / EXECUTION ==========")

pwea_served_vram = sum(r["pwea_served"][2] for r in records)
pwea_acquires = sum(r["pwea_acquires"] for r in records)
pwea_migrations = [
    sum(r["pwea_migrations"][i] for r in records)
    for i in range(5)
]
pwea_migration_failures = [
    sum(r["pwea_migration_failures"][i] for r in records)
    for i in range(5)
]
pwea_bytes = [
    sum(r["pwea_migration_bytes"][i] for r in records)
    for i in range(5)
]
pwea_migration_us = [
    sum(r["pwea_migration_duration_us"][i] for r in records)
    for i in range(5)
]

print("PWEA successful acquisitions:", pwea_acquires)
print("PWEA VRAM-served acquisitions:", pwea_served_vram)
print(
    "VRAM-served share:",
    f"{pwea_served_vram / pwea_acquires * 100:.4f}%"
    if pwea_acquires else "N/A"
)
print("Migration counts by path index:", pwea_migrations)
print("Migration failures by path index:", pwea_migration_failures)
print("Migration bytes by path index:", pwea_bytes)
print("Migration duration (microseconds) by path index:", pwea_migration_us)

print("\n========== CUDA EXECUTION ==========")
for mode in ("ssd", "pwea"):
    values = [r[f"{mode}_cuda"] for r in records]
    if any(v is None for v in values):
        print(mode, "CUDA counters missing in one or more logs")
        continue

    gpu = sum(v[0] for v in values)
    fallback = sum(v[1] for v in values)
    pairs = sum(v[2] for v in values)

    print(
        f"{mode.upper()}: gpu_calls={gpu}, "
        f"cpu_fallbacks={fallback}, pair_calls={pairs}"
    )

print("\n========== ETF / ROUTING VALIDATION ==========")
print("SSD ETF events:", sum(r["ssd_etf_events"] for r in records))
print("PWEA ETF events:", sum(r["pwea_etf_events"] for r in records))
print("Total dropped SSD events:", sum(r["ssd_dropped"] for r in records))
print("Total dropped PWEA events:", sum(r["pwea_dropped"] for r in records))

same_routes = sum(r["routing"]["same_routes"] for r in records)
same_lengths = sum(r["routing"]["same_length"] for r in records)
weight_diffs = [
    r["routing"]["max_weight_diff"]
    for r in records
    if r["routing"]["max_weight_diff"] is not None
]

print(f"Same ETF route sequence: {same_routes}/36")
print(f"Same ETF event count: {same_lengths}/36")
print(
    "Maximum routing-weight difference:",
    f"{max(weight_diffs):.12g}" if weight_diffs else "N/A"
)

# Save machine-readable analysis.
output = {
    "schema": "phaseE-analysis-v1",
    "manifest_sha256_expected":
        "fff9e6e45f5d901528287f6059637f29df2c77437cbbeaee1584ca7289f16976",
    "prompt_count": 12,
    "paired_run_count": len(records),
    "category_results": {},
    "prompt_results": prompt_results,
    "run_results": records,
}

for category in categories:
    subset = [r for r in records if r["category"] == category]
    output["category_results"][category] = {
        "paired_runs": len(subset),
        "mean_inference_improvement_pct": mean(
            [r["inference_improvement_pct"] for r in subset]
        ),
        "mean_total_improvement_pct": mean(
            [r["total_improvement_pct"] for r in subset]
        ),
    }

out = ROOT / "phaseE_analysis.json"
out.write_text(json.dumps(output, indent=2) + "\n")

print("\nAnalysis saved to:", out)
print("PHASE E AGGREGATE ANALYSIS: PASS")
