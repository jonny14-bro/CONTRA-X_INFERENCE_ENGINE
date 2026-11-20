#!/usr/bin/env python3
"""Build deterministic workload-conditioned expert profiles from ETF JSONL."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA_VERSION = 1
WORKLOAD_FIELDS = (
    "language",
    "framework",
    "task_category",
    "workload_phase",
)


class ValidationError(Exception):
    pass


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_int(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{field} must be an integer >= {minimum}")
    return value


def require_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    if value.strip().lower() == "unknown":
        raise ValidationError(f"{field} is unknown; labelled data is required")
    return value.strip()


def workload_key(workload):
    return tuple(workload[field] for field in WORKLOAD_FIELDS)


def load_records(path, n_layer, n_expert):
    records = []
    seen_prompt_ids = set()
    source_workload_ids = set()
    total_activations = 0

    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue

            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"line {line_number}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValidationError(f"line {line_number}: record must be an object")

            if row.get("record_type") != "etf_prompt":
                raise ValidationError(f"line {line_number}: unsupported record_type")

            if row.get("schema_version") != SCHEMA_VERSION:
                raise ValidationError(
                    f"line {line_number}: unsupported schema_version "
                    f"{row.get('schema_version')!r}"
                )

            prompt_id = require_int(row.get("prompt_id"), "prompt_id", 1)
            if prompt_id in seen_prompt_ids:
                raise ValidationError(
                    f"line {line_number}: duplicate prompt_id {prompt_id}; "
                    "use one ETF log per engine session"
                )
            seen_prompt_ids.add(prompt_id)

            workload = row.get("workload")
            if not isinstance(workload, dict):
                raise ValidationError(f"line {line_number}: workload must be an object")

            labels = {
                field: require_string(workload.get(field), f"workload.{field}")
                for field in WORKLOAD_FIELDS
            }
            workload_id = require_string(
                workload.get("workload_id"), "workload.workload_id"
            )
            source_workload_ids.add(workload_id)

            window = row.get("window")
            if not isinstance(window, dict):
                raise ValidationError(f"line {line_number}: window must be an object")

            first_id = require_int(window.get("first_prompt_id"), "first_prompt_id", 1)
            last_id = require_int(window.get("last_prompt_id"), "last_prompt_id", 1)
            completed = require_int(window.get("completed_prompts"), "completed_prompts", 1)
            dropped = require_int(window.get("dropped_events"), "dropped_events")

            if not first_id <= prompt_id <= last_id:
                raise ValidationError(
                    f"line {line_number}: prompt_id outside its telemetry window"
                )
            if completed < 1:
                raise ValidationError(f"line {line_number}: invalid completed_prompts")
            if dropped:
                raise ValidationError(
                    f"line {line_number}: dropped_events={dropped}; "
                    "incomplete telemetry is not accepted by default"
                )

            activations = row.get("activations")
            if not isinstance(activations, list):
                raise ValidationError(f"line {line_number}: activations must be a list")

            counts = Counter()
            weight_sums = defaultdict(float)

            for index, event in enumerate(activations):
                if not isinstance(event, dict):
                    raise ValidationError(
                        f"line {line_number}: activation {index} must be an object"
                    )

                layer = require_int(event.get("layer"), "activation.layer")
                expert = require_int(event.get("expert_id"), "activation.expert_id")

                if layer >= n_layer:
                    raise ValidationError(
                        f"line {line_number}: layer {layer} outside model dimensions"
                    )
                if expert >= n_expert:
                    raise ValidationError(
                        f"line {line_number}: expert {expert} outside model dimensions"
                    )

                token_position = event.get("token_position")
                if isinstance(token_position, bool) or not isinstance(token_position, int):
                    raise ValidationError(
                        f"line {line_number}: token_position must be an integer"
                    )

                weight = event.get("routing_weight")
                if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                    raise ValidationError(
                        f"line {line_number}: routing_weight must be numeric"
                    )
                if not math.isfinite(weight):
                    raise ValidationError(
                        f"line {line_number}: routing_weight must be finite"
                    )

                key = (layer, expert)
                counts[key] += 1
                weight_sums[key] += float(weight)

            records.append({
                "prompt_id": prompt_id,
                "workload_id": workload_id,
                "labels": labels,
                "counts": counts,
                "weight_sums": weight_sums,
                "activation_count": sum(counts.values()),
            })
            total_activations += sum(counts.values())

    if not records:
        raise ValidationError("input contains no ETF prompt records")

    return records, source_workload_ids, total_activations



def load_multiple_records(paths, n_layer, n_expert):
    """Load independent ETF session files without conflating prompt IDs."""
    if not paths:
        raise ValidationError("at least one ETF input file is required")

    normalized = [Path(path).resolve() for path in paths]
    if len(set(normalized)) != len(normalized):
        raise ValidationError("the same ETF input file was supplied more than once")

    all_records = []
    all_workload_ids = set()
    total_activations = 0
    source_files = []

    for path in normalized:
        if not path.is_file():
            raise ValidationError(f"input file does not exist: {path}")

        records, workload_ids, activation_count = load_records(
            path, n_layer, n_expert
        )

        overlap = all_workload_ids.intersection(workload_ids)
        if overlap:
            raise ValidationError(
                f"{path.name}: workload_id reused across source files: "
                f"{sorted(overlap)}"
            )

        all_workload_ids.update(workload_ids)
        all_records.extend(records)
        total_activations += activation_count

        source_files.append({
            "file_name": path.name,
            "sha256": sha256_file(path),
            "prompt_records": len(records),
            "source_workload_ids": sorted(workload_ids),
            "activation_events": activation_count,
        })

    return all_records, all_workload_ids, total_activations, source_files


def build_profile(
    records, n_layer, n_expert, min_prompts, min_workloads
):
    groups = defaultdict(list)
    for record in records:
        groups[workload_key(record["labels"])].append(record)

    profiles = []
    for key in sorted(groups):
        samples = groups[key]
        prompt_count = len(samples)
        source_workload_count = len({
            row["workload_id"] for row in samples
        })
        total_activations = sum(row["activation_count"] for row in samples)

        expert_counts = Counter()
        expert_weight_sums = defaultdict(float)
        layer_totals = Counter()
        prompt_share_sums = defaultdict(float)
        prompt_share_counts = Counter()

        for row in samples:
            prompt_layer_totals = Counter()
            for (layer, expert), count in row["counts"].items():
                expert_counts[(layer, expert)] += count
                expert_weight_sums[(layer, expert)] += row["weight_sums"][(layer, expert)]
                prompt_layer_totals[layer] += count

            # Normalize within each prompt and layer so each prompt
            # contributes equally, regardless of its activation length.
            for (layer, expert), count in row["counts"].items():
                denominator = prompt_layer_totals[layer]
                if denominator:
                    prompt_share_sums[(layer, expert)] += count / denominator
                    prompt_share_counts[(layer, expert)] += 1

            for layer, count in prompt_layer_totals.items():
                layer_totals[layer] += count

        eligible = (
            prompt_count >= min_prompts
            and source_workload_count >= min_workloads
        )

        layers = []
        for layer in range(n_layer):
            layer_total = layer_totals[layer]
            experts = []

            for expert in range(n_expert):
                count = expert_counts[(layer, expert)]
                prompt_observations = prompt_share_counts[(layer, expert)]
                aggregate_share = count / layer_total if layer_total else 0.0
                mean_prompt_share = (
                    prompt_share_sums[(layer, expert)] / prompt_count
                )

                experts.append({
                    "expert_id": expert,
                    "activation_count": count,
                    "aggregate_activation_share": aggregate_share,
                    "mean_prompt_activation_share": mean_prompt_share,
                    "prompt_observation_count": prompt_observations,
                    "mean_routing_weight": (
                        expert_weight_sums[(layer, expert)] / count
                        if count else 0.0
                    ),
                })

            experts.sort(
                key=lambda item: (
                    -item["mean_prompt_activation_share"],
                    item["expert_id"],
                )
            )

            layers.append({
                "layer": layer,
                "activation_count": layer_total,
                "experts": experts,
            })

        profiles.append({
            "workload": dict(zip(WORKLOAD_FIELDS, key)),
            "support": {
                "prompt_count": prompt_count,
                "source_workload_count": source_workload_count,
                "activation_count": total_activations,
                "minimum_prompt_support": min_prompts,
                "minimum_independent_workload_support": min_workloads,
                "eligible_for_prediction": eligible,
                "support_status": "supported" if eligible else "insufficient",
            },
            "layers": layers,
        })

    return profiles


def main():
    parser = argparse.ArgumentParser(
        description="Build PWEA workload-conditioned expert profiles from ETF JSONL."
    )
    parser.add_argument("--input", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--n-layer", required=True, type=int)
    parser.add_argument("--n-expert", required=True, type=int)
    parser.add_argument("--min-prompts", type=int, default=20)
    parser.add_argument("--min-workloads", type=int, default=3)
    args = parser.parse_args()

    if args.n_layer <= 0 or args.n_expert <= 0:
        parser.error("--n-layer and --n-expert must be positive")
    if args.min_prompts <= 0:
        parser.error("--min-prompts must be positive")
    if args.min_workloads <= 0:
        parser.error("--min-workloads must be positive")
    resolved_inputs = [path.resolve() for path in args.input]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        parser.error("the same --input file was supplied more than once")
    for path in resolved_inputs:
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
        if path == args.output.resolve():
            parser.error("input and output paths must differ")

    try:
        records, workload_ids, activation_total, source_files = (
            load_multiple_records(args.input, args.n_layer, args.n_expert)
        )
        profiles = build_profile(
            records, args.n_layer, args.n_expert,
            args.min_prompts, args.min_workloads,
        )

        artifact = {
            "artifact_type": "pwea_workload_expert_profile",
            "schema_version": 1,
            "method": "empirical_activation_distribution",
            "model": {
                "model_id": args.model_id,
                "n_layer": args.n_layer,
                "n_expert": args.n_expert,
            },
            "source": {
                "prompt_records": len(records),
                "source_workload_ids": sorted(workload_ids),
                "activation_events": activation_total,
                "dropped_events": 0,
                **(
                    {
                        "file_name": source_files[0]["file_name"],
                        "sha256": source_files[0]["sha256"],
                    }
                    if len(source_files) == 1
                    else {"files": source_files}
                ),
            },
            "workload_fields": list(WORKLOAD_FIELDS),
            "profiles": profiles,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                artifact, indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)

        supported = sum(
            profile["support"]["eligible_for_prediction"]
            for profile in profiles
        )
        print(f"PWEA profile written: {args.output}")
        print(f"prompt records: {len(records)}")
        print(f"activation events: {activation_total}")
        print(f"workload groups: {len(profiles)}")
        print(f"supported groups: {supported}")
        print(f"insufficient-support groups: {len(profiles) - supported}")
        if "sha256" in artifact["source"]:
            print(f"source SHA-256: {artifact['source']['sha256']}")
        else:
            for source_file in artifact["source"]["files"]:
                print(
                    f"source SHA-256 ({source_file['file_name']}): "
                    f"{source_file['sha256']}"
                )

    except (ValidationError, OSError) as exc:
        print(f"PWEA validation/build failed: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
