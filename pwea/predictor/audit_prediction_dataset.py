#!/usr/bin/env python3
"""Read-only audit of the PWEA prompt-to-telemetry prediction dataset."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "pwea/artifacts/loco_five_category_cohort_manifest_20261115.json"
INTEGRITY = ROOT / "pwea/artifacts/loco_dataset_integrity_audit_20261115.json"
EXPERIMENT = ROOT / "pwea/artifacts/experiment_manifest_20261112.json"
DEBUG_MANIFEST = ROOT / "pwea/prompts/python_debugging_distinct_suites_manifest.json"
OUTPUT = ROOT / "pwea/artifacts/prediction/prediction_dataset_audit_20261115.json"

N_LAYER = 48
N_EXPERT = 128
EXPECTED_SESSIONS = 25
EXPECTED_PROMPTS = 175
EXPECTED_PER_SESSION = 7


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path):
    return ROOT / path


def read_suite(path):
    # splitlines preserves literal backslash-n sequences inside each prompt.
    with path.open(encoding="utf-8") as stream:
        prompts = stream.read().splitlines()

    if len(prompts) != EXPECTED_PER_SESSION:
        raise ValueError(
            f"{path}: expected 7 physical prompt lines, found {len(prompts)}"
        )
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError(f"{path}: empty prompt line detected")
    return prompts


def main():
    errors = []
    warnings = []
    session_results = []
    category_sessions = Counter()
    category_prompts = Counter()
    totals = {
        "sessions": 0,
        "prompts": 0,
        "activation_events": 0,
        "text_mapped_prompts": 0,
        "text_unresolved_prompts": 0,
        "generation_censored_prompts": 0,
        "zero_event_layers": 0,
    }

    cohort = load_json(COHORT)
    integrity = load_json(INTEGRITY)
    experiment = load_json(EXPERIMENT)
    debug_manifest = load_json(DEBUG_MANIFEST)

    if integrity.get("status") != "passed" or integrity.get("errors") != []:
        errors.append("Existing dataset integrity audit is not clean")

    audit_sessions = {
        row["session_id"]: row
        for row in integrity.get("sessions", [])
    }

    debug_experiment = experiment.get("distinct_debugging_experiment", {})
    debug_sessions = {
        row["workload_id"]: row
        for row in debug_experiment.get("sessions", [])
    }

    debug_suites = debug_manifest.get("suites", {})

    for session in cohort.get("sessions", []):
        category = session["category"]
        session_id = session["session_id"]
        raw_path = resolve(session["input"])
        audit_row = audit_sessions.get(session_id)

        result = {
            "session_id": session_id,
            "category": category,
            "input": session["input"],
            "prompt_count": 0,
            "activation_events": 0,
            "prompt_text_status": "unresolved",
            "prompt_suite": None,
            "prompt_suite_sha256": None,
            "generation_cap_prompt_ids": [],
            "zero_event_layers_by_prompt": {},
            "status": "passed",
        }

        try:
            if not raw_path.is_file():
                raise ValueError(f"Missing telemetry file: {raw_path}")

            actual_hash = sha256(raw_path)
            if actual_hash != session["sha256"]:
                raise ValueError("Telemetry SHA-256 differs from cohort manifest")

            if not audit_row:
                raise ValueError("Session absent from integrity audit")

            if audit_row.get("input") != session["input"]:
                raise ValueError("Integrity audit input path mismatch")

            suite_rel = audit_row.get("prompt_suite")
            suite_hash = None
            prompts = None
            cap_ids = set()

            if category == "debugging":
                # Match workload ID from the raw records to the frozen
                # debugging experiment's exact suite assignment.
                suite_rel = None

            if category == "debugging":
                # Resolve the debugging suite after reading the telemetry ID.
                pass

            with raw_path.open(encoding="utf-8") as stream:
                records = [
                    json.loads(line)
                    for line in stream
                    if line.strip()
                ]

            if len(records) != EXPECTED_PER_SESSION:
                raise ValueError(
                    f"Expected 7 telemetry records, found {len(records)}"
                )

            prompt_ids = [row.get("prompt_id") for row in records]
            if sorted(prompt_ids) != list(range(1, 8)):
                raise ValueError(f"Invalid prompt IDs: {prompt_ids}")

            workload_ids = {
                row.get("workload", {}).get("workload_id")
                for row in records
            }
            if len(workload_ids) != 1:
                raise ValueError(f"Mixed workload IDs: {workload_ids}")

            workload_id = next(iter(workload_ids))

            if category == "debugging":
                debug_row = debug_sessions.get(workload_id)
                if not debug_row:
                    raise ValueError(
                        f"Debugging workload missing from experiment manifest: "
                        f"{workload_id}"
                    )
                suite_letter = debug_row["suite"].lower()
                suite_info = debug_suites[suite_letter]
                suite_rel = suite_info["file"]
                suite_hash = suite_info["sha256"]

            elif category == "code-generation":
                # No verified original prompt source was recovered.
                suite_rel = None

            else:
                if not audit_row.get("config") or not audit_row.get("report"):
                    raise ValueError("Pilot config/report reference missing")

                config_path = resolve(audit_row["config"])
                report_path = resolve(audit_row["report"])

                config = load_json(config_path)
                report = load_json(report_path)

                if config.get("session_id") != workload_id:
                    raise ValueError("Pilot config workload/session mismatch")

                suite_info = config.get("prompt_suite", {})
                suite_rel = suite_info.get("path")
                suite_hash = suite_info.get("sha256")

                if suite_rel != audit_row.get("prompt_suite"):
                    raise ValueError("Config and integrity audit suite mismatch")

                if report.get("session_id") != workload_id:
                    raise ValueError("Pilot report workload/session mismatch")

                prompt_results = report.get("prompt_results", [])
                if len(prompt_results) != EXPECTED_PER_SESSION:
                    raise ValueError(
                        f"Expected 7 prompt results, found {len(prompt_results)}"
                    )

                for prompt_id, prompt_result in enumerate(prompt_results, 1):
                    if prompt_result.get("hit_token_limit") is True:
                        cap_ids.add(prompt_id)

                if sorted(cap_ids) != sorted(
                    audit_row.get("generation_cap_prompt_ids", [])
                ):
                    raise ValueError("Generation-cap IDs disagree with audit")

            if suite_rel:
                suite_path = resolve(suite_rel)
                if not suite_path.is_file():
                    raise ValueError(f"Missing prompt suite: {suite_path}")

                actual_suite_hash = sha256(suite_path)
                if actual_suite_hash != suite_hash:
                    raise ValueError("Prompt-suite SHA-256 mismatch")

                prompts = read_suite(suite_path)
                result["prompt_text_status"] = "verified"
                result["prompt_suite"] = suite_rel
                result["prompt_suite_sha256"] = actual_suite_hash
            else:
                result["prompt_text_status"] = "unresolved"

            for row in records:
                if row.get("record_type") != "etf_prompt":
                    raise ValueError("Unexpected telemetry record type")
                if row.get("schema_version") != 1:
                    raise ValueError("Unsupported telemetry schema")

                workload = row.get("workload", {})
                if workload.get("task_category") != category:
                    raise ValueError("Telemetry category mismatch")
                if workload.get("language") != "Python":
                    raise ValueError("Unexpected language")
                if workload.get("framework") != "none":
                    raise ValueError("Unexpected framework")
                if workload.get("workload_phase") != "inference":
                    raise ValueError("Unexpected workload phase")

                window = row.get("window", {})
                if window.get("dropped_events") != 0:
                    raise ValueError("Telemetry contains dropped events")

                counts = [[0] * N_EXPERT for _ in range(N_LAYER)]
                for event in row.get("activations", []):
                    layer = event.get("layer")
                    expert = event.get("expert_id")
                    weight = event.get("routing_weight")

                    if (
                        isinstance(layer, bool) or not isinstance(layer, int)
                        or not 0 <= layer < N_LAYER
                    ):
                        raise ValueError(f"Invalid layer index: {layer}")

                    if (
                        isinstance(expert, bool) or not isinstance(expert, int)
                        or not 0 <= expert < N_EXPERT
                    ):
                        raise ValueError(f"Invalid expert index: {expert}")

                    if (
                        isinstance(weight, bool)
                        or not isinstance(weight, (int, float))
                        or not math.isfinite(weight)
                    ):
                        raise ValueError("Invalid routing weight")

                    counts[layer][expert] += 1

                zero_layers = [
                    layer for layer, row_counts in enumerate(counts)
                    if sum(row_counts) == 0
                ]

                result["zero_event_layers_by_prompt"][str(row["prompt_id"])] = (
                    zero_layers
                )
                totals["zero_event_layers"] += len(zero_layers)

                # Verify each observed layer produces a probability distribution.
                for layer, row_counts in enumerate(counts):
                    denominator = sum(row_counts)
                    if denominator:
                        shares = [count / denominator for count in row_counts]
                        if not math.isclose(sum(shares), 1.0, abs_tol=1e-12):
                            raise ValueError(
                                f"Layer {layer}: normalized shares do not sum to 1"
                            )

                result["activation_events"] += len(row.get("activations", []))
                result["prompt_count"] += 1

            result["generation_cap_prompt_ids"] = sorted(cap_ids)
            result["generation_censored_prompt_count"] = len(cap_ids)

            if result["prompt_text_status"] == "verified":
                totals["text_mapped_prompts"] += result["prompt_count"]
            else:
                totals["text_unresolved_prompts"] += result["prompt_count"]

            totals["generation_censored_prompts"] += len(cap_ids)
            totals["prompts"] += result["prompt_count"]
            totals["activation_events"] += result["activation_events"]
            totals["sessions"] += 1
            category_sessions[category] += 1
            category_prompts[category] += result["prompt_count"]

        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            errors.append(f"{session_id}: {exc}")

        session_results.append(result)

    expected_activation_events = cohort.get("activation_event_count")
    if totals["activation_events"] != expected_activation_events:
        errors.append(
            "Activation total mismatch: "
            f"{totals['activation_events']} != {expected_activation_events}"
        )

    if totals["sessions"] != EXPECTED_SESSIONS:
        errors.append(f"Expected {EXPECTED_SESSIONS} sessions")

    if totals["prompts"] != EXPECTED_PROMPTS:
        errors.append(f"Expected {EXPECTED_PROMPTS} prompts")

    if any(count != 5 for count in category_sessions.values()):
        errors.append(f"Expected five sessions per category: {dict(category_sessions)}")

    if any(count != 35 for count in category_prompts.values()):
        errors.append(f"Expected 35 prompts per category: {dict(category_prompts)}")

    if totals["text_unresolved_prompts"] != 35:
        warnings.append(
            "Expected 35 unresolved code-generation prompt texts; "
            f"found {totals['text_unresolved_prompts']}"
        )

    report = {
        "artifact_type": "pwea_prediction_dataset_join_audit",
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "model_dimensions": {
            "layers": N_LAYER,
            "experts_per_layer": N_EXPERT,
            "target_values_per_prompt": N_LAYER * N_EXPERT,
        },
        "target_definition": {
            "activation_share": (
                "activation_count(prompt, layer, expert) / "
                "sum_expert activation_count(prompt, layer, expert)"
            ),
            "active_mask": "activation_count > 0",
            "valid_layer_mask": "layer activation total > 0",
            "zero_event_layer_policy": "mask layer; do not treat as zero distribution",
        },
        "totals": totals,
        "category_sessions": dict(category_sessions),
        "category_prompts": dict(category_prompts),
        "errors": errors,
        "warnings": warnings,
        "sessions": session_results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Prediction dataset join audit:", report["status"].upper())
    print("Sessions:", totals["sessions"])
    print("Prompts:", totals["prompts"])
    print("Activation events:", totals["activation_events"])
    print("Verified prompt text:", totals["text_mapped_prompts"])
    print("Unresolved prompt text:", totals["text_unresolved_prompts"])
    print("Generation-censored prompts:", totals["generation_censored_prompts"])
    print("Zero-event prompt/layers:", totals["zero_event_layers"])
    print("Errors:", len(errors))
    print("Warnings:", len(warnings))
    print("Report:", OUTPUT.relative_to(ROOT))

    for error in errors:
        print("ERROR:", error)
    for warning in warnings:
        print("WARNING:", warning)

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
