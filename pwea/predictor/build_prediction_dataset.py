#!/usr/bin/env python3
"""Build prompt-level expert activation targets from the audited PWEA cohort."""

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "pwea/artifacts/prediction/prediction_dataset_audit_20261115.json"
COHORT = ROOT / "pwea/artifacts/loco_five_category_cohort_manifest_20261115.json"

OUT_DIR = ROOT / "pwea/artifacts/prediction"
OUT_JSONL = OUT_DIR / "expert_prediction_samples_20261115.jsonl"
OUT_MANIFEST = OUT_DIR / "expert_prediction_dataset_manifest_20261115.json"

N_LAYER = 48
N_EXPERT = 128
EXPECTED_PROMPTS = 175


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_suite(path):
    with path.open(encoding="utf-8") as stream:
        prompts = stream.read().splitlines()
    if len(prompts) != 7 or any(not item.strip() for item in prompts):
        raise ValueError(f"Invalid seven-prompt suite: {path}")
    return prompts


def build_sample(session, row, prompt_text, cap_ids, device):
    prompt_id = row["prompt_id"]
    counts = [[0] * N_EXPERT for _ in range(N_LAYER)]

    for event in row["activations"]:
        layer = event["layer"]
        expert = event["expert_id"]

        if not 0 <= layer < N_LAYER:
            raise ValueError(f"Invalid layer {layer}")
        if not 0 <= expert < N_EXPERT:
            raise ValueError(f"Invalid expert {expert}")

        counts[layer][expert] += 1

    activation_share = []
    active_mask = []
    valid_layer_mask = []
    layer_event_counts = []

    for layer_counts in counts:
        total = sum(layer_counts)
        valid = total > 0

        valid_layer_mask.append(valid)
        layer_event_counts.append(total)

        if valid:
            shares = [count / total for count in layer_counts]
            if not math.isclose(sum(shares), 1.0, abs_tol=1e-12):
                raise ValueError("Target normalization failure")
        else:
            # Invalid layers are masked, not represented as valid zero targets.
            shares = [0.0] * N_EXPERT

        activation_share.append(shares)
        active_mask.append([count > 0 for count in layer_counts])

    if len(activation_share) != N_LAYER:
        raise ValueError("Incorrect layer dimension")
    if any(len(values) != N_EXPERT for values in activation_share):
        raise ValueError("Incorrect expert dimension")

    workload = row["workload"]
    return {
        "sample_id": f'{session["session_id"]}:prompt-{prompt_id:02d}',
        "session_id": session["session_id"],
        "prompt_id": prompt_id,
        "category": session["category"],
        "language": workload["language"],
        "framework": workload["framework"],
        "workload_phase": workload["workload_phase"],
        "prompt_text": prompt_text,
        "prompt_text_status": session["prompt_text_status"],
        "generation_censored": prompt_id in cap_ids,
        "target": {
            "activation_share": activation_share,
            "active_mask": active_mask,
            "valid_layer_mask": valid_layer_mask,
            "layer_event_counts": layer_event_counts,
        },
        "telemetry": {
            "activation_event_count": len(row["activations"]),
            "dropped_events": row["window"]["dropped_events"],
            "device": device,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing output files",
    )
    args = parser.parse_args()

    for output in (OUT_JSONL, OUT_MANIFEST):
        if output.exists() and not args.force:
            raise SystemExit(
                f"Refusing to overwrite {output}. Use --force if intentional."
            )

    audit = load_json(AUDIT)
    cohort = load_json(COHORT)

    if audit.get("status") != "passed":
        raise SystemExit("Prediction dataset audit has not passed")

    cohort_sessions = {
        session["session_id"]: session
        for session in cohort["sessions"]
    }

    samples = []
    category_counts = Counter()
    censored_counts = Counter()
    text_status_counts = Counter()
    total_events = 0

    for audited in audit["sessions"]:
        if audited["status"] != "passed":
            raise ValueError(f'Audit session failed: {audited["session_id"]}')

        session_id = audited["session_id"]
        frozen = cohort_sessions.get(session_id)
        if frozen is None:
            raise ValueError(f"Session missing from cohort: {session_id}")

        raw_path = ROOT / frozen["input"]
        if sha256(raw_path) != frozen["sha256"]:
            raise ValueError(f"Raw telemetry hash mismatch: {session_id}")

        prompt_texts = None
        if audited["prompt_text_status"] == "verified":
            suite_path = ROOT / audited["prompt_suite"]
            if sha256(suite_path) != audited["prompt_suite_sha256"]:
                raise ValueError(f"Prompt suite hash mismatch: {session_id}")
            prompt_texts = read_suite(suite_path)

        cap_ids = set(audited["generation_cap_prompt_ids"])

        with raw_path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]

        if len(records) != 7:
            raise ValueError(f"{session_id}: expected seven records")

        if sorted(row["prompt_id"] for row in records) != list(range(1, 8)):
            raise ValueError(f"{session_id}: invalid prompt ID sequence")

        for row in sorted(records, key=lambda item: item["prompt_id"]):
            if row["workload"]["workload_id"] != session_id:
                raise ValueError(f"{session_id}: workload ID mismatch")
            if row["workload"]["task_category"] != audited["category"]:
                raise ValueError(f"{session_id}: category mismatch")

            prompt_id = row["prompt_id"]
            prompt_text = (
                prompt_texts[prompt_id - 1]
                if prompt_texts is not None
                else None
            )

            sample = build_sample(audited, row, prompt_text, cap_ids, cohort["protocol"]["device"])
            samples.append(sample)

            category_counts[sample["category"]] += 1
            text_status_counts[sample["prompt_text_status"]] += 1
            if sample["generation_censored"]:
                censored_counts[sample["category"]] += 1

            total_events += sample["telemetry"]["activation_event_count"]

    if len(samples) != EXPECTED_PROMPTS:
        raise ValueError(f"Expected 175 samples, generated {len(samples)}")

    if total_events != cohort["activation_event_count"]:
        raise ValueError(
            f"Activation total mismatch: {total_events} != "
            f"{cohort['activation_event_count']}"
        )

    if any(count != 35 for count in category_counts.values()):
        raise ValueError(f"Unexpected category distribution: {dict(category_counts)}")

    if text_status_counts["verified"] != 140:
        raise ValueError("Expected 140 samples with verified prompt text")

    if text_status_counts["unresolved"] != 35:
        raise ValueError("Expected 35 samples with unresolved prompt text")

    if sum(censored_counts.values()) != 20:
        raise ValueError("Expected 20 generation-censored samples")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_jsonl = OUT_JSONL.with_suffix(".jsonl.tmp")
    temp_manifest = OUT_MANIFEST.with_suffix(".json.tmp")

    try:
        with temp_jsonl.open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(
                    json.dumps(
                        sample,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())

        manifest = {
            "artifact_type": "pwea_prompt_level_expert_prediction_dataset",
            "schema_version": 1,
            "status": "passed",
            "source_audit": str(AUDIT.relative_to(ROOT)),
            "source_cohort": str(COHORT.relative_to(ROOT)),
            "sample_file": str(OUT_JSONL.relative_to(ROOT)),
            "sample_count": len(samples),
            "category_counts": dict(category_counts),
            "prompt_text_status_counts": dict(text_status_counts),
            "generation_censored_counts": dict(censored_counts),
            "activation_event_count": total_events,
            "model_dimensions": {
                "layers": N_LAYER,
                "experts_per_layer": N_EXPERT,
                "target_values_per_prompt": N_LAYER * N_EXPERT,
            },
            "target_definition": {
                "activation_share": "per-prompt activation count normalized across experts within each layer",
                "active_mask": "true when the expert has one or more activation events",
                "valid_layer_mask": "true when the layer has one or more activation events",
                "invalid_layer_target": "zero-filled with valid_layer_mask=false",
                "routing_weights_used_as_target": False,
            },
            "limitations": [
                "35 code-generation samples have unresolved prompt text",
                "20 algorithmic-problem-solving samples are generation-censored",
                "Dataset is telemetry from simulated device mode",
                "Dataset export does not establish predictor performance or placement benefit",
            ],
        }

        temp_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        manifest["sample_file_sha256"] = sha256(temp_jsonl)
        temp_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        os.replace(temp_jsonl, OUT_JSONL)
        os.replace(temp_manifest, OUT_MANIFEST)

    finally:
        temp_jsonl.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)

    print("Prediction dataset export: PASSED")
    print("Samples:", len(samples))
    print("Category distribution:", dict(category_counts))
    print("Prompt text status:", dict(text_status_counts))
    print("Generation-censored:", sum(censored_counts.values()))
    print("Activation events:", total_events)
    print("JSONL:", OUT_JSONL.relative_to(ROOT))
    print("Manifest:", OUT_MANIFEST.relative_to(ROOT))
    print("JSONL SHA-256:", sha256(OUT_JSONL))


if __name__ == "__main__":
    main()
