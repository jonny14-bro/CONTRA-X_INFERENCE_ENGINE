#!/usr/bin/env python3
"""Reproducible leave-one-task-category-out PWEA evaluation runner."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import build_profile as builder
import evaluate_category_transfer as evaluator


class ConfigError(Exception):
    pass


EXPECTED_CATEGORIES = (
    "code-generation",
    "debugging",
    "code-explanation",
    "code-transformation",
    "algorithmic-problem-solving",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def resolve(value, base, name):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path):
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Invalid JSON at line {exc.lineno}: {exc.msg}"
        ) from exc

    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ConfigError("Expected a JSON object with schema_version: 1")

    model = config.get("model")
    if not isinstance(model, dict):
        raise ConfigError("model must be an object")

    model_id = model.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError("model.model_id must be a non-empty string")

    n_layer = positive_int(model.get("n_layer"), "model.n_layer")
    n_expert = positive_int(model.get("n_expert"), "model.n_expert")
    if n_expert <= 1:
        raise ConfigError("model.n_expert must exceed 1")

    categories = config.get("categories")
    if categories != list(EXPECTED_CATEGORIES):
        raise ConfigError(
            "categories must exactly match the five-category protocol "
            "in the documented order"
        )

    raw_sessions = config.get("sessions")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ConfigError("sessions must be a non-empty array")

    sessions = []
    seen_paths = set()
    for index, item in enumerate(raw_sessions):
        if not isinstance(item, dict):
            raise ConfigError(f"sessions[{index}] must be an object")

        category = item.get("task_category")
        if category not in EXPECTED_CATEGORIES:
            raise ConfigError(
                f"sessions[{index}].task_category is not an expected category"
            )

        session_id = item.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ConfigError(
                f"sessions[{index}].session_id must be a non-empty string"
            )

        input_path = resolve(item.get("input"), path.parent,
                             f"sessions[{index}].input")
        if input_path in seen_paths:
            raise ConfigError(f"duplicate session input: {input_path}")
        seen_paths.add(input_path)

        if not input_path.is_file():
            raise ConfigError(f"input file does not exist: {input_path}")

        sessions.append({
            "task_category": category,
            "session_id": session_id.strip(),
            "path": input_path,
        })

    ids = [item["session_id"] for item in sessions]
    if len(ids) != len(set(ids)):
        raise ConfigError("session_id values must be unique")

    counts = {category: 0 for category in EXPECTED_CATEGORIES}
    for item in sessions:
        counts[item["task_category"]] += 1

    insufficient = {
        category: count for category, count in counts.items() if count < 3
    }
    if insufficient:
        details = ", ".join(
            f"{category}={count}" for category, count in insufficient.items()
        )
        raise ConfigError(
            "at least three independent session files per category are "
            f"required; insufficient coverage: {details}"
        )

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ConfigError("evaluation must be an object")

    output = resolve(evaluation.get("output"), path.parent, "evaluation.output")
    summary = resolve(
        config.get("run_summary"),
        path.parent,
        "run_summary",
    )
    checksums = resolve(
        config.get("checksums"),
        path.parent,
        "checksums",
    )

    k = positive_int(evaluation.get("k", 5), "evaluation.k")
    relevant_k = positive_int(
        evaluation.get("relevant_k", max(1, (n_expert + 9) // 10)),
        "evaluation.relevant_k",
    )
    if k > n_expert or relevant_k > n_expert:
        raise ConfigError("k and relevant_k cannot exceed n_expert")

    min_prompts = positive_int(
        evaluation.get("min_prompts", 20), "evaluation.min_prompts"
    )
    min_workloads = positive_int(
        evaluation.get("min_workloads", 3), "evaluation.min_workloads"
    )
    if min_workloads < 3:
        raise ConfigError("evaluation.min_workloads must be at least 3")

    outputs = [output, summary, checksums]
    if len(set(outputs)) != len(outputs):
        raise ConfigError("output, run_summary and checksums must be distinct")

    protected = {path.resolve(), *(item["path"] for item in sessions)}
    if any(item in protected for item in outputs):
        raise ConfigError("an output would overwrite an input or config")

    for item in outputs:
        if item.exists():
            raise ConfigError(f"refusing to overwrite existing output: {item}")

    return {
        "model_id": model_id.strip(),
        "n_layer": n_layer,
        "n_expert": n_expert,
        "categories": tuple(EXPECTED_CATEGORIES),
        "sessions": sessions,
        "output": output,
        "summary": summary,
        "checksums": checksums,
        "k": k,
        "relevant_k": relevant_k,
        "min_prompts": min_prompts,
        "min_workloads": min_workloads,
    }


def load_sessions(settings):
    loaded = []
    all_records = []
    all_workload_ids = set()
    source_files = []

    for item in settings["sessions"]:
        path = item["path"]
        records, workload_ids, activation_count = builder.load_records(
            path, settings["n_layer"], settings["n_expert"]
        )

        if len(records) != 7:
            raise ConfigError(
                f"{path.name}: expected exactly 7 prompt records, "
                f"found {len(records)}"
            )

        if len(workload_ids) != 1:
            raise ConfigError(
                f"{path.name} must contain exactly one workload_id"
            )

        workload_id = next(iter(workload_ids))
        categories = {record["labels"]["task_category"] for record in records}
        if categories != {item["task_category"]}:
            raise ConfigError(
                f"{path.name}: configured category does not match JSONL labels"
            )

        if workload_id != item["session_id"]:
            raise ConfigError(
                f"{path.name}: configured session_id does not match "
                f"workload_id {workload_id!r}"
            )

        if workload_id in all_workload_ids:
            raise ConfigError(f"duplicate workload_id: {workload_id}")
        all_workload_ids.add(workload_id)

        labels = {
            tuple(record["labels"][field] for field in (
                "language", "framework", "workload_phase"
            ))
            for record in records
        }
        if len(labels) != 1:
            raise ConfigError(
                f"{path.name}: non-category labels vary within session"
            )

        loaded.append({
            "path": str(path),
            "sha256": sha256(path),
            "session_id": item["session_id"],
            "workload_id": workload_id,
            "task_category": item["task_category"],
            "records": records,
            "prompt_count": len(records),
            "activation_count": activation_count,
        })

        all_records.extend(records)
        source_files.append({
            "file": str(path),
            "file_name": path.name,
            "sha256": sha256(path),
            "session_id": item["session_id"],
            "workload_id": workload_id,
            "task_category": item["task_category"],
            "prompt_records": len(records),
            "activation_events": activation_count,
        })

    label_sets = {
        tuple(record["labels"][field] for field in (
            "language", "framework", "workload_phase"
        ))
        for record in all_records
    }
    if len(label_sets) != 1:
        raise ConfigError(
            "all sessions must share language, framework and workload_phase"
        )

    return loaded, source_files


def main():
    parser = argparse.ArgumentParser(
        description="Run PWEA leave-one-task-category-out evaluation."
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    created = []
    try:
        config_path = args.config.expanduser().resolve()
        settings = load_config(config_path)

        print("=== LOAD AND VALIDATE CATEGORY SESSIONS ===", flush=True)
        sessions, source_files = load_sessions(settings)

        protocol_labels = {
            field: sessions[0]["records"][0]["labels"][field]
            for field in ("language", "framework", "workload_phase")
        }
        expected_protocol = {
            "language": "Python",
            "framework": "none",
            "workload_phase": "inference",
        }
        if protocol_labels != expected_protocol:
            raise ConfigError(
                "source data does not match the fixed protocol; "
                f"expected {expected_protocol}, got {protocol_labels}"
            )

        counts = {
            category: sum(
                item["task_category"] == category for item in sessions
            )
            for category in settings["categories"]
        }
        print("Session coverage:", counts)
        print("Prompt records:", sum(item["prompt_count"] for item in sessions))
        print(
            "Activation events:",
            sum(item["activation_count"] for item in sessions),
        )

        print("\n=== RUN CATEGORY-TRANSFER EVALUATION ===", flush=True)
        result = evaluator.evaluate_category_transfer(
            sessions,
            settings["categories"],
            settings["n_layer"],
            settings["n_expert"],
            k=settings["k"],
            relevant_k=settings["relevant_k"],
            min_prompts=settings["min_prompts"],
            min_workloads=settings["min_workloads"],
        )

        result["model"] = {
            "model_id": settings["model_id"],
            "n_layer": settings["n_layer"],
            "n_expert": settings["n_expert"],
        }
        result["protocol"] = {
            **protocol_labels,
            "minimum_sessions_per_category": 3,
            "minimum_training_prompts": settings["min_prompts"],
            "minimum_training_sessions": settings["min_workloads"],
        }
        result["source_files"] = source_files
        result["config_sha256"] = sha256(config_path)
        result["evaluator_sha256"] = sha256(
            Path(evaluator.__file__).resolve()
        )
        result["builder_sha256"] = sha256(
            Path(builder.__file__).resolve()
        )

        settings["output"].parent.mkdir(parents=True, exist_ok=True)
        settings["output"].write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        created.append(settings["output"])

        summary = {
            "artifact_type": "pwea_category_transfer_run_summary",
            "schema_version": 1,
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "evaluation_file": str(settings["output"]),
            "evaluation_sha256": sha256(settings["output"]),
            "source_files": source_files,
            "runner_sha256": sha256(Path(__file__).resolve()),
            "evaluator_sha256": result["evaluator_sha256"],
            "builder_sha256": result["builder_sha256"],
        }

        settings["summary"].parent.mkdir(parents=True, exist_ok=True)
        settings["summary"].write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        created.append(settings["summary"])

        checksum_paths = [
            config_path,
            *(Path(item["path"]) for item in sessions),
            Path(__file__).resolve(),
            Path(evaluator.__file__).resolve(),
            Path(builder.__file__).resolve(),
            settings["output"],
            settings["summary"],
        ]
        settings["checksums"].parent.mkdir(parents=True, exist_ok=True)
        settings["checksums"].write_text(
            "".join(
                f"{sha256(path)}  {path}\n" for path in checksum_paths
            ),
            encoding="utf-8",
        )
        created.append(settings["checksums"])

        print("\n=== CATEGORY-TRANSFER RUN COMPLETED ===")
        print("Evaluation:", settings["output"])
        print("Summary:", settings["summary"])
        print("Checksums:", settings["checksums"])
        return 0

    except (
        ConfigError,
        builder.ValidationError,
        OSError,
        ValueError,
        TypeError,
    ) as exc:
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
                print(f"Rolled back: {path}", file=sys.stderr)
            except OSError as cleanup_error:
                print(
                    f"Rollback warning for {path}: {cleanup_error}",
                    file=sys.stderr,
                )
        print(f"Category-transfer runner error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
