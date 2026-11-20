#!/usr/bin/env python3
"""Reproducible prompt-level PWEA category-transfer evaluation."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import build_profile as builder
import evaluate_profile as ranking
import evaluate_category_transfer as category_validation
import evaluate_prompt_transfer as evaluator
import run_category_transfer as base_runner


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Run prompt-level leave-one-category-out PWEA evaluation."
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    created = []
    try:
        config_path = args.config.expanduser().resolve()
        settings = base_runner.load_config(config_path)

        print("=== LOAD AND VALIDATE ORIGINAL COHORT ===", flush=True)
        sessions, source_files = base_runner.load_sessions(settings)

        if len(sessions) != 25:
            raise ValueError(f"expected 25 sessions, found {len(sessions)}")

        prompt_count = sum(item["prompt_count"] for item in sessions)
        if prompt_count != 175:
            raise ValueError(f"expected 175 prompts, found {prompt_count}")

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
            raise ValueError(
                f"unexpected protocol labels: {protocol_labels}"
            )

        print("Sessions:", len(sessions))
        print("Prompts:", prompt_count)
        print("Activation events:", sum(
            item["activation_count"] for item in sessions
        ))

        print("\n=== RUN PROMPT-LEVEL CATEGORY TRANSFER ===", flush=True)
        result = evaluator.evaluate_prompt_transfer(
            sessions,
            settings["categories"],
            settings["n_layer"],
            settings["n_expert"],
            k=settings["k"],
            relevant_k=settings["relevant_k"],
            min_prompts=settings["min_prompts"],
            min_workloads=settings["min_workloads"],
        )

        if result["total_test_prompts"] != 175:
            raise ValueError(
                "prompt-level evaluation did not test exactly 175 prompts"
            )

        result["model"] = {
            "model_id": settings["model_id"],
            "n_layer": settings["n_layer"],
            "n_expert": settings["n_expert"],
        }
        result["protocol"] = {
            **protocol_labels,
            "sessions": 25,
            "prompts": 175,
            "sessions_per_category": 5,
            "prompts_per_session": 7,
            "training_categories_per_fold": 4,
            "training_sessions_per_fold": 20,
            "training_prompts_per_fold": 140,
            "heldout_sessions_per_fold": 5,
            "heldout_prompts_per_fold": 35,
            "minimum_training_prompts": settings["min_prompts"],
            "minimum_training_sessions": settings["min_workloads"],
        }
        result["source_files"] = source_files
        result["config_sha256"] = sha256(config_path)
        result["evaluator_sha256"] = sha256(Path(evaluator.__file__).resolve())
        result["category_validation_sha256"] = sha256(
            Path(category_validation.__file__).resolve()
        )
        result["builder_sha256"] = sha256(Path(builder.__file__).resolve())
        result["ranking_sha256"] = sha256(Path(ranking.__file__).resolve())
        result["runner_sha256"] = sha256(Path(__file__).resolve())

        settings["output"].parent.mkdir(parents=True, exist_ok=True)
        settings["output"].write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        created.append(settings["output"])

        summary = {
            "artifact_type": "pwea_prompt_transfer_run_summary",
            "schema_version": 1,
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "evaluation_file": str(settings["output"]),
            "evaluation_sha256": sha256(settings["output"]),
            "total_test_prompts": result["total_test_prompts"],
            "fold_count": result["fold_count"],
            "source_files": source_files,
            "runner_sha256": result["runner_sha256"],
            "evaluator_sha256": result["evaluator_sha256"],
            "category_validation_sha256": result["category_validation_sha256"],
            "builder_sha256": result["builder_sha256"],
            "ranking_sha256": result["ranking_sha256"],
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
            Path(category_validation.__file__).resolve(),
            Path(builder.__file__).resolve(),
            Path(ranking.__file__).resolve(),
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

        print("\n=== PROMPT-LEVEL EVALUATION COMPLETED ===")
        print("Evaluation:", settings["output"])
        print("Summary:", settings["summary"])
        print("Checksums:", settings["checksums"])
        print("Test prompts:", result["total_test_prompts"])
        print("\nPWEA:", result["overall"]["pwea"])
        print("Frequency baseline:", result["overall"]["frequency_baseline"])
        return 0

    except (
        base_runner.ConfigError,
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
        print(f"Prompt-transfer runner error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
