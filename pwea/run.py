#!/usr/bin/env python3
"""Single-JSON-config orchestration for PWEA."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


class ConfigError(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def resolve(value, base, name):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty path")
    p = Path(value).expanduser()
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def load_config(path):
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON at line {e.lineno}: {e.msg}") from e

    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise ConfigError("Expected a JSON object with schema_version: 1")

    model = cfg.get("model")
    if not isinstance(model, dict):
        raise ConfigError("model must be an object")

    model_id = model.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError("model.model_id must be a non-empty string")

    n_layer = positive_int(model.get("n_layer"), "model.n_layer")
    n_expert = positive_int(model.get("n_expert"), "model.n_expert")
    if n_expert <= 1:
        raise ConfigError("model.n_expert must exceed 1")

    raw_inputs = cfg.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ConfigError("inputs must be a non-empty array")

    inputs = [resolve(x, path.parent, f"inputs[{i}]")
              for i, x in enumerate(raw_inputs)]

    if len(set(inputs)) != len(inputs):
        raise ConfigError("Duplicate input paths")

    for p in inputs:
        if not p.is_file():
            raise ConfigError(f"Input does not exist: {p}")

    profile = cfg.get("profile")
    if not isinstance(profile, dict):
        raise ConfigError("profile must be an object")

    profile_output = resolve(profile.get("output"), path.parent, "profile.output")
    min_prompts = positive_int(profile.get("min_prompts", 20), "profile.min_prompts")
    min_workloads = positive_int(profile.get("min_workloads", 3), "profile.min_workloads")

    evaluation = cfg.get("evaluation", {"enabled": False})
    if not isinstance(evaluation, dict):
        raise ConfigError("evaluation must be an object")

    enabled = evaluation.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("evaluation.enabled must be boolean")

    eval_output = None
    k = None
    if enabled:
        if len(inputs) < 3:
            raise ConfigError("LOSO evaluation requires at least 3 sessions")
        eval_output = resolve(evaluation.get("output"), path.parent, "evaluation.output")
        k = positive_int(evaluation.get("k", 5), "evaluation.k")
        relevant_k = max(1, (n_expert + 9) // 10)
        if k > relevant_k:
            raise ConfigError(f"evaluation.k cannot exceed {relevant_k}")

    summary_output = resolve(
        cfg.get("run_summary", str(profile_output.parent / "pwea_run_summary.json")),
        path.parent, "run_summary"
    )
    checksum_output = summary_output.parent / "PWEA_RUN_SHA256SUMS.txt"

    outputs = [profile_output, summary_output, checksum_output]
    if eval_output is not None:
        outputs.append(eval_output)

    if len(set(outputs)) != len(outputs):
        raise ConfigError("All output paths must be distinct")

    protected = set(inputs) | {path.resolve()}
    if any(p in protected for p in outputs):
        raise ConfigError("An output would overwrite an input or config")

    for p in outputs:
        if p.exists():
            raise ConfigError(f"Refusing to overwrite existing output: {p}")

    return {
        "model_id": model_id,
        "n_layer": n_layer,
        "n_expert": n_expert,
        "inputs": inputs,
        "profile_output": profile_output,
        "min_prompts": min_prompts,
        "min_workloads": min_workloads,
        "evaluation_enabled": enabled,
        "evaluation_output": eval_output,
        "k": k,
        "summary_output": summary_output,
        "checksum_output": checksum_output,
    }


def execute(command, label):
    print(f"\n=== {label} ===", flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Run PWEA profile construction and optional LOSO evaluation."
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()

    created_outputs = []

    try:
        s = load_config(config_path)
        pwea_dir = Path(__file__).resolve().parent
        builder = pwea_dir / "build_profile.py"
        evaluator = pwea_dir / "evaluate_profile.py"
        runner = Path(__file__).resolve()

        build_cmd = [sys.executable, str(builder)]
        for p in s["inputs"]:
            build_cmd += ["--input", str(p)]
        build_cmd += [
            "--output", str(s["profile_output"]),
            "--model-id", s["model_id"],
            "--n-layer", str(s["n_layer"]),
            "--n-expert", str(s["n_expert"]),
            "--min-prompts", str(s["min_prompts"]),
            "--min-workloads", str(s["min_workloads"]),
        ]
        created_outputs.append(s["profile_output"])
        execute(build_cmd, "Build PWEA profile")

        if s["evaluation_enabled"]:
            eval_cmd = [sys.executable, str(evaluator)]
            for p in s["inputs"]:
                eval_cmd += ["--input", str(p)]
            eval_cmd += [
                "--output", str(s["evaluation_output"]),
                "--model-id", s["model_id"],
                "--n-layer", str(s["n_layer"]),
                "--n-expert", str(s["n_expert"]),
                "--k", str(s["k"]),
            ]
            created_outputs.append(s["evaluation_output"])
            execute(eval_cmd, "Run LOSO evaluation")

        artifacts = [s["profile_output"]]
        if s["evaluation_output"]:
            artifacts.append(s["evaluation_output"])

        summary = {
            "artifact_type": "pwea_cli_run_summary",
            "schema_version": 1,
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "model": {
                "model_id": s["model_id"],
                "n_layer": s["n_layer"],
                "n_expert": s["n_expert"],
            },
            "inputs": [
                {"file": str(p), "sha256": sha256(p)} for p in s["inputs"]
            ],
            "profile": {
                "file": str(s["profile_output"]),
                "sha256": sha256(s["profile_output"]),
            },
            "evaluation": (
                None if not s["evaluation_enabled"] else {
                    "file": str(s["evaluation_output"]),
                    "sha256": sha256(s["evaluation_output"]),
                    "k": s["k"],
                }
            ),
            "runner_sha256": sha256(runner),
            "builder_sha256": sha256(builder),
            "evaluator_sha256": sha256(evaluator) if s["evaluation_enabled"] else None,
        }

        created_outputs.append(s["summary_output"])
        s["summary_output"].parent.mkdir(parents=True, exist_ok=True)
        s["summary_output"].write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        checksum_paths = [
            config_path, *s["inputs"], builder, runner, *artifacts,
            s["summary_output"],
        ]
        if s["evaluation_enabled"]:
            checksum_paths.append(evaluator)

        created_outputs.append(s["checksum_output"])
        s["checksum_output"].write_text(
            "".join(f"{sha256(p)}  {p}\n" for p in checksum_paths),
            encoding="utf-8",
        )

        print("\n=== PWEA run completed ===")
        print("Profile:", s["profile_output"])
        if s["evaluation_output"]:
            print("Evaluation:", s["evaluation_output"])
        print("Summary:", s["summary_output"])
        print("Checksums:", s["checksum_output"])
        return 0

    except (ConfigError, OSError, RuntimeError) as e:
        for output in reversed(created_outputs):
            try:
                if output.is_file() or output.is_symlink():
                    output.unlink()
                    print(f"Rolled back: {output}", file=sys.stderr)
            except OSError as cleanup_error:
                print(
                    f"Rollback warning for {output}: {cleanup_error}",
                    file=sys.stderr,
                )
        print(f"PWEA CLI error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
