#!/usr/bin/env python3
"""Controlled one-session PWEA telemetry collection pilot."""

import argparse
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATS_RE = re.compile(
    r"\[prompt\s+(\d+)\s+tok in\s+([0-9.]+)s.*?"
    r"\|\s+gen\s+(\d+)\s+tok in\s+([0-9.]+)s"
)
TIMEOUT_SECONDS = 7200


class PilotError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotError(f"Cannot read JSON {path}: {exc}") from exc


def load_prompts(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 7:
        raise PilotError(
            f"Expected exactly 7 physical prompt lines; found {len(lines)}"
        )
    if any(not line.strip() for line in lines):
        raise PilotError("Prompt suite contains an empty prompt line")
    return lines


def validate_config(config_path):
    cfg = load_json(config_path)

    if cfg.get("schema_version") != 1:
        raise PilotError("Unsupported configuration schema")

    model = cfg["model"]
    engine = cfg["engine"]
    runtime = cfg["runtime"]
    outputs = cfg["outputs"]
    suite = cfg["prompt_suite"]

    engine_path = resolve(engine["path"])
    model_path = resolve(model["path"])
    suite_path = resolve(suite["path"])

    for path, label in (
        (engine_path, "engine"),
        (model_path, "model"),
        (suite_path, "prompt suite"),
    ):
        if not path.is_file():
            raise PilotError(f"{label} file does not exist: {path}")

    if sha256(engine_path) != engine["sha256"]:
        raise PilotError("Engine SHA256 mismatch")
    if sha256(model_path) != model["sha256"]:
        raise PilotError("Model SHA256 mismatch")
    if sha256(suite_path) != suite["sha256"]:
        raise PilotError("Prompt-suite SHA256 mismatch")

    if model["n_layer"] != 48 or model["n_expert"] != 128:
        raise PilotError("Unexpected model dimensions")
    if runtime["telemetry_interval"] != 2:
        raise PilotError("Unexpected telemetry interval")
    if engine["device"] != "sim":
        raise PilotError("Pilot must use the configured sim device")

    prompts = load_prompts(suite_path)
    if len(prompts) != cfg["prompt_count"]:
        raise PilotError("Prompt count does not match configuration")

    output_paths = {key: resolve(value) for key, value in outputs.items()}
    if len(set(output_paths.values())) != len(output_paths):
        raise PilotError("Output paths must be distinct")

    for path in output_paths.values():
        if path.exists():
            raise PilotError(f"Refusing to overwrite existing output: {path}")

    return cfg, prompts, engine_path, model_path, output_paths


def reader_thread(stream, messages):
    try:
        for line in iter(stream.readline, ""):
            messages.put(line)
    finally:
        messages.put(None)


def wait_for_prompt(process, messages, prompt_index, stderr_file, max_new_tokens):
    deadline = time.monotonic() + TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        try:
            item = messages.get(timeout=1)
        except queue.Empty:
            if process.poll() is not None:
                raise PilotError(
                    f"Engine exited during prompt {prompt_index + 1}"
                )
            continue

        if item is None:
            raise PilotError(
                f"Engine stderr closed during prompt {prompt_index + 1}"
            )

        stderr_file.write(item)
        stderr_file.flush()

        if "[context full]" in item:
            raise PilotError(
                f"Context overflow during prompt {prompt_index + 1}"
            )

        match = STATS_RE.search(item)
        if match:
            prompt_tokens = int(match.group(1))
            generated_tokens = int(match.group(3))
            if prompt_tokens <= 0 or generated_tokens <= 0:
                raise PilotError(
                    f"Invalid token counts during prompt {prompt_index + 1}"
                )
            return {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "hit_token_limit": generated_tokens == max_new_tokens,
            }

    raise PilotError(
        f"Timed out waiting for prompt {prompt_index + 1} completion"
    )


def validate_telemetry(path, cfg):
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise PilotError(
                f"Invalid telemetry JSON on line {line_number}: {exc}"
            ) from exc

    if len(records) != 7:
        raise PilotError(f"Expected 7 telemetry records; found {len(records)}")

    expected_workload = {
        "workload_id": cfg["session_id"],
        "language": cfg["workload"]["language"],
        "framework": cfg["workload"]["framework"],
        "task_category": cfg["task_category"],
        "workload_phase": cfg["workload"]["workload_phase"],
    }

    prompt_ids = []
    expected_windows = [(1, 2, 2), (3, 4, 2), (5, 6, 2), (7, 7, 1)]

    for record in records:
        if record.get("record_type") != "etf_prompt":
            raise PilotError("Unexpected telemetry record_type")
        if record.get("schema_version") != 1:
            raise PilotError("Unexpected telemetry schema_version")

        prompt_id = record.get("prompt_id")
        if not isinstance(prompt_id, int):
            raise PilotError("prompt_id must be an integer")
        prompt_ids.append(prompt_id)

        if record.get("workload") != expected_workload:
            raise PilotError(f"Incorrect workload metadata at prompt {prompt_id}")

        window = record.get("window")
        if not isinstance(window, dict):
            raise PilotError(f"Missing window object at prompt {prompt_id}")

        if window.get("dropped_events") != 0:
            raise PilotError(f"Dropped telemetry events at prompt {prompt_id}")

        activations = record.get("activations")
        if not isinstance(activations, list):
            raise PilotError(f"Invalid activations array at prompt {prompt_id}")

        for event in activations:
            if not isinstance(event, dict):
                raise PilotError("Activation event must be an object")
            for field in ("layer", "token_position", "expert_id"):
                if not isinstance(event.get(field), int):
                    raise PilotError(f"Invalid activation field: {field}")
            if not 0 <= event["layer"] < cfg["model"]["n_layer"]:
                raise PilotError("Activation layer out of bounds")
            if not 0 <= event["expert_id"] < cfg["model"]["n_expert"]:
                raise PilotError("Activation expert out of bounds")
            weight = event.get("routing_weight")
            if weight is not None and (
                not isinstance(weight, (int, float)) or not 0 <= weight <= 1.0001
            ):
                raise PilotError("Invalid routing_weight")

    if prompt_ids != list(range(1, 8)):
        raise PilotError(f"Unexpected prompt IDs: {prompt_ids}")

    for first_id, last_id, count in expected_windows:
        members = [
            r for r in records
            if r["window"].get("first_prompt_id") == first_id
            and r["window"].get("last_prompt_id") == last_id
            and r["window"].get("completed_prompts") == count
        ]
        if len(members) != count:
            raise PilotError(
                f"Telemetry window {first_id}-{last_id} is inconsistent"
            )

    return {
        "record_count": len(records),
        "prompt_ids": prompt_ids,
        "window_sizes": [2, 2, 2, 1],
        "activation_counts": [len(r["activations"]) for r in records],
        "total_activations": sum(len(r["activations"]) for r in records),
        "dropped_events": 0,
    }


def run_session(cfg, prompts, engine_path, model_path, paths):
    telemetry = paths["telemetry"]
    stdout_path = paths["stdout"]
    stderr_path = paths["stderr"]

    telemetry.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)

    r = cfg["runtime"]
    w = cfg["workload"]

    command = [
        str(engine_path),
        "-m", str(model_path),
        "-i",
        "-c", str(r["context"]),
        "-n", str(r["max_new_tokens"]),
        "-t", str(r["threads"]),
        "-b", str(r["batch"]),
        "--temp", str(r["temperature"]),
        "--top-k", str(r["top_k"]),
        "--top-p", str(r["top_p"]),
        "--min-p", str(r["min_p"]),
        "--repeat-penalty", str(r["repeat_penalty"]),
        "--seed", str(cfg["seed"]),
        "--device", cfg["engine"]["device"],
        "--workload-id", cfg["session_id"],
        "--language", w["language"],
        "--framework", w["framework"],
        "--task-category", cfg["task_category"],
        "--workload-phase", w["workload_phase"],
        "--etf-log", str(telemetry),
    ]

    started = time.time()
    prompt_results = []

    with stdout_path.open("w", encoding="utf-8") as stdout_file, \
         stderr_path.open("w", encoding="utf-8") as stderr_file:

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=stdout_file,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        messages = queue.Queue()
        thread = threading.Thread(
            target=reader_thread,
            args=(process.stderr, messages),
            daemon=True,
        )
        thread.start()

        try:
            for index, prompt in enumerate(prompts):
                process.stdin.write(prompt + "\n")
                process.stdin.flush()

                result = wait_for_prompt(
                    process, messages, index, stderr_file,
                    cfg["runtime"]["max_new_tokens"],
                )
                prompt_results.append(result)

                if index < len(prompts) - 1:
                    process.stdin.write("/reset\n")
                    process.stdin.flush()

            process.stdin.write("/exit\n")
            process.stdin.flush()
            process.stdin.close()

            while True:
                item = messages.get(timeout=TIMEOUT_SECONDS)
                if item is None:
                    break
                stderr_file.write(item)
                stderr_file.flush()

            return_code = process.wait(timeout=60)
            if return_code != 0:
                raise PilotError(f"Engine exited with status {return_code}")

        except Exception:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        finally:
            if not process.stderr.closed:
                process.stderr.close()

    telemetry_report = validate_telemetry(telemetry, cfg)

    report = {
        "schema_version": 1,
        "status": "validated",
        "session_id": cfg["session_id"],
        "task_category": cfg["task_category"],
        "seed": cfg["seed"],
        "started_unix": started,
        "finished_unix": time.time(),
        "duration_seconds": round(time.time() - started, 3),
        "prompt_results": prompt_results,
        "token_limit_reached": any(
            item["hit_token_limit"] for item in prompt_results
        ),
        "telemetry": telemetry_report,
        "hashes": {
            "engine": sha256(engine_path),
            "model": sha256(model_path),
            "prompt_suite": sha256(resolve(cfg["prompt_suite"]["path"])),
            "telemetry": sha256(telemetry),
            "stdout": sha256(stdout_path),
            "stderr": sha256(stderr_path),
        },
    }

    paths["report"].write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="pwea/configs/pilot_code_explanation_s01_20261114.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    cfg, prompts, engine_path, model_path, paths = validate_config(config_path)

    print("PILOT PREFLIGHT PASSED")
    print(f"Session: {cfg['session_id']}")
    print(f"Category: {cfg['task_category']}")
    print(f"Prompts: {len(prompts)}")
    print(f"Model dimensions: {cfg['model']['n_layer']} layers, "
          f"{cfg['model']['n_expert']} experts")
    print(f"Seed: {cfg['seed']}")
    print(f"Engine: {engine_path}")
    print(f"Model: {model_path}")
    print(f"Telemetry: {paths['telemetry']}")

    if args.dry_run:
        print("DRY RUN ONLY — no inference started.")
        return 0

    confirmation = input(
        "Type RUN PILOT to start the seven-prompt inference session: "
    )
    if confirmation != "RUN PILOT":
        print("Cancelled; no inference started.")
        return 2

    report = run_session(cfg, prompts, engine_path, model_path, paths)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotError as exc:
        print(f"PILOT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
