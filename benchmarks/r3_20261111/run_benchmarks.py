#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks" / "r3_20261111"
MODEL = Path(
    os.environ.get(
        "CONTRAX_MODEL",
        str(Path.home() / "Downloads" / "Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf"),
    )
)
EXECUTABLE = ROOT / "qwen3moe"

PROMPT = "Write a Python function to calculate Fibonacci numbers."
SEED = 20261111
ROUNDS = 5

POLICIES = {
    "lru": ["--expert-policy", "lru"],
    "frequency_1024": [
        "--expert-policy", "frequency",
        "--expert-aging-interval", "1024",
    ],
    "frequency_4096": [
        "--expert-policy", "frequency",
        "--expert-aging-interval", "4096",
    ],
}

SERVING_RE = re.compile(
    r"serving:\s+SSD=(\d+)\s+RAM=(\d+)\s+VRAM=(\d+)"
    r"\s+successful=(\d+)\s+failed=(\d+)\s+accounting=(PASS|FAIL)"
)
GEN_RE = re.compile(
    r"gen\s+(\d+)\s+tok\s+in\s+([\d.]+)s\s+\(([\d.]+)\s+t/s\)"
)
ACQUIRE_RE = re.compile(r"acquires=(\d+)\s+releases=(\d+)")
VRAM_ALLOC_RE = re.compile(
    r"VRAM alloc:\s+failures=(\d+)\s+retries=(\d+)"
    r"\s+recovery_evictions=(\d+)\s+recovered=(\d+)"
    r"\s+unrecovered=(\d+)"
)
MIGRATION_RE = re.compile(
    r"^\s*(SSD->RAM|RAM->VRAM|VRAM->RAM|RAM->SSD|VRAM->SSD)"
    r"\s+n=(\d+)\s+fail=(\d+)\s+bytes=([\d.]+)\s+MiB",
    re.MULTILINE,
)


def timestamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def gpu_snapshot():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception as exc:
        return f"unavailable: {exc}"


def parse_output(output):
    result = {}

    serving = SERVING_RE.search(output)
    if serving:
        keys = ["served_ssd", "served_ram", "served_vram",
                "successful_acquires", "failed_acquires"]
        result.update(dict(zip(keys, map(int, serving.groups()[:5]))))
        result["serving_accounting"] = serving.group(6)

    generation = GEN_RE.search(output)
    if generation:
        result["generated_tokens"] = int(generation.group(1))
        result["generation_seconds"] = float(generation.group(2))
        result["generation_tokens_per_second"] = float(generation.group(3))

    acquire = ACQUIRE_RE.search(output)
    if acquire:
        result["acquires"] = int(acquire.group(1))
        result["releases"] = int(acquire.group(2))

    result["migrations"] = {}
    for match in MIGRATION_RE.finditer(output):
        path, count, failures, mib = match.groups()
        result["migrations"][path] = {
            "successes": int(count),
            "failures": int(failures),
            "successful_mib": float(mib),
        }

    result["vram_alloc_line"] = next(
        (line.strip() for line in output.splitlines()
         if "VRAM alloc:" in line),
        None,
    )

    return result


def run_one(phase, label, policy, index):
    directory = BASE / phase
    directory.mkdir(parents=True, exist_ok=True)

    stem = f"{index:02d}_{label}"
    stdout_path = directory / f"{stem}.stdout.log"
    stderr_path = directory / f"{stem}.stderr.log"
    time_path = directory / f"{stem}.time.txt"
    event_path = directory / f"{stem}.residency.log"

    existing = [
        p for p in (stdout_path, stderr_path, time_path, event_path)
        if p.exists()
    ]
    if existing:
        raise RuntimeError(
            "Refusing to overwrite existing benchmark artifacts: "
            + ", ".join(str(p) for p in existing)
        )

    command = [
        "/usr/bin/time", "-v", "-o", str(time_path),
        str(EXECUTABLE),
        "-m", str(MODEL),
        "-p", PROMPT,
        "-n", "16",
        "-c", "512",
        "-t", "8",
        "--temp", "0",
        "--device", "cuda",
        "--expert-tier", "vram",
        "--ram-budget-mb", "512",
        "--vram-budget-mb", "512",
        "--vram-reserve-mb", "1024",
        *POLICIES[policy],
        "--expert-log", str(event_path),
        "--expert-stats",
    ]

    record = {
        "phase": phase,
        "index": index,
        "policy": policy,
        "started_at": timestamp(),
        "gpu_before": gpu_snapshot(),
        "command": command,
        "stdout_file": str(stdout_path.relative_to(ROOT)),
        "stderr_file": str(stderr_path.relative_to(ROOT)),
        "time_file": str(time_path.relative_to(ROOT)),
        "event_file": str(event_path.relative_to(ROOT)),
    }

    print(f"\n[{phase}] {label}: starting {policy}", flush=True)
    start = time.perf_counter()

    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
            )
            record["return_code"] = completed.returncode
        except Exception as exc:
            record["return_code"] = -1
            record["exception"] = repr(exc)

    record["wall_seconds"] = round(time.perf_counter() - start, 4)
    record["finished_at"] = timestamp()
    record["gpu_after"] = gpu_snapshot()

    stdout = stdout_path.read_text(errors="replace")
    stderr = stderr_path.read_text(errors="replace")
    parsed = parse_output(stdout + "\n" + stderr)
    record.update(parsed)

    served_total = sum(
        record.get(key, 0)
        for key in ("served_ssd", "served_ram", "served_vram")
    )

    vram_line = record.get("vram_alloc_line")
    vram_match = VRAM_ALLOC_RE.search(vram_line or "")
    vram_alloc_ok = (
        vram_match is not None
        and all(int(value) == 0 for value in vram_match.groups())
    )

    valid = (
        record["return_code"] == 0
        and record.get("generated_tokens") == 16
        and record.get("serving_accounting") == "PASS"
        and record.get("successful_acquires", -1) > 0
        and record.get("failed_acquires") == 0
        and record.get("acquires") == record.get("successful_acquires")
        and record.get("releases") == record.get("successful_acquires")
        and served_total == record.get("successful_acquires")
        and vram_alloc_ok
        and event_path.exists()
        and event_path.stat().st_size > 0
    )
    record["validation"] = "PASS" if valid else "FAIL"

    results_path = BASE / "results.jsonl"
    with results_path.open("a") as results_file:
        results_file.write(json.dumps(record, sort_keys=True) + "\n")

    print(
        f"[{phase}] {label}: rc={record['return_code']} "
        f"wall={record['wall_seconds']}s "
        f"generation={record.get('generation_tokens_per_second', 'N/A')} tok/s "
        f"serving={record.get('serving_accounting', 'MISSING')} "
        f"validation={record['validation']}",
        flush=True,
    )

    if not valid:
        print(f"Run failed validation. Inspect {stdout_path} and {stderr_path}.")
        sys.exit(2)

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["warmup", "trials"])
    args = parser.parse_args()

    if not EXECUTABLE.is_file() or not MODEL.is_file():
        sys.exit("Executable or model file is missing.")
    if not MODEL.stat().st_size:
        sys.exit("Model file is empty.")

    rng = random.Random(SEED)

    if args.phase == "warmup":
        order = list(POLICIES)
        rng.shuffle(order)
        print("Warm-up order:", " -> ".join(order))
        for i, policy in enumerate(order, 1):
            run_one("warmup", policy, policy, i)
            time.sleep(15)

    else:
        for round_number in range(1, ROUNDS + 1):
            order = list(POLICIES)
            rng.shuffle(order)
            print(f"\nRound {round_number} order:", " -> ".join(order))
            for policy in order:
                label = f"round{round_number}_{policy}"
                index = (round_number - 1) * len(POLICIES) + order.index(policy) + 1
                run_one("trials", label, policy, index)
                time.sleep(15)

    print(f"\n{args.phase} phase completed.")


if __name__ == "__main__":
    main()
