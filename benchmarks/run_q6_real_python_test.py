import subprocess
import time
import hashlib
import re
from pathlib import Path

ROOT = Path.cwd()
DIR = ROOT / "benchmarks/q6_integration_20261118"
MODEL = Path.home() / "Downloads/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf"

OUT = ROOT / "benchmarks/q6_real_python_20261120"
OUT.mkdir(parents=True, exist_ok=True)

BINARIES = {
    "fallback": DIR / "qwen3moe_q6_fallback",
    "vnni": DIR / "qwen3moe_q6_vnni",
}

PROMPT = """Write a Python 3 program that implements a task manager.
It must define a Task class with title, priority and completed fields.
Implement functions to add a task, mark a task complete, remove a task,
and display all pending tasks sorted by priority.
Include input validation and a small demonstration in a main block.
Use only the Python standard library. Provide clean, executable code.
"""

if not MODEL.is_file():
    raise SystemExit(f"Model not found: {MODEL}")

for name, binary in BINARIES.items():
    if not binary.is_file():
        raise SystemExit(f"Binary not found: {binary}")

common = [
    "-m", str(MODEL),
    "-c", "512",
    "-b", "4",
    "-t", "8",
    "-n", "96",
    "-p", PROMPT,
    "--temp", "0",
    "--top-k", "1",
    "--seed", "42",
    "--device", "sim",
    "--expert-tier", "ssd",
    "--no-etf-collection",
]

# Alternating order reduces systematic ordering bias.
schedule = [
    ("fallback", 1),
    ("vnni", 1),
    ("vnni", 2),
    ("fallback", 2),
]

results = []

for name, repetition in schedule:
    tag = f"{name}_rep{repetition}"
    stdout_path = OUT / f"{tag}.stdout"
    stderr_path = OUT / f"{tag}.stderr"
    time_path = OUT / f"{tag}.time"

    cmd = [
        "/usr/bin/time", "-v", "-o", str(time_path),
        str(BINARIES[name]), *common
    ]

    print(f"\n===== RUN {tag} =====", flush=True)
    start = time.perf_counter()

    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.run(
            cmd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )

    elapsed = time.perf_counter() - start
    output = stdout_path.read_bytes()
    digest = hashlib.sha256(output).hexdigest()

    time_text = time_path.read_text() if time_path.exists() else ""
    rss_match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        time_text
    )

    result = {
        "binary": name,
        "repetition": repetition,
        "exit": proc.returncode,
        "elapsed": elapsed,
        "rss_kb": int(rss_match.group(1)) if rss_match else None,
        "stdout_bytes": len(output),
        "stdout_sha256": digest,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "time": str(time_path),
    }
    results.append(result)

    print("Exit:", proc.returncode)
    print(f"Wall time: {elapsed:.3f} seconds")
    print("Maximum RSS KiB:", result["rss_kb"])
    print("Output bytes:", len(output))
    print("Output SHA256:", digest)
    print("Logs:", stdout_path, stderr_path, time_path, flush=True)

summary = OUT / "summary.txt"

with summary.open("w") as f:
    for r in results:
        f.write(
            f"{r['binary']} rep={r['repetition']} "
            f"exit={r['exit']} elapsed={r['elapsed']:.3f} "
            f"rss_kb={r['rss_kb']} "
            f"bytes={r['stdout_bytes']} "
            f"sha256={r['stdout_sha256']}\n"
        )

print("\n========== SUMMARY ==========")
for r in results:
    print(
        f"{r['binary']:8} rep={r['repetition']} "
        f"exit={r['exit']} elapsed={r['elapsed']:.2f}s "
        f"RSS={r['rss_kb']} KiB"
    )

for name in BINARIES:
    selected = [r for r in results if r["binary"] == name]
    times = [r["elapsed"] for r in selected]
    print(
        f"{name} median wall time: "
        f"{sorted(times)[len(times)//2]:.3f}s"
    )

fallback = [r for r in results if r["binary"] == "fallback"]
vnni = [r for r in results if r["binary"] == "vnni"]

if all(r["exit"] == 0 for r in results):
    print("\nAll four inference runs completed successfully.")
else:
    print("\nWARNING: At least one inference run failed.")

if len(fallback) == 2 and len(vnni) == 2:
    print(
        "Output checksums identical by corresponding repetition:",
        fallback[0]["stdout_sha256"] == vnni[0]["stdout_sha256"]
        and fallback[1]["stdout_sha256"] == vnni[1]["stdout_sha256"]
    )

print("Summary file:", summary)
