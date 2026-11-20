import json
from pathlib import Path

MODEL = Path.home() / "Downloads/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf"
SOURCE = Path("benchmarks/real_activations_q8k_full_20261118")
DEST = Path("benchmarks/real_activation_weight_replay_20261118")

CHUNK = 4 * 1024 * 1024

def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

weights = load_jsonl(DEST / "weights.jsonl")
replay = load_jsonl(DEST / "replay.jsonl")

weight_path = DEST / "weights.bin"
activation_path = SOURCE / "activations.f32"
activation_manifest = load_jsonl(SOURCE / "manifest.jsonl")

weight_size = weight_path.stat().st_size
model_size = MODEL.stat().st_size
activation_size = activation_path.stat().st_size

errors = []
verified_matrices = 0
verified_bytes = 0

# Check that extracted matrix regions are contiguous and in bounds.
ordered = sorted(weights, key=lambda x: x["replay_offset_bytes"])
cursor = 0

for w in ordered:
    start = int(w["replay_offset_bytes"])
    size = int(w["bytes"])

    if start != cursor:
        errors.append(("weight file gap/overlap", w["weight_id"], cursor, start))

    if start < 0 or start + size > weight_size:
        errors.append(("weight file bounds", w["weight_id"]))

    cursor = start + size

if cursor != weight_size:
    errors.append(("weight file final size", cursor, weight_size))

# Compare extracted bytes against original GGUF byte ranges.
with MODEL.open("rb") as mf, weight_path.open("rb") as wf:
    for w in weights:
        src = int(w["gguf_matrix_offset"])
        dst = int(w["replay_offset_bytes"])
        size = int(w["bytes"])

        if src < 0 or src + size > model_size:
            errors.append(("GGUF source bounds", w["weight_id"]))
            continue

        mf.seek(src)
        wf.seek(dst)

        remaining = size
        matrix_ok = True

        while remaining:
            n = min(CHUNK, remaining)
            a = mf.read(n)
            b = wf.read(n)

            if len(a) != n or len(b) != n:
                errors.append(("short read", w["weight_id"], remaining))
                matrix_ok = False
                break

            if a != b:
                errors.append(("byte mismatch", w["weight_id"], size - remaining))
                matrix_ok = False
                break

            remaining -= n

        if matrix_ok:
            verified_matrices += 1
            verified_bytes += size

# Validate every replay reference and activation byte range.
weight_by_id = {w["weight_id"]: w for w in weights}
unresolved = 0
activation_bounds_errors = 0
dimension_errors = 0

for r in replay:
    w = weight_by_id.get(r["weight_id"])

    if w is None:
        unresolved += 1
        continue

    if r["weight_type"] != w["type"]:
        dimension_errors += 1

    if r["weight_rows"] != w["rows"]:
        dimension_errors += 1

    if r["weight_row_bytes"] != w["row_bytes"]:
        dimension_errors += 1

    start = int(r["activation_offset_bytes"])
    size = int(r["activation_bytes"])

    if start < 0 or start + size > activation_size:
        activation_bounds_errors += 1

# Ensure replay record identities and activation offsets correspond
# to the original capture manifest in the same order.
identity_mismatches = 0

if len(replay) != len(activation_manifest):
    identity_mismatches += abs(len(replay) - len(activation_manifest))

for a, r in zip(activation_manifest, replay):
    if (
        int(a["record"]) != int(r["record"])
        or int(a["offset_bytes"]) != int(r["activation_offset_bytes"])
        or int(a["elements"]) != int(r["elements"])
        or a["operation"] != r["operation"]
    ):
        identity_mismatches += 1

print("===== BYTE-LEVEL EXTRACTION VALIDATION =====")
print("Matrices:", len(weights))
print("Matrices byte-identical:", verified_matrices)
print("Bytes verified:", verified_bytes)
print("Extracted file bytes:", weight_size)
print("GGUF source file bytes:", model_size)

print("\n===== REPLAY VALIDATION =====")
print("Replay records:", len(replay))
print("Unresolved weight references:", unresolved)
print("Activation bounds errors:", activation_bounds_errors)
print("Weight metadata errors:", dimension_errors)
print("Capture/replay identity mismatches:", identity_mismatches)

print("\n===== ERRORS =====")
print("Total errors:", len(errors))
for e in errors[:20]:
    print(e)

ok = (
    not errors
    and verified_matrices == len(weights)
    and verified_bytes == weight_size
    and unresolved == 0
    and activation_bounds_errors == 0
    and dimension_errors == 0
    and identity_mismatches == 0
)

print("\nBYTE-LEVEL REPLAY VALIDATION:", "PASS" if ok else "FAIL")
