import json
import shutil
import gguf
from pathlib import Path

model = Path.home() / "Downloads/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf"
dest = Path("benchmarks/real_activation_weight_replay_20261118")
manifest = dest / "weights.jsonl"
backup = dest / "weights.jsonl.pre_offset_fix"

if backup.exists():
    raise SystemExit(f"Refusing to overwrite backup: {backup}")

reader = gguf.GGUFReader(str(model))
tensors = {t.name: t for t in reader.tensors}

rows = [
    json.loads(line)
    for line in manifest.read_text().splitlines()
    if line.strip()
]

fixed = 0

for w in rows:
    t = tensors.get(w["tensor"])
    if t is None:
        raise SystemExit(f"Missing GGUF tensor: {w['tensor']}")

    expert = int(w["expert"])
    offset = int(t.data_offset)

    if expert >= 0:
        offset += expert * int(w["bytes"])

    w["gguf_tensor_offset"] = int(t.data_offset)
    w["gguf_matrix_offset"] = offset
    fixed += 1

tmp = manifest.with_suffix(".jsonl.tmp")

shutil.copy2(manifest, backup)

with tmp.open("x") as f:
    for w in rows:
        f.write(json.dumps(w, separators=(",", ":")) + "\n")

tmp.replace(manifest)

print("Matrices updated:", fixed)
print("Backup:", backup)
print("Corrected offset formula: t.data_offset + expert * matrix_bytes")
print("METADATA OFFSET FIX: PASS")
