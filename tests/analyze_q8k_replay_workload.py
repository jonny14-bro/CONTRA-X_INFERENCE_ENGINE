import json
from pathlib import Path
from collections import Counter, defaultdict

base = Path("benchmarks/real_activation_weight_replay_20261118")

weights = [
    json.loads(x)
    for x in (base / "weights.jsonl").read_text().splitlines()
]
records = [
    json.loads(x)
    for x in (base / "replay.jsonl").read_text().splitlines()
]

by_id = {w["weight_id"]: w for w in weights}

ops = Counter()
types = Counter()
batches = Counter()
operation_types = Counter()
operation_batches = defaultdict(Counter)
operation_macs = Counter()
operation_weight_bytes = Counter()
unique_matrices = defaultdict(set)

total_macs = 0
total_activation_floats = 0

for r in records:
    w = by_id[r["weight_id"]]
    op = r["operation"]
    typ = r["weight_type"]
    nb = r["nb"]
    k = r["k"]
    rows = r["weight_rows"]

    macs = nb * k * rows

    ops[op] += 1
    types[typ] += 1
    batches[nb] += 1
    operation_types[(op, typ)] += 1
    operation_batches[op][nb] += 1
    operation_macs[op] += macs
    operation_weight_bytes[op] += w["bytes"]
    unique_matrices[op].add(r["weight_id"])

    total_macs += macs
    total_activation_floats += r["elements"]

print("===== REPLAY WORKLOAD =====")
print("Records:", len(records))
print("Unique matrices:", len(weights))
print("Activation floats referenced:", total_activation_floats)
print("Total matrix MACs:", total_macs)
print("Total matrix GMACs:", round(total_macs / 1e9, 4))

print("\n===== OPERATIONS =====")
for op in sorted(ops):
    print(
        f"{op:20s}",
        "records=", ops[op],
        "unique_matrices=", len(unique_matrices[op]),
        "GMAC=", round(operation_macs[op] / 1e9, 4),
    )

print("\n===== OPERATION / TYPE =====")
for key, count in sorted(operation_types.items()):
    print(key, count)

print("\n===== BATCH SIZE DISTRIBUTION =====")
for nb, count in sorted(batches.items()):
    print(f"nb={nb:4d}: {count}")

print("\n===== PER-OPERATION BATCH DISTRIBUTION =====")
for op in sorted(operation_batches):
    print(op, dict(sorted(operation_batches[op].items())))

print("\n===== REPLAY RECORD SHAPE EXAMPLES =====")
for r in records[:12]:
    w = by_id[r["weight_id"]]
    print({
        "record": r["record"],
        "operation": r["operation"],
        "layer": r["layer"],
        "expert": r["expert"],
        "type": r["weight_type"],
        "k": r["k"],
        "nb": r["nb"],
        "rows": r["weight_rows"],
        "row_bytes": r["weight_row_bytes"],
    })
