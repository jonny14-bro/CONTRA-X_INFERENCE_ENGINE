import csv
import json
import math
from collections import Counter, defaultdict

PREDICTION = "/tmp/stageA_prediction.csv"
ETF = "/tmp/stageA_etf.jsonl"

LAYERS = 48
EXPERTS = 128
K = 8

# Average ranks, including ties.
def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1

        average = ((i + 1) + j) / 2.0

        for p in range(i, j):
            ranks[order[p]] = average

        i = j

    return ranks


def spearman(a, b):
    ra = average_ranks(a)
    rb = average_ranks(b)

    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)

    numerator = sum(
        (x - ma) * (y - mb) for x, y in zip(ra, rb)
    )

    da = sum((x - ma) ** 2 for x in ra)
    db = sum((y - mb) ** 2 for y in rb)

    if da == 0 or db == 0:
        return None

    return numerator / math.sqrt(da * db)


# Load native PWEA predictions.
pred = {}

with open(PREDICTION, newline="") as f:
    for row in csv.DictReader(f):
        key = (int(row["layer"]), int(row["expert_id"]))
        pred[key] = float(row["predicted_probability"])

assert len(pred) == LAYERS * EXPERTS
assert all(math.isfinite(x) and x >= 0 for x in pred.values())

# Load the exact Stage A ETF record.
with open(ETF) as f:
    records = [json.loads(line) for line in f if line.strip()]

assert len(records) == 1, f"Expected one ETF record, found {len(records)}"

record = records[0]

window = record.get("window", {})

assert window.get("completed_prompts") == 1, \
    "ETF prompt not completed"

assert window.get("dropped_events") == 0, \
    "ETF contains dropped events"

activation_counts = Counter()
activation_weights = defaultdict(float)

for event in record["activations"]:
    layer = int(event["layer"])
    expert = int(event["expert_id"])
    weight = float(event["routing_weight"])

    assert 0 <= layer < LAYERS
    assert 0 <= expert < EXPERTS

    activation_counts[(layer, expert)] += 1
    activation_weights[(layer, expert)] += weight

assert len(activation_counts) > 0

# Per-layer ranking comparison.
layer_correlations = []
topk_overlaps = []
precision_values = []
recall_values = []
active_coverage_values = []

for layer in range(LAYERS):
    predicted = [
        pred[(layer, expert)] for expert in range(EXPERTS)
    ]

    observed = [
        activation_counts[(layer, expert)]
        for expert in range(EXPERTS)
    ]

    rho = spearman(predicted, observed)

    if rho is not None:
        layer_correlations.append(rho)

    predicted_top = sorted(
        range(EXPERTS),
        key=lambda e: (-pred[(layer, e)], e)
    )[:K]

    observed_top = sorted(
        range(EXPERTS),
        key=lambda e: (-activation_counts[(layer, e)], e)
    )[:K]

    predicted_set = set(predicted_top)
    observed_set = set(observed_top)

    overlap = len(predicted_set & observed_set)

    topk_overlaps.append(overlap)
    precision_values.append(overlap / K)
    recall_values.append(overlap / K)

    active = {
        e for e in range(EXPERTS)
        if activation_counts[(layer, e)] > 0
    }

    active_coverage_values.append(
        len(predicted_set & active) / len(active)
        if active else None
    )

# Aggregate metrics.
valid_coverage = [
    x for x in active_coverage_values if x is not None
]

print("========== DATA VALIDATION ==========")
print("ETF records:", len(records))
print("Prompt ID:", record.get("prompt_id"))
print("Completed prompts:", record.get("window", {}).get("completed_prompts"))
print("Dropped events:", record.get("window", {}).get("dropped_events"))
print("Prediction rows:", len(pred))
print("ETF activation events:", sum(activation_counts.values()))
print("Unique activated layer/expert pairs:", len(activation_counts))
print("Active expert coverage: {:.2f}%".format(
    100 * len(activation_counts) / (LAYERS * EXPERTS)
))

print("\n========== PREDICTION VS ROUTING ==========")
print("Mean per-layer Spearman: {:.6f}".format(
    sum(layer_correlations) / len(layer_correlations)
))
print("Layers with valid correlation:", len(layer_correlations), "/", LAYERS)
print("Mean Top-8 overlap:", sum(topk_overlaps) / LAYERS, "/", K)
print("Mean Precision@8: {:.4f}".format(
    sum(precision_values) / LAYERS
))
print("Mean Recall@8: {:.4f}".format(
    sum(recall_values) / LAYERS
))
print("Mean active-expert coverage@8: {:.4f}".format(
    sum(valid_coverage) / len(valid_coverage)
    if valid_coverage else 0.0
))

print("\n========== MIGRATED EXPERTS ==========")

for layer, expert in [(0, 102), (1, 18)]:
    scores = [
        pred[(layer, e)] for e in range(EXPERTS)
    ]

    rank = 1 + sum(
        score > pred[(layer, expert)] for score in scores
    )

    print(
        f"L{layer} E{expert}: "
        f"prediction={pred[(layer, expert)]:.12g}, "
        f"rank={rank}/{EXPERTS}, "
        f"ETF activations={activation_counts[(layer, expert)]}, "
        f"routing_weight_sum={activation_weights[(layer, expert)]:.6f}"
    )

print("\n========== TOP PREDICTED EXPERTS ==========")

for layer in range(LAYERS):
    top = sorted(
        range(EXPERTS),
        key=lambda e: (-pred[(layer, e)], e)
    )[:3]

    formatted = [
        f"E{e}:{pred[(layer, e)]:.5f}"
        for e in top
    ]

    print(f"L{layer:02d}: " + ", ".join(formatted))

print("\n========== TOP OBSERVED EXPERTS ==========")

for layer in range(LAYERS):
    top = sorted(
        range(EXPERTS),
        key=lambda e: (-activation_counts[(layer, e)], e)
    )[:3]

    formatted = [
        f"E{e}:{activation_counts[(layer, e)]}"
        for e in top
    ]

    print(f"L{layer:02d}: " + ", ".join(formatted))

print("\nComparison completed.")
