import csv
import json
import math
from collections import Counter, defaultdict

PREDICTION = "/tmp/stageA_prediction.csv"
ETF = "/tmp/stageA_etf.jsonl"

LAYERS = 48
EXPERTS = 128
TOP_K = (8, 16, 32)
MIGRATED = [(0, 102), (1, 18)]


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1

        rank = ((i + 1) + j) / 2.0

        for p in range(i, j):
            ranks[order[p]] = rank

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


# Fractional membership handles ties at the Top-K boundary.
# Example: if 4 positions remain and 8 experts tie at the
# cutoff, each receives membership weight 4/8 = 0.5.
def fractional_topk(values, k):
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    result = [0.0] * len(values)

    cutoff = values[order[k - 1]]
    above = [i for i in range(len(values)) if values[i] > cutoff]
    tied = [i for i in range(len(values)) if values[i] == cutoff]

    for i in above:
        result[i] = 1.0

    remaining = k - len(above)
    fraction = remaining / len(tied)

    for i in tied:
        result[i] = fraction

    return result


def tie_aware_overlap(predicted, observed, k):
    p = fractional_topk(predicted, k)
    o = fractional_topk(observed, k)

    # Expected intersection under fractional boundary membership.
    return sum(x * y for x, y in zip(p, o))


def macro_mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else float("nan")


# Load predictions.
pred = {}

with open(PREDICTION, newline="") as f:
    for row in csv.DictReader(f):
        key = (int(row["layer"]), int(row["expert_id"]))
        pred[key] = float(row["predicted_probability"])

assert len(pred) == LAYERS * EXPERTS

# Load ETF record.
with open(ETF) as f:
    records = [json.loads(line) for line in f if line.strip()]

assert len(records) == 1
record = records[0]
window = record.get("window", {})

assert window.get("completed_prompts") == 1
assert window.get("dropped_events") == 0

counts = Counter()
weights = defaultdict(float)

for event in record["activations"]:
    layer = int(event["layer"])
    expert = int(event["expert_id"])
    weight = float(event["routing_weight"])

    assert 0 <= layer < LAYERS
    assert 0 <= expert < EXPERTS
    assert math.isfinite(weight)

    counts[(layer, expert)] += 1
    weights[(layer, expert)] += weight

assert sum(counts.values()) == len(record["activations"])

# Store per-layer data and calculate metrics.
count_rhos = []
weight_rhos = []
layer_metrics = {}

for layer in range(LAYERS):
    p = [pred[(layer, e)] for e in range(EXPERTS)]
    c = [counts[(layer, e)] for e in range(EXPERTS)]
    w = [weights[(layer, e)] for e in range(EXPERTS)]

    count_rhos.append(spearman(p, c))
    weight_rhos.append(spearman(p, w))

    metrics = {}

    for k in TOP_K:
        overlap = tie_aware_overlap(p, c, k)
        metrics[k] = overlap

    predicted_top8 = sorted(
        range(EXPERTS),
        key=lambda e: (-pred[(layer, e)], e)
    )[:8]

    active = {
        e for e in range(EXPERTS)
        if counts[(layer, e)] > 0
    }

    active_coverage = (
        len(set(predicted_top8) & active) / len(active)
        if active else None
    )

    layer_metrics[layer] = {
        "count_rho": count_rhos[-1],
        "weight_rho": weight_rhos[-1],
        "topk": metrics,
        "active_count": len(active),
        "active_coverage": active_coverage,
    }


print("========== INPUT VALIDATION ==========")
print("Prompt ID:", record.get("prompt_id"))
print("Completed prompts:", window.get("completed_prompts"))
print("Dropped events:", window.get("dropped_events"))
print("Prediction rows:", len(pred))
print("ETF activation events:", sum(counts.values()))
print("Unique active layer/expert pairs:", len(counts))
print("Active expert coverage: {:.2f}%".format(
    100 * len(counts) / (LAYERS * EXPERTS)
))

print("\n========== RANK CORRELATION ==========")
print("Mean per-layer Spearman vs activation count: {:.6f}".format(
    macro_mean(count_rhos)
))
print("Mean per-layer Spearman vs routing weight: {:.6f}".format(
    macro_mean(weight_rhos)
))
print("Layers evaluated:", LAYERS)

print("\n========== TIE-AWARE TOP-K OVERLAP ==========")
print("Random baseline expected overlap = K*K/128")

for k in TOP_K:
    values = [
        layer_metrics[l]["topk"][k]
        for l in range(LAYERS)
    ]

    observed = macro_mean(values)
    random_expected = (k * k) / EXPERTS

    print(
        f"Top-{k}: overlap={observed:.4f}/{k} "
        f"({100 * observed / k:.2f}%), "
        f"random_expected={random_expected:.4f}"
    )

print("\n========== ACTIVE EXPERT COVERAGE ==========")
print("Mean Top-8 coverage of active experts: {:.4f}".format(
    macro_mean([
        layer_metrics[l]["active_coverage"]
        for l in range(LAYERS)
    ])
))

print("\n========== MIGRATED EXPERT AUDIT ==========")

for layer, expert in MIGRATED:
    score = pred[(layer, expert)]
    layer_scores = [pred[(layer, e)] for e in range(EXPERTS)]

    rank = 1 + sum(x > score for x in layer_scores)

    print(
        f"L{layer} E{expert}: "
        f"PWEA rank={rank}/{EXPERTS}, "
        f"score={score:.12g}, "
        f"activations={counts[(layer, expert)]}, "
        f"routing_weight={weights[(layer, expert)]:.6f}"
    )

print("\n========== HIGH-DEMAND PREDICTION MISSES ==========")

misses = []

for layer in range(LAYERS):
    observed_order = sorted(
        range(EXPERTS),
        key=lambda e: (-counts[(layer, e)], e)
    )

    predicted_order = sorted(
        range(EXPERTS),
        key=lambda e: (-pred[(layer, e)], e)
    )

    predicted_rank = {
        expert: rank + 1
        for rank, expert in enumerate(predicted_order)
    }

    for expert in observed_order[:8]:
        misses.append((
            counts[(layer, expert)],
            predicted_rank[expert],
            layer,
            expert,
            pred[(layer, expert)],
            weights[(layer, expert)]
        ))

misses.sort(key=lambda x: (-x[0], x[1]))

for count, rank, layer, expert, score, weight in misses[:30]:
    print(
        f"L{layer:02d} E{expert:03d}: "
        f"activations={count:2d}, "
        f"PWEA_rank={rank:3d}, "
        f"prediction={score:.6f}, "
        f"routing_weight={weight:.6f}"
    )

print("\n========== PER-LAYER SUMMARY ==========")
print("Layer | Count rho | Weight rho | Active | Top8 overlap")

for layer in range(LAYERS):
    m = layer_metrics[layer]
    print(
        f"{layer:02d} | "
        f"{m['count_rho']:+.4f} | "
        f"{m['weight_rho']:+.4f} | "
        f"{m['active_count']:3d} | "
        f"{m['topk'][8]:.2f}"
    )

print("\nRanking audit completed.")
