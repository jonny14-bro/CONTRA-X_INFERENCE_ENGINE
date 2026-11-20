#!/usr/bin/env python3
"""Leave-one-workload-session-out evaluation for PWEA rankings."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import build_profile as builder


MIN_PROMPTS = 20
MIN_WORKLOADS = 3


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_ranks(values):
    """Return one-based average ranks, with equal values sharing ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)

    start = 0
    while start < len(order):
        end = start + 1
        value = values[order[start]]

        while end < len(order) and values[order[end]] == value:
            end += 1

        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank

        start = end

    return ranks


def spearman_correlation(predicted, actual):
    """Spearman correlation using average ranks for ties."""
    x = average_ranks(predicted)
    y = average_ranks(actual)

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = sum(
        (a - mean_x) * (b - mean_y)
        for a, b in zip(x, y)
    )
    sum_x = sum((a - mean_x) ** 2 for a in x)
    sum_y = sum((b - mean_y) ** 2 for b in y)

    denominator = math.sqrt(sum_x * sum_y)
    if denominator == 0:
        return 0.0

    return numerator / denominator


def _expected_top_k_probabilities(scores, limit):
    """Expected inclusion probabilities under uniform random tie-breaking."""
    if not 0 < limit <= len(scores):
        raise ValueError("invalid top-k limit")

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    probabilities = [0.0] * len(scores)
    remaining = limit
    position = 0

    while position < len(order) and remaining > 0:
        end = position + 1
        value = scores[order[position]]

        while end < len(order) and scores[order[end]] == value:
            end += 1

        group = order[position:end]
        probability = min(1.0, remaining / len(group))

        for expert in group:
            probabilities[expert] = probability

        remaining -= probability * len(group)
        position = end

    return probabilities


def tie_aware_ranking_metrics(predicted_scores, actual_scores, k, relevant_k):
    """Expected top-k metrics under uniform random tie-breaking."""
    if len(predicted_scores) != len(actual_scores) or not predicted_scores:
        raise ValueError("score vectors must have equal nonzero length")
    if not 0 < k <= len(predicted_scores):
        raise ValueError("invalid k")
    if not 0 < relevant_k <= len(actual_scores):
        raise ValueError("invalid relevant_k")

    predicted_probability = _expected_top_k_probabilities(
        predicted_scores, k
    )
    actual_relevance = _expected_top_k_probabilities(
        actual_scores, relevant_k
    )

    expected_hits = sum(
        predicted_probability[i] * actual_relevance[i]
        for i in range(len(predicted_scores))
    )

    order = sorted(
        range(len(predicted_scores)),
        key=lambda i: -predicted_scores[i],
    )

    dcg = 0.0
    position = 0

    while position < k:
        end = position + 1
        value = predicted_scores[order[position]]

        while end < len(order) and predicted_scores[order[end]] == value:
            end += 1

        group = order[position:end]
        included_end = min(end, k)

        average_gain = sum(
            actual_scores[expert] for expert in group
        ) / len(group)

        discount_sum = sum(
            1.0 / math.log2(rank + 2)
            for rank in range(position, included_end)
        )

        dcg += average_gain * discount_sum
        position = end

    ideal = sorted(actual_scores, reverse=True)[:k]
    idcg = sum(
        gain / math.log2(position + 2)
        for position, gain in enumerate(ideal)
    )

    return {
        "tie_aware_precision_at_k": expected_hits / k,
        "tie_aware_recall_at_k": expected_hits / relevant_k,
        "tie_aware_ndcg_at_k": dcg / idcg if idcg > 0 else 0.0,
    }


def ranking_metrics(predicted_scores, actual_scores, k, relevant_k):
    """Evaluate one layer's predicted ranking against held-out relevance."""
    predicted_order = sorted(
        range(len(predicted_scores)),
        key=lambda expert: (-predicted_scores[expert], expert),
    )
    actual_order = sorted(
        range(len(actual_scores)),
        key=lambda expert: (-actual_scores[expert], expert),
    )

    predicted_top = predicted_order[:k]
    relevant = set(actual_order[:relevant_k])
    hits = sum(expert in relevant for expert in predicted_top)

    precision = hits / k
    recall = hits / relevant_k

    dcg = sum(
        actual_scores[expert] / math.log2(position + 2)
        for position, expert in enumerate(predicted_top)
    )

    ideal = actual_order[:k]
    idcg = sum(
        actual_scores[expert] / math.log2(position + 2)
        for position, expert in enumerate(ideal)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    correlation = spearman_correlation(
        predicted_scores,
        actual_scores,
    )

    tie_aware = tie_aware_ranking_metrics(
        predicted_scores, actual_scores, k, relevant_k
    )

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "ndcg_at_k": ndcg,
        "spearman": correlation,
        "hits": hits,
        **tie_aware,
    }


def mean_metric(rows, field):
    return sum(row[field] for row in rows) / len(rows)


def evaluate_fold(
    heldout_index,
    paths,
    loaded_sessions,
    n_layer,
    n_expert,
    k,
    relevant_k,
):
    training_paths = [
        path for index, path in enumerate(paths)
        if index != heldout_index
    ]
    training_records = [
        record
        for index, session in enumerate(loaded_sessions)
        if index != heldout_index
        for record in session["records"]
    ]

    heldout = loaded_sessions[heldout_index]
    heldout_records = heldout["records"]

    train_profiles = builder.build_profile(
        training_records,
        n_layer,
        n_expert,
        MIN_PROMPTS,
        MIN_WORKLOADS,
    )
    test_profiles = builder.build_profile(
        heldout_records,
        n_layer,
        n_expert,
        MIN_PROMPTS,
        MIN_WORKLOADS,
    )

    if len(train_profiles) != 1 or len(test_profiles) != 1:
        raise builder.ValidationError(
            "each evaluation fold must contain exactly one workload group"
        )

    train_profile = train_profiles[0]
    test_profile = test_profiles[0]

    if train_profile["workload"] != test_profile["workload"]:
        raise builder.ValidationError(
            "training and held-out workload labels do not match"
        )

    pwea_rows = []
    baseline_rows = []

    for layer_id in range(n_layer):
        train_layer = train_profile["layers"][layer_id]
        test_layer = test_profile["layers"][layer_id]

        pwea_scores = [
            item["mean_prompt_activation_share"]
            for item in train_layer["experts"]
        ]
        baseline_scores = [
            item["aggregate_activation_share"]
            for item in train_layer["experts"]
        ]
        actual_scores = [
            item["mean_prompt_activation_share"]
            for item in test_layer["experts"]
        ]

        pwea_rows.append(
            ranking_metrics(pwea_scores, actual_scores, k, relevant_k)
        )
        baseline_rows.append(
            ranking_metrics(baseline_scores, actual_scores, k, relevant_k)
        )

    def summarize(rows):
        return {
            "precision_at_k": mean_metric(rows, "precision_at_k"),
            "recall_at_k": mean_metric(rows, "recall_at_k"),
            "ndcg_at_k": mean_metric(rows, "ndcg_at_k"),
            "tie_aware_precision_at_k": mean_metric(rows, "tie_aware_precision_at_k"),
            "tie_aware_recall_at_k": mean_metric(rows, "tie_aware_recall_at_k"),
            "tie_aware_ndcg_at_k": mean_metric(rows, "tie_aware_ndcg_at_k"),
            "spearman": mean_metric(rows, "spearman"),
        }

    return {
        "heldout_session": heldout["workload_id"],
        "training_sessions": sorted(
            session["workload_id"]
            for index, session in enumerate(loaded_sessions)
            if index != heldout_index
        ),
        "training_support": train_profile["support"],
        "heldout_support": test_profile["support"],
        "layer_count": n_layer,
        "pwea": summarize(pwea_rows),
        "frequency_baseline": summarize(baseline_rows),
        "pwea_layer_metrics": pwea_rows,
        "baseline_layer_metrics": baseline_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PWEA rankings with leave-one-session-out testing."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="One ETF JSONL session file; supply at least three independent sessions.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--n-layer", required=True, type=int)
    parser.add_argument("--n-expert", required=True, type=int)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if len(args.input) < 3:
        parser.error("at least three independent session files are required")
    if args.n_layer <= 0 or args.n_expert <= 1:
        parser.error("model dimensions must be positive; n-expert must exceed 1")
    if args.k <= 0 or args.k > args.n_expert:
        parser.error("--k must be between 1 and n-expert")

    relevant_k = max(1, math.ceil(args.n_expert * 0.10))
    if args.k > relevant_k:
        parser.error("--k cannot exceed the held-out relevant set size")

    paths = [path.resolve() for path in args.input]
    output = args.output.resolve()

    if len(set(paths)) != len(paths):
        parser.error("duplicate input paths are not allowed")
    if output in paths:
        parser.error("output path must differ from all input paths")
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")

    try:
        loaded_sessions = []
        all_workload_ids = set()

        for path in paths:
            if not path.is_file():
                raise builder.ValidationError(
                    f"input file does not exist: {path}"
                )

            records, workload_ids, activation_count = builder.load_records(
                path,
                args.n_layer,
                args.n_expert,
            )

            if len(workload_ids) != 1:
                raise builder.ValidationError(
                    f"{path.name}: expected exactly one workload ID"
                )

            workload_id = next(iter(workload_ids))
            if workload_id in all_workload_ids:
                raise builder.ValidationError(
                    f"workload ID reused across sessions: {workload_id}"
                )
            all_workload_ids.add(workload_id)

            loaded_sessions.append({
                "path": path,
                "records": records,
                "workload_id": workload_id,
                "activation_count": activation_count,
            })

        reference_labels = loaded_sessions[0]["records"][0]["labels"]
        for session in loaded_sessions[1:]:
            if any(
                record["labels"] != reference_labels
                for record in session["records"]
            ):
                raise builder.ValidationError(
                    "all sessions must belong to the same workload category"
                )

        folds = [
            evaluate_fold(
                index,
                paths,
                loaded_sessions,
                args.n_layer,
                args.n_expert,
                args.k,
                relevant_k,
            )
            for index in range(len(paths))
        ]

        def overall_summary(method):
            return {
                metric: sum(fold[method][metric] for fold in folds) / len(folds)
                for metric in (
                    "precision_at_k",
                    "recall_at_k",
                    "ndcg_at_k",
                    "tie_aware_precision_at_k",
                    "tie_aware_recall_at_k",
                    "tie_aware_ndcg_at_k",
                    "spearman",
                )
            }

        artifact = {
            "artifact_type": "pwea_leave_one_session_out_evaluation",
            "schema_version": 1,
            "method": {
                "evaluation": "leave_one_workload_session_out",
                "pwea_score": "mean_prompt_activation_share",
                "baseline_score": "aggregate_activation_share",
                "heldout_relevance": "mean_prompt_activation_share",
                "relevant_expert_count": relevant_k,
                "k": args.k,
                "metrics_averaged": "macro_average_over_layers_then_folds",
                "minimum_production_prompts": MIN_PROMPTS,
                "minimum_production_workloads": MIN_WORKLOADS,
            },
            "model": {
                "model_id": args.model_id,
                "n_layer": args.n_layer,
                "n_expert": args.n_expert,
            },
            "workload": reference_labels,
            "sources": [
                {
                    "file_name": session["path"].name,
                    "sha256": sha256_file(session["path"]),
                    "workload_id": session["workload_id"],
                    "prompt_records": len(session["records"]),
                    "activation_events": session["activation_count"],
                }
                for session in loaded_sessions
            ],
            "overall": {
                "pwea": overall_summary("pwea"),
                "frequency_baseline": overall_summary("frequency_baseline"),
            },
            "folds": folds,
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                artifact,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

        print(f"Evaluation written: {output}")
        print(f"Folds: {len(folds)}")
        print(f"Layers per fold: {args.n_layer}")
        print(f"Relevant experts per layer: {relevant_k}")
        print(f"K: {args.k}")

        for name in ("pwea", "frequency_baseline"):
            print(f"\n{name}:")
            for metric, value in artifact["overall"][name].items():
                print(f"  {metric}: {value:.6f}")

        print("\nFold training eligibility:")
        for fold in folds:
            support = fold["training_support"]
            print(
                f"  heldout={fold['heldout_session']} "
                f"prompts={support['prompt_count']} "
                f"workloads={support['source_workload_count']} "
                f"eligible={support['eligible_for_prediction']}"
            )

        print("\nEvaluation SHA-256:", sha256_file(output))

    except (builder.ValidationError, OSError, ValueError) as exc:
        print(f"PWEA evaluation error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
