#!/usr/bin/env python3
"""Prompt-level leave-one-category-out evaluation for PWEA."""

import copy
import math

import build_profile as builder
import evaluate_profile as ranking
import evaluate_category_transfer as category_validation


METRICS = (
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "tie_aware_precision_at_k",
    "tie_aware_recall_at_k",
    "tie_aware_ndcg_at_k",
    "spearman",
)


def _mean(rows, field):
    if not rows:
        raise ValueError("cannot average an empty result set")
    return sum(row[field] for row in rows) / len(rows)


def _summarize(rows):
    return {field: _mean(rows, field) for field in METRICS}


def _profile_pair_metrics(
    train_profile, test_record, n_layer, k, relevant_k
):
    """Compare a pooled training profile with exactly one test prompt."""
    test_profiles = builder.build_profile(
        [test_record], n_layer, len(train_profile["layers"][0]["experts"]),
        1, 1,
    )
    if len(test_profiles) != 1:
        raise ValueError("one held-out prompt must produce exactly one profile")

    test_profile = test_profiles[0]
    pwea_rows = []
    baseline_rows = []

    for layer_id in range(n_layer):
        train_experts = train_profile["layers"][layer_id]["experts"]
        test_experts = test_profile["layers"][layer_id]["experts"]

        # Profiles sort experts by activation share, so never use their
        # list positions as expert identities. Align all score vectors by ID.
        def scores_by_expert_id(experts, field):
            scores = {item["expert_id"]: item[field] for item in experts}
            expected_ids = set(range(len(experts)))
            if set(scores) != expected_ids:
                raise ValueError(
                    "profile expert IDs must uniquely cover 0..n_expert-1"
                )
            return [scores[expert_id] for expert_id in range(len(experts))]

        pwea_scores = scores_by_expert_id(
            train_experts, "mean_prompt_activation_share"
        )
        baseline_scores = scores_by_expert_id(
            train_experts, "aggregate_activation_share"
        )
        actual_scores = scores_by_expert_id(
            test_experts, "mean_prompt_activation_share"
        )

        pwea_rows.append(
            ranking.ranking_metrics(
                pwea_scores, actual_scores, k, relevant_k
            )
        )
        baseline_rows.append(
            ranking.ranking_metrics(
                baseline_scores, actual_scores, k, relevant_k
            )
        )

    return {
        "pwea": _summarize(pwea_rows),
        "frequency_baseline": _summarize(baseline_rows),
        "pwea_layer_metrics": pwea_rows,
        "baseline_layer_metrics": baseline_rows,
        "test_activation_count": test_record["activation_count"],
    }


def evaluate_category_fold(
    heldout_category,
    loaded_sessions,
    categories,
    n_layer,
    n_expert,
    k=5,
    relevant_k=None,
    min_prompts=20,
    min_workloads=3,
):
    """Train on four categories and test every prompt in the fifth."""
    categories = tuple(categories)
    if relevant_k is None:
        relevant_k = max(1, math.ceil(n_expert * 0.10))

    if len(categories) < 2 or len(set(categories)) != len(categories):
        raise ValueError("categories must contain at least two unique values")
    if heldout_category not in categories:
        raise ValueError(f"unknown held-out category: {heldout_category}")
    if n_layer <= 0 or n_expert <= 1:
        raise ValueError("invalid model dimensions")
    if not 0 < k <= n_expert:
        raise ValueError("k must be between 1 and n_expert")
    if not 0 < relevant_k <= n_expert:
        raise ValueError("relevant_k must be between 1 and n_expert")
    if min_prompts <= 0 or min_workloads <= 0:
        raise ValueError("support thresholds must be positive")

    category_validation._validate_sessions(loaded_sessions, categories)

    training_sessions = [
        session for session in loaded_sessions
        if session["task_category"] != heldout_category
    ]
    heldout_sessions = [
        session for session in loaded_sessions
        if session["task_category"] == heldout_category
    ]

    if len(training_sessions) == 0 or len(heldout_sessions) == 0:
        raise ValueError("training and held-out sessions must both be non-empty")

    training_categories = {
        session["task_category"] for session in training_sessions
    }
    if training_categories != set(categories) - {heldout_category}:
        raise ValueError("training pool does not contain every other category")

    training_records = []
    for session in training_sessions:
        if len(session["records"]) != 7:
            raise ValueError(
                f"session must contain exactly 7 prompts: {session['workload_id']}"
            )
        for record in session["records"]:
            copied = copy.deepcopy(record)
            copied["labels"]["task_category"] = "__pooled_training__"
            training_records.append(copied)

    train_profiles = builder.build_profile(
        training_records, n_layer, n_expert, min_prompts, min_workloads
    )
    if len(train_profiles) != 1:
        raise ValueError("training data did not produce one pooled profile")

    train_profile = train_profiles[0]
    if not train_profile["support"]["eligible_for_prediction"]:
        raise ValueError("pooled training profile has insufficient support")

    prompt_results = []
    seen_test_prompts = set()

    for session in heldout_sessions:
        if len(session["records"]) != 7:
            raise ValueError(
                f"held-out session must contain exactly 7 prompts: "
                f"{session['workload_id']}"
            )

        session_prompt_ids = set()
        for record in session["records"]:
            prompt_id = record["prompt_id"]
            if prompt_id in session_prompt_ids:
                raise ValueError(
                    f"duplicate prompt_id {prompt_id} in {session['workload_id']}"
                )
            session_prompt_ids.add(prompt_id)

            identity = (session["workload_id"], prompt_id)
            if identity in seen_test_prompts:
                raise ValueError(f"duplicate held-out prompt: {identity}")
            seen_test_prompts.add(identity)

            metrics = _profile_pair_metrics(
                train_profile, record, n_layer, k, relevant_k
            )
            prompt_results.append({
                "workload_id": session["workload_id"],
                "task_category": heldout_category,
                "prompt_id": prompt_id,
                **metrics,
            })

    if not prompt_results:
        raise ValueError("held-out category contains no prompts")

    return {
        "heldout_category": heldout_category,
        "training_categories": sorted(training_categories),
        "training_sessions": sorted(
            session["workload_id"] for session in training_sessions
        ),
        "heldout_sessions": sorted(
            session["workload_id"] for session in heldout_sessions
        ),
        "training_support": train_profile["support"],
        "heldout_prompt_count": len(prompt_results),
        "layer_count": n_layer,
        "expert_count": n_expert,
        "k": k,
        "relevant_k": relevant_k,
        "aggregation": "equal-weight mean across individual prompts",
        "prompt_results": prompt_results,
        "pwea": _summarize([row["pwea"] for row in prompt_results]),
        "frequency_baseline": _summarize(
            [row["frequency_baseline"] for row in prompt_results]
        ),
    }


def evaluate_prompt_transfer(
    loaded_sessions,
    categories,
    n_layer,
    n_expert,
    k=5,
    relevant_k=None,
    min_prompts=20,
    min_workloads=3,
):
    """Run five prompt-level held-out-category folds."""
    categories = tuple(categories)
    if relevant_k is None:
        relevant_k = max(1, math.ceil(n_expert * 0.10))

    category_validation._validate_sessions(loaded_sessions, categories)

    folds = [
        evaluate_category_fold(
            category,
            loaded_sessions,
            categories,
            n_layer,
            n_expert,
            k,
            relevant_k,
            min_prompts,
            min_workloads,
        )
        for category in categories
    ]

    def macro_average(method):
        return {
            field: _mean([fold[method] for fold in folds], field)
            for field in METRICS
        }

    return {
        "evaluation": "leave-one-task-category-out-prompt-level",
        "test_unit": "one individual prompt record",
        "categories": list(categories),
        "category_count": len(categories),
        "fold_count": len(folds),
        "total_test_prompts": sum(
            fold["heldout_prompt_count"] for fold in folds
        ),
        "layer_count": n_layer,
        "expert_count": n_expert,
        "k": k,
        "relevant_k": relevant_k,
        "aggregation": "macro-average across categories; equal-weight prompts within each category",
        "folds": folds,
        "overall": {
            "pwea": macro_average("pwea"),
            "frequency_baseline": macro_average("frequency_baseline"),
        },
        "limitations": [
            "Prompts within a session may be correlated; prompt results are not necessarily statistically independent.",
            "Algorithmic problem-solving contains generation-censored outputs from the original collection.",
            "This experiment evaluates expert-demand prediction, not memory placement or inference latency.",
        ],
    }
