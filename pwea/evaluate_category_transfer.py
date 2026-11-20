#!/usr/bin/env python3
"""Leave-one-task-category-out evaluation for PWEA expert rankings."""

import copy
import math
from collections import defaultdict

import build_profile as builder
import evaluate_profile as ranking


MIN_PROMPTS = 20
MIN_WORKLOADS = 3

CATEGORY_FIELD = "task_category"
NON_CATEGORY_FIELDS = (
    "language",
    "framework",
    "workload_phase",
)


def _validate_sessions(sessions, categories):
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("loaded_sessions must be a non-empty list")

    workload_ids = set()
    observed_categories = set()
    reference_labels = None

    for session in sessions:
        for field in ("records", "workload_id", "task_category"):
            if field not in session:
                raise ValueError(f"session is missing required field: {field}")

        workload_id = session["workload_id"]
        category = session["task_category"]
        records = session["records"]

        if not isinstance(workload_id, str) or not workload_id.strip():
            raise ValueError("workload_id must be a non-empty string")

        if workload_id in workload_ids:
            raise ValueError(f"duplicate workload_id: {workload_id}")
        workload_ids.add(workload_id)

        if category not in categories:
            raise ValueError(f"unexpected task category: {category}")

        if not records:
            raise ValueError(f"session has no records: {workload_id}")

        observed_categories.add(category)

        for record in records:
            labels = record.get("labels")
            if not isinstance(labels, dict):
                raise ValueError(f"record has invalid labels: {workload_id}")

            if labels.get(CATEGORY_FIELD) != category:
                raise ValueError(
                    f"session category does not match record labels: "
                    f"{workload_id}"
                )

            current = tuple(labels.get(field) for field in NON_CATEGORY_FIELDS)

            if any(value is None for value in current):
                raise ValueError(
                    f"record has missing non-category workload labels: "
                    f"{workload_id}"
                )

            if reference_labels is None:
                reference_labels = current
            elif current != reference_labels:
                raise ValueError(
                    "inconsistent non-category workload labels"
                )

            if record.get("workload_id") != workload_id:
                raise ValueError(
                    f"record workload_id does not match session: "
                    f"{workload_id}"
                )

    missing = set(categories) - observed_categories
    if missing:
        raise ValueError(
            "missing task categories: " + ", ".join(sorted(missing))
        )


def _support_summary(profiles):
    if len(profiles) != 1:
        raise ValueError("expected exactly one profile for pooled records")
    return profiles[0]["support"]


def _scores_by_expert_id(experts, field):
    scores = {item["expert_id"]: item[field] for item in experts}
    expected_ids = set(range(len(experts)))

    if set(scores) != expected_ids:
        raise ValueError(
            "profile expert IDs must uniquely cover 0..n_expert-1"
        )

    return [scores[expert_id] for expert_id in range(len(experts))]


def _evaluate_profile_pair(
    train_profile,
    test_profile,
    n_layer,
    k,
    relevant_k,
):
    pwea_rows = []
    baseline_rows = []

    for layer_id in range(n_layer):
        train_layer = train_profile["layers"][layer_id]
        test_layer = test_profile["layers"][layer_id]

        # Profiles sort experts by activation share, so list positions
        # must never be treated as expert identities.
        pwea_scores = _scores_by_expert_id(
            train_layer["experts"], "mean_prompt_activation_share"
        )
        baseline_scores = _scores_by_expert_id(
            train_layer["experts"], "aggregate_activation_share"
        )
        actual_scores = _scores_by_expert_id(
            test_layer["experts"], "mean_prompt_activation_share"
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

    def summarize(rows):
        return {
            metric: ranking.mean_metric(rows, metric)
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

    return {
        "pwea": summarize(pwea_rows),
        "frequency_baseline": summarize(baseline_rows),
        "pwea_layer_metrics": pwea_rows,
        "baseline_layer_metrics": baseline_rows,
    }


def evaluate_category_fold(
    heldout_category,
    loaded_sessions,
    n_layer,
    n_expert,
    k,
    relevant_k,
    categories=None,
    min_prompts=MIN_PROMPTS,
    min_workloads=MIN_WORKLOADS,
):
    """Train on all other categories and test each held-out session."""
    if categories is None:
        categories = tuple(sorted({
            session["task_category"] for session in loaded_sessions
        }))

    categories = tuple(categories)

    if len(categories) < 2 or len(set(categories)) != len(categories):
        raise ValueError("categories must contain at least two unique values")

    if heldout_category not in categories:
        raise ValueError(f"unknown held-out category: {heldout_category}")

    if n_layer <= 0 or n_expert <= 1:
        raise ValueError("invalid model dimensions")

    if k <= 0 or k > n_expert:
        raise ValueError("k must be between 1 and n_expert")

    if relevant_k <= 0 or relevant_k > n_expert:
        raise ValueError("relevant_k must be between 1 and n_expert")

    if min_prompts <= 0 or min_workloads <= 0:
        raise ValueError("support thresholds must be positive")

    _validate_sessions(loaded_sessions, categories)

    training_sessions = [
        session for session in loaded_sessions
        if session["task_category"] != heldout_category
    ]
    heldout_sessions = [
        session for session in loaded_sessions
        if session["task_category"] == heldout_category
    ]

    if not training_sessions:
        raise ValueError("no training sessions remain")

    if not heldout_sessions:
        raise ValueError("held-out category has no sessions")

    training_categories = sorted({
        session["task_category"] for session in training_sessions
    })

    if len(training_categories) != len(categories) - 1:
        raise ValueError("training pool does not contain every other category")

    # Pool categories by normalizing only the task_category label.
    # Deep copies ensure original loaded records remain unchanged.
    training_records = []

    for session in training_sessions:
        for record in session["records"]:
            copied = copy.deepcopy(record)
            copied["labels"][CATEGORY_FIELD] = "__pooled_training__"
            training_records.append(copied)

    train_profiles = builder.build_profile(
        training_records,
        n_layer,
        n_expert,
        min_prompts,
        min_workloads,
    )
    train_profile = train_profiles[0] if len(train_profiles) == 1 else None

    if train_profile is None:
        raise ValueError("training records did not produce one pooled profile")

    if not train_profile["support"]["eligible_for_prediction"]:
        raise ValueError(
            "pooled training profile has insufficient prompt/session support"
        )

    session_results = []

    for session in heldout_sessions:
        test_profiles = builder.build_profile(
            session["records"],
            n_layer,
            n_expert,
            1,
            1,
        )

        if len(test_profiles) != 1:
            raise ValueError(
                f"held-out session must produce one profile: "
                f"{session['workload_id']}"
            )

        test_profile = test_profiles[0]

        metrics = _evaluate_profile_pair(
            train_profile,
            test_profile,
            n_layer,
            k,
            relevant_k,
        )

        session_results.append({
            "workload_id": session["workload_id"],
            "prompt_count": test_profile["support"]["prompt_count"],
            "activation_count": test_profile["support"]["activation_count"],
            "support": test_profile["support"],
            **metrics,
        })

    def average_session_metrics(method):
        fields = (
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k",
            "tie_aware_precision_at_k",
            "tie_aware_recall_at_k",
            "tie_aware_ndcg_at_k",
            "spearman",
        )
        return {
            field: sum(
                result[method][field] for result in session_results
            ) / len(session_results)
            for field in fields
        }

    return {
        "heldout_category": heldout_category,
        "training_categories": training_categories,
        "training_sessions": sorted(
            session["workload_id"] for session in training_sessions
        ),
        "heldout_sessions": sorted(
            session["workload_id"] for session in heldout_sessions
        ),
        "training_support": train_profile["support"],
        "heldout_support": {
            "prompt_count": sum(
                item["prompt_count"] for item in session_results
            ),
            "source_workload_count": len(session_results),
            "activation_count": sum(
                item["activation_count"] for item in session_results
            ),
            "eligible_for_prediction": (
                sum(item["prompt_count"] for item in session_results)
                >= min_prompts
                and len(session_results) >= min_workloads
            ),
        },
        "layer_count": n_layer,
        "session_results": session_results,
        "pwea": average_session_metrics("pwea"),
        "frequency_baseline": average_session_metrics(
            "frequency_baseline"
        ),
    }


def evaluate_category_transfer(
    loaded_sessions,
    categories,
    n_layer,
    n_expert,
    k=5,
    relevant_k=None,
    min_prompts=MIN_PROMPTS,
    min_workloads=MIN_WORKLOADS,
):
    """Run one complete held-out-category fold per category."""
    if relevant_k is None:
        relevant_k = max(1, math.ceil(n_expert * 0.10))

    _validate_sessions(loaded_sessions, tuple(categories))

    folds = [
        evaluate_category_fold(
            category,
            loaded_sessions,
            n_layer,
            n_expert,
            k,
            relevant_k,
            categories=categories,
            min_prompts=min_prompts,
            min_workloads=min_workloads,
        )
        for category in categories
    ]

    def macro_average(method):
        fields = (
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k",
            "tie_aware_precision_at_k",
            "tie_aware_recall_at_k",
            "tie_aware_ndcg_at_k",
            "spearman",
        )
        return {
            field: sum(
                fold[method][field] for fold in folds
            ) / len(folds)
            for field in fields
        }

    return {
        "evaluation": "leave-one-task-category-out",
        "categories": list(categories),
        "category_count": len(categories),
        "fold_count": len(folds),
        "layer_count": n_layer,
        "expert_count": n_expert,
        "k": k,
        "relevant_k": relevant_k,
        "minimum_training_prompts": min_prompts,
        "minimum_training_workloads": min_workloads,
        "folds": folds,
        "overall": {
            "aggregation": "macro-average across held-out categories",
            "pwea": macro_average("pwea"),
            "frequency_baseline": macro_average(
                "frequency_baseline"
            ),
        },
    }
