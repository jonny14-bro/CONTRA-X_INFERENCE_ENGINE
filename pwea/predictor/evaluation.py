#!/usr/bin/env python3
"""Prompt-level expert prediction evaluation with validity-mask support."""

import sys
from pathlib import Path

import numpy as np

# evaluate_profile.py uses a legacy top-level import of build_profile.
PWEA_DIR = Path(__file__).resolve().parents[1]
if str(PWEA_DIR) not in sys.path:
    sys.path.insert(0, str(PWEA_DIR))

from pwea.evaluate_profile import ranking_metrics


DEFAULT_K = 5
DEFAULT_RELEVANT_K = 13

RANKING_FIELDS = (
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "spearman",
    "tie_aware_precision_at_k",
    "tie_aware_recall_at_k",
    "tie_aware_ndcg_at_k",
)

COMPOSITE_FIELDS = (
    "tie_aware_precision_at_k",
    "tie_aware_recall_at_k",
    "tie_aware_ndcg_at_k",
)


def evaluate_prompt_predictions(
    samples,
    indices,
    predictions,
    k=DEFAULT_K,
    relevant_k=DEFAULT_RELEVANT_K,
    n_layer=48,
    n_expert=128,
):
    """
    Evaluate predictions aligned positionally with `indices`.

    Returns one result per prompt. Layer metrics are averaged within each
    prompt. Regression metrics are calculated over all valid layer-expert
    elements in that prompt.
    """
    indices = list(indices)
    predictions = np.asarray(predictions, dtype=np.float64)

    if not indices:
        raise ValueError("evaluation indices must not be empty")

    if len(set(indices)) != len(indices):
        raise ValueError("evaluation indices must be unique")

    if predictions.shape == (len(indices), n_layer * n_expert):
        predictions = predictions.reshape(len(indices), n_layer, n_expert)

    if predictions.shape != (len(indices), n_layer, n_expert):
        raise ValueError("prediction dimensions do not match evaluation samples")

    if not np.all(np.isfinite(predictions)):
        raise ValueError("predictions must contain only finite values")

    if not 0 < k <= n_expert or not 0 < relevant_k <= n_expert:
        raise ValueError("invalid ranking cutoffs")

    results = []

    for position, sample_index in enumerate(indices):
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError("evaluation indices must be integers")
        if not 0 <= sample_index < len(samples):
            raise IndexError(f"evaluation index out of range: {sample_index}")

        sample = samples[sample_index]
        target = sample["target"]

        actual = np.asarray(target["activation_share"], dtype=np.float64)
        valid = np.asarray(target["valid_layer_mask"], dtype=bool)

        if actual.shape != (n_layer, n_expert):
            raise ValueError("target dimensions do not match model dimensions")
        if valid.shape != (n_layer,):
            raise ValueError("valid_layer_mask has incorrect dimensions")
        if not np.all(np.isfinite(actual)):
            raise ValueError("target contains non-finite values")

        layer_results = []
        absolute_errors = []
        squared_errors = []

        for layer in range(n_layer):
            if not valid[layer]:
                continue

            if np.any(actual[layer] < 0):
                raise ValueError("target shares must be nonnegative")
            if not np.isclose(actual[layer].sum(), 1.0, atol=1e-10):
                raise ValueError("valid target layer must sum to one")

            predicted_layer = predictions[position, layer]
            actual_layer = actual[layer]

            metrics = ranking_metrics(
                predicted_layer.tolist(),
                actual_layer.tolist(),
                k,
                relevant_k,
            )
            layer_results.append(metrics)

            errors = predicted_layer - actual_layer
            absolute_errors.extend(np.abs(errors).tolist())
            squared_errors.extend((errors ** 2).tolist())

        if not layer_results:
            raise ValueError(
                f"{sample.get('sample_id')}: no valid target layers"
            )

        layer_count = len(layer_results)
        averaged = {
            field: sum(row[field] for row in layer_results) / layer_count
            for field in RANKING_FIELDS
        }

        mae = sum(absolute_errors) / len(absolute_errors)
        rmse = float(np.sqrt(sum(squared_errors) / len(squared_errors)))

        composite = sum(
            averaged[field] for field in COMPOSITE_FIELDS
        ) / len(COMPOSITE_FIELDS)

        results.append({
            "sample_id": sample["sample_id"],
            "session_id": sample["session_id"],
            "category": sample["category"],
            "prompt_id": sample["prompt_id"],
            "generation_censored": bool(
                sample.get("generation_censored", False)
            ),
            "valid_layer_count": layer_count,
            **averaged,
            "mae": mae,
            "rmse": rmse,
            "composite_score": composite,
        })

    return results


def aggregate_prompt_results(prompt_results):
    """
    Aggregate equally weighted prompts within categories, then categories.

    Returns category-level means and an unweighted category macro-average.
    """
    if not prompt_results:
        raise ValueError("prompt_results must not be empty")

    fields = (*RANKING_FIELDS, "mae", "rmse", "composite_score")
    grouped = {}

    for row in prompt_results:
        category = row["category"]
        grouped.setdefault(category, []).append(row)

    category_results = {}

    for category in sorted(grouped):
        rows = grouped[category]
        category_results[category] = {
            "prompt_count": len(rows),
            **{
                field: sum(row[field] for row in rows) / len(rows)
                for field in fields
            },
        }

    macro = {
        field: sum(
            result[field] for result in category_results.values()
        ) / len(category_results)
        for field in fields
    }

    return {
        "category_results": category_results,
        "category_macro_average": macro,
        "category_count": len(category_results),
        "prompt_count": len(prompt_results),
    }
