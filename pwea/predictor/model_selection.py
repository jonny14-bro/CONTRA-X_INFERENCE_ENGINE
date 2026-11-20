#!/usr/bin/env python3
"""Session-grouped inner model selection for PWEA text ridge regression."""

from pwea.predictor.inner_validation import build_inner_folds
from pwea.predictor.text_features import (
    FEATURE_VARIANTS,
    transform_texts_variant,
)
from pwea.predictor.ridge import MultiOutputRidge
from pwea.predictor.evaluation import (
    evaluate_prompt_predictions,
    aggregate_prompt_results,
)


DEFAULT_ALPHA_GRID = (
    0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0
)
N_LAYER = 48
N_EXPERT = 128


def run_inner_model_selection(
    samples,
    outer_training_categories,
    alpha_grid=DEFAULT_ALPHA_GRID,
):
    """
    Select feature variant and alpha using only verified-text samples
    belonging to the outer training categories.

    All five session-grouped validation folds are evaluated. Prompt results
    are pooled across folds and aggregated equally within category, then
    equally across categories.

    Returns a JSON-serializable report. Does not write files.
    """
    categories = tuple(outer_training_categories)
    alphas = tuple(float(value) for value in alpha_grid)

    if not categories or len(set(categories)) != len(categories):
        raise ValueError("outer_training_categories must be non-empty and unique")
    if not alphas or any(value <= 0 for value in alphas):
        raise ValueError("alpha_grid must contain positive values")
    if len(set(alphas)) != len(alphas):
        raise ValueError("alpha_grid values must be unique")

    eligible = [
        sample for sample in samples
        if sample.get("category") in categories
        and sample.get("prompt_text_status") == "verified"
    ]

    if any(not isinstance(s.get("prompt_text"), str)
           or not s["prompt_text"].strip() for s in eligible):
        raise ValueError("verified samples must contain non-empty prompt text")

    if not eligible:
        raise ValueError("no verified-text samples in outer training categories")

    # Folds are built on this compact, verified-text-only sample list.
    folds = build_inner_folds(eligible, categories, n_splits=5)

    targets = []
    for sample in eligible:
        matrix = sample["target"]["activation_share"]
        if len(matrix) != N_LAYER or any(len(row) != N_EXPERT for row in matrix):
            raise ValueError(
                f"{sample.get('sample_id')}: expected target shape "
                f"{N_LAYER}x{N_EXPERT}"
            )
        targets.append(matrix)

    import numpy as np
    Y = np.asarray(targets, dtype=np.float64).reshape(
        len(eligible), N_LAYER * N_EXPERT
    )
    if not np.all(np.isfinite(Y)):
        raise ValueError("training targets contain non-finite values")

    texts = [sample["prompt_text"] for sample in eligible]
    configurations = []

    for variant in FEATURE_VARIANTS:
        X = transform_texts_variant(texts, variant)

        if X.shape[0] != len(eligible) or not np.all(np.isfinite(X)):
            raise ValueError(f"invalid feature matrix for variant {variant}")

        for alpha in alphas:
            all_prompt_results = []
            fold_reports = []

            for fold in folds:
                train_indices = fold["train_indices"]
                validation_indices = fold["validation_indices"]

                model = MultiOutputRidge(alpha=alpha)
                model.fit(X[train_indices], Y[train_indices])

                predictions = model.predict_simplex(
                    X[validation_indices],
                    n_layer=N_LAYER,
                    n_expert=N_EXPERT,
                )

                prompt_results = evaluate_prompt_predictions(
                    eligible,
                    validation_indices,
                    predictions,
                    k=5,
                    relevant_k=13,
                    n_layer=N_LAYER,
                    n_expert=N_EXPERT,
                )
                fold_summary = aggregate_prompt_results(prompt_results)

                fold_reports.append({
                    "fold": fold["fold"],
                    "training_session_count": len(fold["train_session_ids"]),
                    "validation_session_count": len(fold["validation_session_ids"]),
                    "training_prompt_count": len(train_indices),
                    "validation_prompt_count": len(validation_indices),
                    "validation_sessions": fold["validation_session_ids"],
                    "metrics": fold_summary,
                })
                all_prompt_results.extend(prompt_results)

            summary = aggregate_prompt_results(all_prompt_results)
            macro = summary["category_macro_average"]

            configurations.append({
                "feature_variant": variant,
                "alpha": alpha,
                "validation_metrics": summary,
                "folds": fold_reports,
            })

    # Stable final tie-breaks after the protocol's composite and MAE rules:
    # declared feature-variant order, then ascending alpha.
    variant_order = {
        name: index for index, name in enumerate(FEATURE_VARIANTS)
    }
    configurations.sort(
        key=lambda row: (
            -row["validation_metrics"]["category_macro_average"]["composite_score"],
            row["validation_metrics"]["category_macro_average"]["mae"],
            variant_order[row["feature_variant"]],
            row["alpha"],
        )
    )

    best = configurations[0]
    return {
        "status": "completed",
        "selection_method": "five_fold_category_stratified_session_grouped",
        "training_categories": list(categories),
        "verified_text_sample_count": len(eligible),
        "session_count": len({s["session_id"] for s in eligible}),
        "prompt_count": len(eligible),
        "feature_variants": list(FEATURE_VARIANTS),
        "alpha_grid": list(alphas),
        "configuration_count": len(configurations),
        "selection_rule": [
            "highest macro-averaged tie-aware ranking composite",
            "lowest macro-averaged MAE",
            "feature variant declaration order",
            "ascending alpha",
        ],
        "best_configuration": {
            "feature_variant": best["feature_variant"],
            "alpha": best["alpha"],
            "validation_metrics": best["validation_metrics"],
        },
        "configurations": configurations,
    }
