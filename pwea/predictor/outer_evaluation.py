#!/usr/bin/env python3
"""Leakage-controlled outer evaluation for PWEA's five-category LOCO design."""

import hashlib
import json
from pathlib import Path

import numpy as np

from pwea.predictor.baselines import build_activation_baselines
from pwea.predictor.evaluation import (
    aggregate_prompt_results,
    evaluate_prompt_predictions,
)
from pwea.predictor.ridge import MultiOutputRidge
from pwea.predictor.text_features import transform_texts_variant


N_LAYER = 48
N_EXPERT = 128

EXPECTED_DATASET_SHA256 = (
    "8a516ca4ff7618be88ca41309943b0c4ff33deab89469ec521bd72afc2ac968a"
)
EXPECTED_SELECTION_SUMMARY_SHA256 = (
    "3e6a09486b1d4501d56c96d00e5840a1a3bd1e8bfb4729d694a325a8938eaa6d"
)
EXPECTED_AMENDMENT_SHA256 = (
    "c152f7f0d99baff36a2063dbb90cb5cfbe97c408ed9d7a22631f0b259ed98a3e"
)

EXPECTED_CATEGORIES = (
    "algorithmic-problem-solving",
    "code-explanation",
    "code-generation",
    "code-transformation",
    "debugging",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path):
    samples = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: {exc}"
                ) from exc
    if not samples:
        raise ValueError("Dataset contains no samples")
    return samples


def _validate_inputs(samples, summary):
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a non-empty list")

    if set(summary.get("folds", {})) != set(EXPECTED_CATEGORIES):
        raise ValueError("selection summary does not contain the five expected folds")

    if summary.get("outer_test_evaluated") is not False:
        raise ValueError("selection summary indicates outer-test evaluation was used")

    sample_ids = [sample.get("sample_id") for sample in samples]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be present and unique")

    categories = {sample.get("category") for sample in samples}
    if categories != set(EXPECTED_CATEGORIES):
        raise ValueError("dataset categories do not match the frozen five-category cohort")

    for sample in samples:
        if sample.get("telemetry", {}).get("device") != "sim":
            raise ValueError("unexpected device: this evaluation is frozen for simulated telemetry")

        target = sample.get("target", {})
        shares = np.asarray(target.get("activation_share"), dtype=np.float64)
        valid = np.asarray(target.get("valid_layer_mask"), dtype=bool)
        events = np.asarray(target.get("layer_event_counts"), dtype=np.float64)

        if shares.shape != (N_LAYER, N_EXPERT):
            raise ValueError(f"{sample['sample_id']}: invalid activation_share shape")
        if valid.shape != (N_LAYER,) or events.shape != (N_LAYER,):
            raise ValueError(f"{sample['sample_id']}: invalid layer metadata shape")
        if not np.all(np.isfinite(shares)) or not np.all(np.isfinite(events)):
            raise ValueError(f"{sample['sample_id']}: non-finite target data")
        if np.any(shares < 0) or np.any(events < 0):
            raise ValueError(f"{sample['sample_id']}: negative target data")

        for layer in range(N_LAYER):
            if valid[layer]:
                if events[layer] <= 0 or not np.isclose(
                    shares[layer].sum(), 1.0, atol=1e-10
                ):
                    raise ValueError(f"{sample['sample_id']}: invalid target layer")
            elif events[layer] != 0:
                raise ValueError(f"{sample['sample_id']}: invalid layer has events")


def _load_and_verify_fold_report(category, item, report_dir, samples):
    report_path = Path(report_dir) / item["report"]
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing selection report: {report_path}")

    actual_hash = _sha256(report_path)
    if actual_hash != item.get("report_sha256"):
        raise ValueError(f"{category}: selection report SHA256 mismatch")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    outer = report.get("outer_fold", {})
    expected_training = sorted(
        candidate
        for candidate in EXPECTED_CATEGORIES
        if candidate != category
        and (candidate != "code-generation" or category == "code-generation")
    )

    if outer.get("heldout_category") != category:
        raise ValueError(f"{category}: held-out category mismatch")
    if sorted(report.get("training_categories", [])) != expected_training:
        raise ValueError(f"{category}: training category leakage or mismatch")
    if outer.get("outer_test_used_for_selection") is not False:
        raise ValueError(f"{category}: outer test was used during model selection")

    variant = item.get("selected_feature_variant")
    alpha = item.get("selected_alpha")
    best = report.get("best_configuration", {})

    if best.get("feature_variant") != variant or best.get("alpha") != alpha:
        raise ValueError(f"{category}: selected configuration disagrees with fold report")

    return {
        "path": str(report_path),
        "sha256": actual_hash,
        "training_categories": expected_training,
        "feature_variant": variant,
        "alpha": float(alpha),
    }


def _target_matrix(samples, indices):
    return np.vstack([
        np.asarray(
            samples[index]["target"]["activation_share"],
            dtype=np.float64,
        ).reshape(1, -1)
        for index in indices
    ])


def _evaluate_method(samples, indices, prediction):
    if not indices:
        return []
    predictions = np.repeat(
        np.asarray(prediction, dtype=np.float64)[None, :, :],
        len(indices),
        axis=0,
    ) if np.asarray(prediction).ndim == 2 else np.asarray(prediction)

    return evaluate_prompt_predictions(
        samples,
        indices,
        predictions,
        n_layer=N_LAYER,
        n_expert=N_EXPERT,
    )


def _run_scenario(samples, summary, report_dir, exclude_censored):
    folds = {}
    all_baseline_results = []
    all_text_results = []

    for category in EXPECTED_CATEGORIES:
        item = summary["folds"][category]
        provenance = _load_and_verify_fold_report(
            category, item, report_dir, samples
        )

        eligible = [
            index for index, sample in enumerate(samples)
            if not exclude_censored or not sample.get("generation_censored", False)
        ]
        training = [
            index for index in eligible
            if samples[index]["category"] != category
        ]
        testing = [
            index for index in eligible
            if samples[index]["category"] == category
        ]

        text_training = [
            index for index in training
            if samples[index].get("prompt_text_status") == "verified"
            and isinstance(samples[index].get("prompt_text"), str)
            and samples[index]["prompt_text"].strip()
        ]
        text_testing = [
            index for index in testing
            if samples[index].get("prompt_text_status") == "verified"
            and isinstance(samples[index].get("prompt_text"), str)
            and samples[index]["prompt_text"].strip()
        ]

        if not training or not testing:
            raise ValueError(f"{category}: empty outer training or test partition")

        if any(samples[i]["category"] == category for i in training):
            raise RuntimeError(f"{category}: training leakage detected")
        if any(samples[i]["category"] != category for i in testing):
            raise RuntimeError(f"{category}: test partition contains another category")

        baseline = build_activation_baselines(
            samples, training, n_layer=N_LAYER, n_expert=N_EXPERT
        )

        equal_results = _evaluate_method(
            samples, testing, baseline["mean_prompt_activation_share"]
        )
        pooled_results = _evaluate_method(
            samples, testing, baseline["aggregate_activation_share"]
        )

        all_baseline_results.extend(equal_results)
        all_baseline_results.extend(pooled_results)

        fold = {
            "heldout_category": category,
            "training_categories": provenance["training_categories"],
            "training_prompt_count": len(training),
            "test_prompt_count": len(testing),
            "text_training_prompt_count": len(text_training),
            "text_test_prompt_count": len(text_testing),
            "generation_censored_excluded": bool(exclude_censored),
            "selection": provenance,
            "baselines": {
                "equal_prompt": {
                    "prompt_results": equal_results,
                    "aggregate": aggregate_prompt_results(equal_results),
                },
                "pooled_frequency": {
                    "prompt_results": pooled_results,
                    "aggregate": aggregate_prompt_results(pooled_results),
                },
            },
        }

        if text_testing:
            if not text_training:
                raise ValueError(f"{category}: text test exists but text training is empty")

            variant = provenance["feature_variant"]
            alpha = provenance["alpha"]

            train_texts = [samples[i]["prompt_text"] for i in text_training]
            test_texts = [samples[i]["prompt_text"] for i in text_testing]

            x_train = transform_texts_variant(train_texts, variant)
            x_test = transform_texts_variant(test_texts, variant)
            y_train = _target_matrix(samples, text_training)

            model = MultiOutputRidge(alpha=alpha).fit(x_train, y_train)
            predictions = model.predict_simplex(
                x_test, n_layer=N_LAYER, n_expert=N_EXPERT
            ).reshape(len(text_testing), N_LAYER, N_EXPERT)

            text_results = evaluate_prompt_predictions(
                samples,
                text_testing,
                predictions,
                n_layer=N_LAYER,
                n_expert=N_EXPERT,
            )
            all_text_results.extend(text_results)

            fold["text_ridge"] = {
                "status": "evaluated",
                "feature_variant": variant,
                "alpha": alpha,
                "prompt_results": text_results,
                "aggregate": aggregate_prompt_results(text_results),
            }
        else:
            fold["text_ridge"] = {
                "status": "unavailable",
                "reason": "No verified prompt text in the held-out category",
                "feature_variant": provenance["feature_variant"],
                "alpha": provenance["alpha"],
                "prompt_results": [],
                "aggregate": None,
            }

        folds[category] = fold

    baseline_equal = [
        row
        for category in EXPECTED_CATEGORIES
        for row in folds[category]["baselines"]["equal_prompt"]["prompt_results"]
    ]
    baseline_pooled = [
        row
        for category in EXPECTED_CATEGORIES
        for row in folds[category]["baselines"]["pooled_frequency"]["prompt_results"]
    ]

    return {
        "censoring_analysis": (
            "sensitivity_excluding_generation_censored"
            if exclude_censored else "primary_including_generation_censored"
        ),
        "folds": folds,
        "overall": {
            "equal_prompt_baseline": aggregate_prompt_results(baseline_equal),
            "pooled_frequency_baseline": aggregate_prompt_results(baseline_pooled),
            "text_ridge": (
                aggregate_prompt_results(all_text_results)
                if all_text_results else None
            ),
        },
        "coverage": {
            "baseline_test_prompts": len(baseline_equal),
            "text_ridge_test_prompts": len(all_text_results),
            "text_ridge_categories": sorted({
                row["category"] for row in all_text_results
            }),
        },
    }


def run_outer_evaluation_from_files(
    dataset_path,
    selection_summary_path,
    amendment_path,
    report_dir,
):
    """Verify frozen artifacts, then run primary and censoring sensitivity."""
    dataset_path = Path(dataset_path)
    selection_summary_path = Path(selection_summary_path)
    amendment_path = Path(amendment_path)

    dataset_hash = _sha256(dataset_path)
    summary_hash = _sha256(selection_summary_path)
    amendment_hash = _sha256(amendment_path)

    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise ValueError("Dataset SHA256 does not match frozen protocol")
    if summary_hash != EXPECTED_SELECTION_SUMMARY_SHA256:
        raise ValueError("Selection summary SHA256 does not match frozen artifact")
    if amendment_hash != EXPECTED_AMENDMENT_SHA256:
        raise ValueError("Protocol amendment SHA256 does not match frozen artifact")

    samples = _load_jsonl(dataset_path)
    summary = json.loads(selection_summary_path.read_text(encoding="utf-8"))

    _validate_inputs(samples, summary)

    provenance = summary.get("provenance", {})
    if provenance.get("dataset_sha256") != dataset_hash:
        raise ValueError("Selection summary dataset provenance mismatch")
    if provenance.get("protocol_amendment_sha256") != amendment_hash:
        raise ValueError("Selection summary amendment provenance mismatch")

    primary = _run_scenario(
        samples, summary, report_dir, exclude_censored=False
    )
    sensitivity = _run_scenario(
        samples, summary, report_dir, exclude_censored=True
    )

    return {
        "status": "completed",
        "evaluation_design": "five_fold_leave_one_category_out",
        "device_scope": "simulated_telemetry_only",
        "provenance": {
            "dataset_sha256": dataset_hash,
            "selection_summary_sha256": summary_hash,
            "protocol_amendment_sha256": amendment_hash,
        },
        "sample_count": len(samples),
        "primary": primary,
        "censoring_sensitivity": sensitivity,
        "limitations": [
            "All telemetry was collected with device=sim; no real hardware speedup is established.",
            "Code-generation prompt text is unresolved, so text ridge cannot be evaluated on that held-out category.",
            "Text ridge metrics cover verified-text held-out categories only; baseline metrics cover all held-out categories.",
            "The primary analysis retains generation-censored prompts; the sensitivity analysis excludes them.",
        ],
    }
