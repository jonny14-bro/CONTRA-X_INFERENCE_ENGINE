import unittest
from unittest.mock import patch

import numpy as np

from pwea.predictor import outer_evaluation


CATEGORIES = outer_evaluation.EXPECTED_CATEGORIES


def make_samples():
    samples = []
    shares = np.full((48, 128), 1.0 / 128.0).tolist()

    for category in CATEGORIES:
        unresolved = category == "code-generation"
        samples.append({
            "sample_id": f"{category}-prompt-1",
            "session_id": f"{category}-session-1",
            "prompt_id": 1,
            "category": category,
            "prompt_text": None if unresolved else f"Verified {category} prompt",
            "prompt_text_status": "unresolved" if unresolved else "verified",
            "generation_censored": False,
            "telemetry": {"device": "sim"},
            "target": {
                "activation_share": shares,
                "valid_layer_mask": [True] * 48,
                "layer_event_counts": [100] * 48,
            },
        })

    # One generation-censored algorithmic prompt for sensitivity checks.
    samples.append({
        "sample_id": "algorithmic-censored-prompt-2",
        "session_id": "algorithmic-session-2",
        "prompt_id": 2,
        "category": "algorithmic-problem-solving",
        "prompt_text": "Verified but generation-censored prompt",
        "prompt_text_status": "verified",
        "generation_censored": True,
        "telemetry": {"device": "sim"},
        "target": {
            "activation_share": shares,
            "valid_layer_mask": [True] * 48,
            "layer_event_counts": [100] * 48,
        },
    })

    return samples


def fake_fold_report(category, item, report_dir, samples):
    return {
        "path": f"mock/{category}.json",
        "sha256": "mock-hash",
        "training_categories": sorted(set(CATEGORIES) - {category}),
        "feature_variant": "balanced_hybrid",
        "alpha": 0.01 if category == "debugging" else 0.1,
    }


def fake_evaluate(samples, indices, predictions, **kwargs):
    rows = []
    for index in indices:
        sample = samples[index]
        rows.append({
            "sample_id": sample["sample_id"],
            "session_id": sample["session_id"],
            "category": sample["category"],
            "prompt_id": sample["prompt_id"],
            "generation_censored": bool(
                sample.get("generation_censored", False)
            ),
            "valid_layer_count": 48,
            "precision_at_k": 0.2,
            "recall_at_k": 0.1,
            "ndcg_at_k": 0.3,
            "spearman": 0.0,
            "tie_aware_precision_at_k": 0.2,
            "tie_aware_recall_at_k": 0.1,
            "tie_aware_ndcg_at_k": 0.3,
            "mae": 0.05,
            "rmse": 0.06,
            "composite_score": (0.2 + 0.1 + 0.3) / 3,
        })
    return rows


class FakeRidge:
    instances = []

    def __init__(self, alpha):
        self.alpha = alpha
        type(self).instances.append(self)

    def fit(self, X, Y):
        self.training_rows = len(X)
        self.target_rows = len(Y)
        return self

    def predict_simplex(self, X, n_layer=48, n_expert=128):
        return np.full(
            (len(X), n_layer * n_expert),
            1.0 / n_expert,
        )


class TestOuterEvaluation(unittest.TestCase):
    def setUp(self):
        FakeRidge.instances = []
        self.samples = make_samples()
        self.summary = {
            "folds": {category: {} for category in CATEGORIES}
        }

    def run_scenario(self, exclude_censored=False):
        with (
            patch.object(
                outer_evaluation,
                "_load_and_verify_fold_report",
                side_effect=fake_fold_report,
            ),
            patch.object(
                outer_evaluation,
                "evaluate_prompt_predictions",
                side_effect=fake_evaluate,
            ),
            patch.object(
                outer_evaluation,
                "transform_texts_variant",
                side_effect=lambda texts, variant: np.ones((len(texts), 3)),
            ),
            patch.object(outer_evaluation, "MultiOutputRidge", FakeRidge),
        ):
            return outer_evaluation._run_scenario(
                self.samples,
                self.summary,
                "mock-reports",
                exclude_censored=exclude_censored,
            )

    def test_primary_keeps_censored_prompt_and_all_baseline_tests(self):
        report = self.run_scenario(exclude_censored=False)

        self.assertEqual(report["coverage"]["baseline_test_prompts"], 6)
        self.assertEqual(report["coverage"]["text_ridge_test_prompts"], 5)
        self.assertEqual(
            report["folds"]["algorithmic-problem-solving"]["test_prompt_count"],
            2,
        )
        self.assertEqual(
            report["folds"]["algorithmic-problem-solving"]["text_test_prompt_count"],
            2,
        )

    def test_sensitivity_excludes_censored_prompt_from_train_and_test(self):
        report = self.run_scenario(exclude_censored=True)

        self.assertEqual(report["coverage"]["baseline_test_prompts"], 5)
        self.assertEqual(report["coverage"]["text_ridge_test_prompts"], 4)
        self.assertEqual(
            report["folds"]["algorithmic-problem-solving"]["test_prompt_count"],
            1,
        )
        self.assertEqual(
            report["folds"]["debugging"]["training_prompt_count"],
            4,
        )
        self.assertEqual(
            report["folds"]["debugging"]["text_training_prompt_count"],
            3,
        )

    def test_unresolved_codegen_text_is_not_used_or_evaluated(self):
        report = self.run_scenario()

        codegen = report["folds"]["code-generation"]["text_ridge"]
        self.assertEqual(codegen["status"], "unavailable")
        self.assertEqual(codegen["prompt_results"], [])
        self.assertIsNone(codegen["aggregate"])
        self.assertEqual(
            report["folds"]["code-generation"]["text_training_prompt_count"],
            5,
        )

    def test_selected_alpha_and_feature_variant_are_used(self):
        self.run_scenario()

        self.assertEqual(len(FakeRidge.instances), 4)
        self.assertEqual(
            sorted(model.alpha for model in FakeRidge.instances),
            [0.01, 0.1, 0.1, 0.1],
        )
        self.assertTrue(
            all(model.training_rows > 0 for model in FakeRidge.instances)
        )

    def test_each_fold_trains_only_on_other_categories(self):
        report = self.run_scenario()

        for category, fold in report["folds"].items():
            self.assertEqual(
                fold["training_categories"],
                sorted(set(CATEGORIES) - {category}),
            )
            self.assertEqual(
                fold["training_prompt_count"],
                4 if category == "algorithmic-problem-solving" else 5,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
