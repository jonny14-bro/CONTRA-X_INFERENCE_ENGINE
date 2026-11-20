import unittest
from unittest.mock import patch

import numpy as np

from pwea.predictor import model_selection


CATEGORIES = ("debugging", "code-explanation")


def make_samples():
    samples = []
    target = {
        "activation_share": (
            np.full((48, 128), 1.0 / 128.0).tolist()
        )
    }

    for category in CATEGORIES:
        for session_number in range(1, 6):
            session_id = f"{category}-session-{session_number}"
            for prompt_id in range(1, 8):
                samples.append({
                    "sample_id": f"{session_id}-prompt-{prompt_id}",
                    "session_id": session_id,
                    "prompt_id": prompt_id,
                    "category": category,
                    "prompt_text": f"Verified prompt {category} {prompt_id}",
                    "prompt_text_status": "verified",
                    "generation_censored": False,
                    "target": target,
                })

    # Unresolved text in an outer-training category must be excluded.
    samples.append({
        "sample_id": "debugging-unresolved",
        "session_id": "debugging-session-1",
        "prompt_id": 1,
        "category": "debugging",
        "prompt_text": None,
        "prompt_text_status": "unresolved",
        "target": target,
    })

    # A category outside the outer-training set must also be excluded.
    samples.append({
        "sample_id": "outer-heldout-sample",
        "session_id": "heldout-session-1",
        "prompt_id": 1,
        "category": "code-generation",
        "prompt_text": "Must not be used",
        "prompt_text_status": "verified",
        "target": target,
    })

    return samples


class FakeRidge:
    fit_calls = 0

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, Y):
        type(self).fit_calls += 1
        self.training_rows = len(X)
        return self

    def predict_simplex(self, X, n_layer=48, n_expert=128):
        return np.full((len(X), n_layer, n_expert), 1.0 / n_expert)


def fake_evaluate(samples, indices, predictions, **kwargs):
    return [
        {
            "sample_id": samples[index]["sample_id"],
            "session_id": samples[index]["session_id"],
            "category": samples[index]["category"],
            "prompt_id": samples[index]["prompt_id"],
            "generation_censored": False,
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
            "composite_score": 0.2,
        }
        for index in indices
    ]


class TestInnerModelSelection(unittest.TestCase):
    def setUp(self):
        FakeRidge.fit_calls = 0

    def test_runs_every_configuration_on_all_five_folds(self):
        samples = make_samples()

        with (
            patch.object(
                model_selection,
                "transform_texts_variant",
                side_effect=lambda texts, variant: np.ones((len(texts), 4)),
            ),
            patch.object(model_selection, "MultiOutputRidge", FakeRidge),
            patch.object(
                model_selection,
                "evaluate_prompt_predictions",
                side_effect=fake_evaluate,
            ),
        ):
            report = model_selection.run_inner_model_selection(
                samples, CATEGORIES, alpha_grid=(0.1, 1.0)
            )

        self.assertEqual(report["verified_text_sample_count"], 70)
        self.assertEqual(report["session_count"], 10)
        self.assertEqual(report["prompt_count"], 70)
        self.assertEqual(report["configuration_count"], 6)
        self.assertEqual(len(report["configurations"]), 6)
        self.assertEqual(FakeRidge.fit_calls, 3 * 2 * 5)

        for configuration in report["configurations"]:
            self.assertEqual(len(configuration["folds"]), 5)
            self.assertEqual(
                sum(
                    fold["validation_prompt_count"]
                    for fold in configuration["folds"]
                ),
                70,
            )
            for fold in configuration["folds"]:
                self.assertEqual(fold["training_prompt_count"], 56)
                self.assertEqual(fold["validation_prompt_count"], 14)
                self.assertEqual(fold["training_session_count"], 8)
                self.assertEqual(fold["validation_session_count"], 2)

    def test_tie_breaks_are_deterministic(self):
        samples = make_samples()

        with (
            patch.object(
                model_selection,
                "transform_texts_variant",
                side_effect=lambda texts, variant: np.ones((len(texts), 4)),
            ),
            patch.object(model_selection, "MultiOutputRidge", FakeRidge),
            patch.object(
                model_selection,
                "evaluate_prompt_predictions",
                side_effect=fake_evaluate,
            ),
        ):
            report = model_selection.run_inner_model_selection(
                samples, CATEGORIES, alpha_grid=(1.0, 0.1)
            )

        best = report["best_configuration"]
        self.assertEqual(best["feature_variant"], "word_only")
        self.assertEqual(best["alpha"], 0.1)

    def test_rejects_duplicate_categories(self):
        with self.assertRaises(ValueError):
            model_selection.run_inner_model_selection(
                make_samples(), ("debugging", "debugging")
            )

    def test_rejects_duplicate_alpha_values(self):
        with self.assertRaises(ValueError):
            model_selection.run_inner_model_selection(
                make_samples(), CATEGORIES, alpha_grid=(0.1, 0.1)
            )


if __name__ == "__main__":
    unittest.main()
