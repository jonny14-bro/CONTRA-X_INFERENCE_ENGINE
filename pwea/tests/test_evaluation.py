import unittest

import numpy as np

from pwea.predictor.evaluation import (
    aggregate_prompt_results,
    evaluate_prompt_predictions,
)


def make_sample(category="debugging", valid=None):
    if valid is None:
        valid = [True, True]

    actual = np.zeros((2, 13), dtype=float)
    actual[:, 0] = 1.0

    return {
        "sample_id": f"{category}:prompt-01",
        "session_id": f"{category}-s01",
        "category": category,
        "prompt_id": 1,
        "generation_censored": False,
        "target": {
            "activation_share": actual.tolist(),
            "valid_layer_mask": valid,
        },
    }


class TestPromptEvaluation(unittest.TestCase):

    def test_perfect_predictions_score_one(self):
        sample = make_sample()
        actual = np.asarray(sample["target"]["activation_share"])

        result = evaluate_prompt_predictions(
            [sample], [0], actual[None, :, :],
            n_layer=2, n_expert=13,
        )[0]

        expected_composite = (1.0 + 5.0 / 13.0 + 1.0) / 3.0
        self.assertAlmostEqual(
            result["composite_score"], expected_composite
        )
        self.assertAlmostEqual(result["tie_aware_precision_at_k"], 1.0)
        self.assertAlmostEqual(
            result["tie_aware_recall_at_k"], 5.0 / 13.0
        )
        self.assertAlmostEqual(result["tie_aware_ndcg_at_k"], 1.0)
        self.assertAlmostEqual(result["mae"], 0.0)
        self.assertAlmostEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["spearman"], 1.0)

    def test_invalid_layers_are_excluded(self):
        sample = make_sample(valid=[True, False])
        prediction = np.zeros((1, 2, 13))
        prediction[0, 0, 0] = 1.0
        prediction[0, 1, :] = 100.0

        result = evaluate_prompt_predictions(
            [sample], [0], prediction,
            n_layer=2, n_expert=13,
        )[0]

        self.assertEqual(result["valid_layer_count"], 1)
        self.assertAlmostEqual(result["mae"], 0.0)

    def test_flat_prediction_shape_is_supported(self):
        sample = make_sample()
        actual = np.asarray(sample["target"]["activation_share"])

        result = evaluate_prompt_predictions(
            [sample], [0], actual.reshape(1, -1),
            n_layer=2, n_expert=13,
        )

        self.assertEqual(len(result), 1)

    def test_rejects_non_finite_predictions(self):
        sample = make_sample()
        prediction = np.zeros((1, 2, 13))
        prediction[0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_prompt_predictions(
                [sample], [0], prediction,
                n_layer=2, n_expert=13,
            )

    def test_rejects_no_valid_layers(self):
        sample = make_sample(valid=[False, False])
        prediction = np.zeros((1, 2, 13))

        with self.assertRaisesRegex(ValueError, "no valid"):
            evaluate_prompt_predictions(
                [sample], [0], prediction,
                n_layer=2, n_expert=13,
            )


class TestPromptAggregation(unittest.TestCase):

    def test_categories_receive_equal_macro_weight(self):
        metric_fields = (
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k",
            "spearman",
            "tie_aware_precision_at_k",
            "tie_aware_recall_at_k",
            "tie_aware_ndcg_at_k",
            "mae",
            "rmse",
        )

        def row(category, score):
            return {
                "category": category,
                **{field: score for field in metric_fields},
                "composite_score": score,
            }

        rows = [
            row("A", 1.0),
            row("A", 1.0),
            row("B", 0.0),
        ]

        result = aggregate_prompt_results(rows)

        self.assertAlmostEqual(
            result["category_macro_average"]["composite_score"],
            0.5,
        )
        self.assertEqual(result["prompt_count"], 3)
        self.assertEqual(result["category_count"], 2)

    def test_rejects_empty_results(self):
        with self.assertRaises(ValueError):
            aggregate_prompt_results([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
