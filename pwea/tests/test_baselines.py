import unittest

import numpy as np

from pwea.predictor.baselines import build_activation_baselines


def make_sample(shares, events, valid=None):
    shares = np.asarray(shares, dtype=float)
    if valid is None:
        valid = [True] * len(events)

    return {
        "sample_id": "synthetic",
        "target": {
            "activation_share": shares.tolist(),
            "valid_layer_mask": list(valid),
            "layer_event_counts": list(events),
        },
    }


class TestActivationBaselines(unittest.TestCase):

    def test_equal_prompt_weights_prompts_equally(self):
        samples = [
            make_sample([[1.0, 0.0]], [100]),
            make_sample([[0.0, 1.0]], [1]),
        ]

        result = build_activation_baselines(
            samples, [0, 1], n_layer=1, n_expert=2
        )

        np.testing.assert_allclose(
            result["mean_prompt_activation_share"],
            [[0.5, 0.5]],
        )

    def test_pooled_frequency_weights_by_event_count(self):
        samples = [
            make_sample([[1.0, 0.0]], [100]),
            make_sample([[0.0, 1.0]], [1]),
        ]

        result = build_activation_baselines(
            samples, [0, 1], n_layer=1, n_expert=2
        )

        np.testing.assert_allclose(
            result["aggregate_activation_share"],
            [[100 / 101, 1 / 101]],
        )

    def test_inactive_expert_contributes_zero(self):
        samples = [
            make_sample([[1.0, 0.0]], [10]),
            make_sample([[1.0, 0.0]], [10]),
        ]

        result = build_activation_baselines(
            samples, [0, 1], n_layer=1, n_expert=2
        )

        np.testing.assert_allclose(
            result["mean_prompt_activation_share"],
            [[1.0, 0.0]],
        )

    def test_invalid_layer_is_excluded_from_pooled_counts(self):
        samples = [
            make_sample([[1.0, 0.0]], [10]),
            make_sample([[0.0, 0.0]], [0], [False]),
        ]

        result = build_activation_baselines(
            samples, [0, 1], n_layer=1, n_expert=2
        )

        # Equal-prompt behavior matches build_profile.py's denominator:
        # the invalid prompt contributes zero but remains in prompt_count.
        np.testing.assert_allclose(
            result["mean_prompt_activation_share"],
            [[0.5, 0.0]],
        )
        np.testing.assert_allclose(
            result["aggregate_activation_share"],
            [[1.0, 0.0]],
        )

    def test_empty_layer_has_zero_baselines(self):
        samples = [
            make_sample([[0.0, 0.0]], [0], [False]),
        ]

        result = build_activation_baselines(
            samples, [0], n_layer=1, n_expert=2
        )

        np.testing.assert_array_equal(
            result["mean_prompt_activation_share"],
            [[0.0, 0.0]],
        )
        np.testing.assert_array_equal(
            result["aggregate_activation_share"],
            [[0.0, 0.0]],
        )

    def test_rejects_duplicate_training_indices(self):
        samples = [make_sample([[1.0, 0.0]], [1])]

        with self.assertRaisesRegex(ValueError, "unique"):
            build_activation_baselines(
                samples, [0, 0], n_layer=1, n_expert=2
            )

    def test_rejects_invalid_target_normalization(self):
        samples = [make_sample([[0.2, 0.2]], [10])]

        with self.assertRaisesRegex(ValueError, "sum to one"):
            build_activation_baselines(
                samples, [0], n_layer=1, n_expert=2
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
