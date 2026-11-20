import unittest

import numpy as np

from pwea.predictor.ridge import (
    MultiOutputRidge,
    project_simplex,
)


class TestSimplexProjection(unittest.TestCase):

    def test_nonnegative_and_sums_to_one(self):
        result = project_simplex(
            np.array([-0.5, 0.2, 0.8, 1.5])
        )

        self.assertTrue(np.all(result >= 0))
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_already_valid_distribution(self):
        values = np.array([0.2, 0.3, 0.5])
        result = project_simplex(values)

        np.testing.assert_allclose(result, values, atol=1e-12)

    def test_uniform_distribution_from_equal_scores(self):
        result = project_simplex(np.zeros(4))

        np.testing.assert_allclose(
            result,
            np.full(4, 0.25),
            atol=1e-12,
        )

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            project_simplex(np.array([]))

        with self.assertRaises(ValueError):
            project_simplex(np.array([1.0, np.nan]))


class TestMultiOutputRidge(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)

        self.X = rng.normal(size=(30, 12))
        self.W = rng.normal(size=(12, 8))
        self.Y = self.X @ self.W

    def test_fit_and_predict_shapes(self):
        model = MultiOutputRidge(alpha=0.01).fit(
            self.X, self.Y
        )

        predictions = model.predict(self.X[:5])

        self.assertEqual(predictions.shape, (5, 8))

    def test_dual_solution_matches_primal_solution(self):
        alpha = 0.5

        model = MultiOutputRidge(alpha=alpha).fit(
            self.X, self.Y
        )

        dual_prediction = model.predict(self.X)

        primal_weights = np.linalg.solve(
            self.X.T @ self.X + alpha * np.eye(self.X.shape[1]),
            self.X.T @ self.Y,
        )

        primal_prediction = self.X @ primal_weights

        np.testing.assert_allclose(
            dual_prediction,
            primal_prediction,
            rtol=1e-9,
            atol=1e-9,
        )

    def test_prediction_before_fit_rejected(self):
        model = MultiOutputRidge()

        with self.assertRaises(RuntimeError):
            model.predict(self.X[:1])

    def test_invalid_alpha_rejected(self):
        for alpha in [0, -1, np.nan, np.inf]:
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    MultiOutputRidge(alpha=alpha)

    def test_mismatched_sample_counts_rejected(self):
        model = MultiOutputRidge()

        with self.assertRaises(ValueError):
            model.fit(self.X, self.Y[:10])

    def test_non_finite_training_data_rejected(self):
        X = self.X.copy()
        X[0, 0] = np.nan

        with self.assertRaises(ValueError):
            MultiOutputRidge().fit(X, self.Y)

    def test_simplex_prediction_shape(self):
        rng = np.random.default_rng(7)

        X = rng.normal(size=(8, 10))
        Y = rng.random(size=(8, 48 * 128))

        model = MultiOutputRidge(alpha=1.0).fit(X, Y)
        predictions = model.predict_simplex(
            X[:2],
            n_layer=48,
            n_expert=128,
        )

        self.assertEqual(predictions.shape, (2, 6144))

        matrices = predictions.reshape(2, 48, 128)

        self.assertTrue(np.all(matrices >= 0))
        np.testing.assert_allclose(
            matrices.sum(axis=2),
            np.ones((2, 48)),
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
