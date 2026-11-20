#!/usr/bin/env python3
"""Multi-output dual ridge regression for expert activation shares."""

import numpy as np


def project_simplex(vector):
    """Project a finite vector onto the nonnegative unit simplex."""
    vector = np.asarray(vector, dtype=np.float64)

    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("vector must be a non-empty one-dimensional array")

    if not np.all(np.isfinite(vector)):
        raise ValueError("vector must contain only finite values")

    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, vector.size + 1)

    valid = ordered - cumulative / indices > 0

    if not np.any(valid):
        raise RuntimeError("simplex projection failed")

    rho = np.nonzero(valid)[0][-1]
    threshold = cumulative[rho] / (rho + 1)

    projected = np.maximum(vector - threshold, 0.0)
    projected /= projected.sum()

    return projected


class MultiOutputRidge:
    """Dual-form ridge regression with multiple output dimensions."""

    def __init__(self, alpha=1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")

        self.alpha = float(alpha)
        self.X_train = None
        self.dual_coefficients = None
        self.output_dimension = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must both be two-dimensional")

        if X.shape[0] != Y.shape[0] or X.shape[0] == 0:
            raise ValueError("X and Y must have equal nonzero sample counts")

        if X.shape[1] == 0 or Y.shape[1] == 0:
            raise ValueError("X and Y must have nonzero feature dimensions")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            raise ValueError("X and Y must contain only finite values")

        gram = X @ X.T
        regularized = gram + self.alpha * np.eye(X.shape[0])

        self.dual_coefficients = np.linalg.solve(regularized, Y)
        self.X_train = X.copy()
        self.output_dimension = Y.shape[1]

        return self

    def predict(self, X):
        if self.dual_coefficients is None:
            raise RuntimeError("model must be fitted before prediction")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("prediction feature dimensions do not match")

        if not np.all(np.isfinite(X)):
            raise ValueError("prediction features must be finite")

        kernel = X @ self.X_train.T
        return kernel @ self.dual_coefficients

    def predict_simplex(self, X, n_layer=48, n_expert=128):
        """Predict and normalize each layer's expert distribution."""
        predictions = self.predict(X)

        expected = n_layer * n_expert
        if predictions.shape[1] != expected:
            raise ValueError(
                f"expected {expected} outputs, got {predictions.shape[1]}"
            )

        output = np.empty_like(predictions)

        for sample_index in range(predictions.shape[0]):
            matrix = predictions[sample_index].reshape(n_layer, n_expert)

            for layer_index in range(n_layer):
                output[sample_index, layer_index * n_expert:
                       (layer_index + 1) * n_expert] = project_simplex(
                    matrix[layer_index]
                )

        return output
