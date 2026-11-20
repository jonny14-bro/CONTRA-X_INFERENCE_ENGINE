import unittest

import numpy as np

from pwea.predictor.text_features import (
    FEATURE_DIMENSIONS,
    _hash_feature,
    transform_text,
    transform_texts,
)


class TestTextFeatures(unittest.TestCase):

    def test_hashing_is_deterministic(self):
        self.assertEqual(
            _hash_feature("word:python"),
            _hash_feature("word:python"),
        )

    def test_output_dimension_and_dtype(self):
        vector = transform_text("Explain this Python function.")

        self.assertEqual(vector.shape, (FEATURE_DIMENSIONS,))
        self.assertEqual(vector.dtype, np.float64)

    def test_output_is_l2_normalized(self):
        vector = transform_text("Find the maximum subarray sum.")

        self.assertAlmostEqual(np.linalg.norm(vector), 1.0)

    def test_same_text_produces_same_vector(self):
        text = "def add(a, b): return a + b"

        np.testing.assert_array_equal(
            transform_text(text),
            transform_text(text),
        )

    def test_different_texts_produce_different_vectors(self):
        first = transform_text("Python function for sorting")
        second = transform_text("Database transaction rollback")

        self.assertFalse(np.array_equal(first, second))

    def test_batch_shape(self):
        matrix = transform_texts([
            "Explain recursion.",
            "Implement binary search.",
        ])

        self.assertEqual(
            matrix.shape,
            (2, FEATURE_DIMENSIONS),
        )

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            transform_text("   ")

    def test_empty_batch_rejected(self):
        with self.assertRaises(ValueError):
            transform_texts([])


if __name__ == "__main__":
    unittest.main()


class TestFeatureVariants(unittest.TestCase):

    def setUp(self):
        from pwea.predictor.text_features import (
            transform_text_variant,
            transform_texts_variant,
        )
        self.transform = transform_text_variant
        self.transform_batch = transform_texts_variant

    def test_word_only_shape_and_norm(self):
        result = self.transform("Explain Python dictionaries.", "word_only")
        self.assertEqual(result.shape, (16384,))
        self.assertAlmostEqual(np.linalg.norm(result), 1.0)

    def test_character_only_shape_and_norm(self):
        result = self.transform(
            "Explain Python dictionaries.", "character_only"
        )
        self.assertEqual(result.shape, (16384,))
        self.assertAlmostEqual(np.linalg.norm(result), 1.0)

    def test_balanced_hybrid_has_equal_block_norms(self):
        result = self.transform(
            "Explain Python dictionaries.", "balanced_hybrid"
        )
        self.assertEqual(result.shape, (32768,))
        self.assertAlmostEqual(
            np.linalg.norm(result[:16384]), 1.0 / np.sqrt(2.0)
        )
        self.assertAlmostEqual(
            np.linalg.norm(result[16384:]), 1.0 / np.sqrt(2.0)
        )
        self.assertAlmostEqual(np.linalg.norm(result), 1.0)

    def test_batch_transform_shape(self):
        result = self.transform_batch(
            ["Sort a list.", "Explain a loop."],
            "balanced_hybrid",
        )
        self.assertEqual(result.shape, (2, 32768))

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            self.transform("Some prompt.", "unknown")

    def test_empty_batch_rejected(self):
        with self.assertRaises(ValueError):
            self.transform_batch([], "word_only")
