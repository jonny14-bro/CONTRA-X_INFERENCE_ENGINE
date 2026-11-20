
# pwea/tests/test_deployment.py

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pwea.deployment.model_loader import (
    ArtifactError,
    PWEAPredictor,
)
from pwea.predictor.ridge import MultiOutputRidge
from pwea.predictor.text_features import transform_texts_variant


ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    ROOT / "pwea/artifacts/prediction/"
    "expert_prediction_samples_20261115.jsonl"
)

ARTIFACT_DIR = ROOT / "pwea/artifacts/deployment"
MODEL = ARTIFACT_DIR / "pwea_model_v1.npz"
MANIFEST = ARTIFACT_DIR / "pwea_model_v1_manifest.json"


class TestPWEADeployment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = PWEAPredictor.load(MODEL, MANIFEST)

    def test_artifact_loads(self):
        self.assertEqual(
            self.predictor.manifest["model_version"],
            "pwea_model_v1",
        )

    def test_prediction_shape(self):
        result = self.predictor.predict(
            "Implement a binary search tree in Python."
        )
        self.assertEqual(result.shape, (48, 128))

    def test_prediction_simplex(self):
        result = self.predictor.predict(
            "Explain recursion and demonstrate factorial."
        )

        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(np.isfinite(result)))

        np.testing.assert_allclose(
            result.sum(axis=1),
            np.ones(48),
            atol=1e-10,
        )

    def test_invalid_prompt_rejected(self):
        for prompt in ("", "   ", None, 123):
            with self.subTest(prompt=prompt):
                with self.assertRaises(ValueError):
                    self.predictor.predict(prompt)

    def test_saved_model_matches_fresh_training(self):
        samples = [
            json.loads(line)
            for line in DATASET.read_text().splitlines()
            if line.strip()
        ]

        eligible = [
            s for s in samples
            if s.get("prompt_text_status") == "verified"
            and isinstance(s.get("prompt_text"), str)
            and s["prompt_text"].strip()
        ]

        texts = [s["prompt_text"] for s in eligible]
        targets = np.asarray([
            np.asarray(
                s["target"]["activation_share"],
                dtype=np.float64,
            ).reshape(6144)
            for s in eligible
        ])

        X = transform_texts_variant(texts, "balanced_hybrid")

        reference = MultiOutputRidge(alpha=0.1).fit(X, targets)

        loaded = self.predictor.model

        np.testing.assert_array_equal(
            loaded.X_train,
            reference.X_train,
        )

        np.testing.assert_array_equal(
            loaded.dual_coefficients,
            reference.dual_coefficients,
        )

        probe = transform_texts_variant(
            ["Write a Python function to merge two sorted arrays."],
            "balanced_hybrid",
        )

        expected = reference.predict_simplex(probe)
        actual = loaded.predict_simplex(probe)

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_corrupted_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)

            bad_model = temporary / MODEL.name
            bad_manifest = temporary / MANIFEST.name

            shutil.copy2(MODEL, bad_model)
            shutil.copy2(MANIFEST, bad_manifest)

            with bad_model.open("ab") as stream:
                stream.write(b"corruption")

            with self.assertRaises(ArtifactError):
                PWEAPredictor.load(bad_model, bad_manifest)


if __name__ == "__main__":
    unittest.main()
