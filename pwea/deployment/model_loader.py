
# pwea/deployment/model_loader.py

import hashlib
import json
from pathlib import Path

import numpy as np

from pwea.predictor.ridge import MultiOutputRidge
from pwea.predictor.text_features import transform_text_variant


N_LAYER = 48
N_EXPERT = 128
INPUT_DIM = 32768
OUTPUT_DIM = N_LAYER * N_EXPERT


class ArtifactError(RuntimeError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PWEAPredictor:

    def __init__(self, model, manifest):
        self.model = model
        self.manifest = manifest

    @classmethod
    def load(cls, model_path=None, manifest_path=None):
        base = Path(__file__).resolve().parents[1]
        artifact_dir = base / "artifacts" / "deployment"

        model_path = Path(model_path or artifact_dir / "pwea_model_v1.npz")
        manifest_path = Path(
            manifest_path or artifact_dir / "pwea_model_v1_manifest.json"
        )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactError(f"Cannot read manifest: {exc}") from exc

        if manifest.get("artifact_type") != "pwea_deployment_model":
            raise ArtifactError("Invalid artifact type")

        if manifest.get("schema_version") != 1:
            raise ArtifactError("Unsupported manifest schema")

        config = manifest.get("model", {})

        if config.get("algorithm") != "dual_multi_output_ridge":
            raise ArtifactError("Unsupported model algorithm")

        if config.get("feature_variant") != "balanced_hybrid":
            raise ArtifactError("Unexpected feature variant")

        if float(config.get("alpha", 0)) != 0.1:
            raise ArtifactError("Unexpected alpha")

        if config.get("feature_dimensions") != INPUT_DIM:
            raise ArtifactError("Unexpected feature dimensions")

        if config.get("output_dimensions") != OUTPUT_DIM:
            raise ArtifactError("Unexpected output dimensions")

        if config.get("layers") != N_LAYER:
            raise ArtifactError("Unexpected layer count")

        if config.get("experts_per_layer") != N_EXPERT:
            raise ArtifactError("Unexpected expert count")

        if manifest.get("model_file") != model_path.name:
            raise ArtifactError("Model filename does not match manifest")

        expected_hash = manifest.get("model_file_sha256")
        if not expected_hash or file_sha256(model_path) != expected_hash:
            raise ArtifactError("Model SHA-256 verification failed")

        try:
            with np.load(model_path, allow_pickle=False) as archive:
                if set(archive.files) != {"X_train", "dual_coefficients"}:
                    raise ArtifactError("Unexpected model array names")

                X_train = archive["X_train"]
                coefficients = archive["dual_coefficients"]
        except ArtifactError:
            raise
        except Exception as exc:
            raise ArtifactError(f"Cannot load model arrays: {exc}") from exc

        if X_train.shape != (140, INPUT_DIM):
            raise ArtifactError(f"Invalid X_train shape: {X_train.shape}")

        if coefficients.shape != (140, OUTPUT_DIM):
            raise ArtifactError(
                f"Invalid coefficient shape: {coefficients.shape}"
            )

        if X_train.dtype != np.float64:
            raise ArtifactError("X_train must be float64")

        if coefficients.dtype != np.float64:
            raise ArtifactError("Coefficients must be float64")

        if not np.all(np.isfinite(X_train)):
            raise ArtifactError("X_train contains non-finite values")

        if not np.all(np.isfinite(coefficients)):
            raise ArtifactError("Coefficients contain non-finite values")

        expected_arrays = manifest.get("arrays", {})

        for name, array in (
            ("X_train", X_train),
            ("dual_coefficients", coefficients),
        ):
            metadata = expected_arrays.get(name, {})
            if metadata.get("shape") != list(array.shape):
                raise ArtifactError(f"{name} manifest shape mismatch")
            if metadata.get("dtype") != str(array.dtype):
                raise ArtifactError(f"{name} manifest dtype mismatch")

        model = MultiOutputRidge(alpha=float(config["alpha"]))
        model.X_train = X_train
        model.dual_coefficients = coefficients
        model.output_dimension = OUTPUT_DIM

        return cls(model, manifest)

    def predict(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        features = transform_text_variant(
            prompt,
            self.manifest["model"]["feature_variant"],
        )

        if features.shape != (INPUT_DIM,):
            raise RuntimeError("Unexpected feature vector dimensions")

        prediction = self.model.predict_simplex(
            features.reshape(1, -1),
            n_layer=N_LAYER,
            n_expert=N_EXPERT,
        )

        matrix = prediction.reshape(N_LAYER, N_EXPERT)

        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("Prediction contains non-finite values")

        if not np.all(matrix >= 0):
            raise RuntimeError("Prediction contains negative values")

        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
            raise RuntimeError("Prediction simplex validation failed")

        return matrix
