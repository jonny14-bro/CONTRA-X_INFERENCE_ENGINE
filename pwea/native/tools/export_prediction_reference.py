import json
import struct
from pathlib import Path

import numpy as np

from pwea.deployment.model_loader import PWEAPredictor


ROOT = Path(__file__).resolve().parents[3]

FEATURE_MANIFEST = (
    ROOT / "pwea/native/tests/fixtures/"
    "pwea_feature_reference.json"
)

OUTPUT = (
    ROOT / "pwea/native/tests/fixtures/"
    "pwea_prediction_reference.bin"
)

MAGIC = b"PWEAPRF1"
N_LAYER = 48
N_EXPERT = 128
OUTPUT_DIM = N_LAYER * N_EXPERT


def main():
    manifest = json.loads(FEATURE_MANIFEST.read_text())

    if manifest["feature_variant"] != "balanced_hybrid":
        raise RuntimeError("Unexpected feature variant")

    predictor = PWEAPredictor.load()

    cases = manifest["cases"]

    with OUTPUT.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<I", len(cases)))

        for case in cases:
            prompt = case["prompt"]
            prompt_bytes = prompt.encode("utf-8")

            prediction = np.asarray(
                predictor.predict(prompt),
                dtype="<f8",
            )

            if prediction.shape != (N_LAYER, N_EXPERT):
                raise RuntimeError(
                    f"Unexpected prediction shape: {prediction.shape}"
                )

            if not np.isfinite(prediction).all():
                raise RuntimeError("Non-finite reference prediction")

            if np.any(prediction < 0):
                raise RuntimeError("Negative reference prediction")

            sums = prediction.sum(axis=1)

            if not np.allclose(sums, 1.0, atol=1e-10):
                raise RuntimeError("Python simplex validation failed")

            stream.write(struct.pack("<I", len(prompt_bytes)))
            stream.write(prompt_bytes)
            stream.write(prediction.tobytes(order="C"))

            print(
                f"{case['name']}: PASS | "
                f"shape={prediction.shape} | "
                f"min={prediction.min():.12g} | "
                f"max={prediction.max():.12g}"
            )

    print()
    print("PYTHON PREDICTION REFERENCE: PASS")
    print("Cases:", len(cases))
    print("Output:", OUTPUT)
    print("Bytes:", OUTPUT.stat().st_size)


if __name__ == "__main__":
    main()
