import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]

NPZ_PATH = ROOT / "pwea/artifacts/deployment/pwea_model_v1.npz"
MANIFEST_PATH = ROOT / "pwea/artifacts/deployment/pwea_model_v1_manifest.json"

OUTPUT_DIR = ROOT / "pwea/native/models"
OUTPUT_PATH = OUTPUT_DIR / "pwea_model_v1.bin"
OUTPUT_MANIFEST = OUTPUT_DIR / "pwea_model_v1_native_manifest.json"

MAGIC = b"PWEAMDL1"
FORMAT_VERSION = 1

N_SAMPLES = 140
INPUT_DIM = 32768
OUTPUT_DIM = 6144
N_LAYER = 48
N_EXPERT = 128

# magic, version, samples, input_dim, output_dim,
# layers, experts, dtype_code, payload_bytes,
# payload_sha256, source_npz_sha256
HEADER_FORMAT = "<8sIIIIIIIQ32s32s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest):
    assert manifest["artifact_type"] == "pwea_deployment_model"
    assert manifest["schema_version"] == 1

    model = manifest["model"]

    assert model["algorithm"] == "dual_multi_output_ridge"
    assert model["feature_variant"] == "balanced_hybrid"
    assert model["alpha"] == 0.1
    assert model["feature_dimensions"] == INPUT_DIM
    assert model["output_dimensions"] == OUTPUT_DIM
    assert model["layers"] == N_LAYER
    assert model["experts_per_layer"] == N_EXPERT
    assert model["training_samples"] == N_SAMPLES

    assert manifest["arrays"]["X_train"]["shape"] == [
        N_SAMPLES, INPUT_DIM
    ]
    assert manifest["arrays"]["dual_coefficients"]["shape"] == [
        N_SAMPLES, OUTPUT_DIM
    ]

    assert manifest["arrays"]["X_train"]["dtype"] == "float64"
    assert manifest["arrays"]["dual_coefficients"]["dtype"] == "float64"


def main():
    if not NPZ_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Original model artifact or manifest missing")

    manifest = json.loads(MANIFEST_PATH.read_text())
    validate_manifest(manifest)

    source_sha = sha256_file(NPZ_PATH)

    if source_sha != manifest["model_file_sha256"]:
        raise RuntimeError("Original NPZ SHA-256 does not match manifest")

    with np.load(NPZ_PATH, allow_pickle=False) as archive:
        if set(archive.files) != {"X_train", "dual_coefficients"}:
            raise RuntimeError(f"Unexpected arrays: {archive.files}")

        X_train = archive["X_train"]
        coefficients = archive["dual_coefficients"]

        if X_train.shape != (N_SAMPLES, INPUT_DIM):
            raise RuntimeError(f"Invalid X_train shape: {X_train.shape}")

        if coefficients.shape != (N_SAMPLES, OUTPUT_DIM):
            raise RuntimeError(
                f"Invalid coefficient shape: {coefficients.shape}"
            )

        if X_train.dtype != np.float64 or coefficients.dtype != np.float64:
            raise RuntimeError("Both arrays must be float64")

        if not np.isfinite(X_train).all():
            raise RuntimeError("X_train contains non-finite values")

        if not np.isfinite(coefficients).all():
            raise RuntimeError("Coefficients contain non-finite values")

        # Explicit little-endian serialization, preserving float64 values.
        X_bytes = np.asarray(X_train, dtype="<f8", order="C").tobytes()
        C_bytes = np.asarray(coefficients, dtype="<f8", order="C").tobytes()

    payload = X_bytes + C_bytes
    payload_sha = hashlib.sha256(payload).digest()

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        FORMAT_VERSION,
        N_SAMPLES,
        INPUT_DIM,
        OUTPUT_DIM,
        N_LAYER,
        N_EXPERT,
        1,  # dtype code 1 = float64
        len(payload),
        payload_sha,
        bytes.fromhex(source_sha),
    )

    if len(header) != HEADER_SIZE:
        raise RuntimeError("Internal header size mismatch")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write atomically so an interrupted export cannot leave a partial model.
    fd, temporary_path = tempfile.mkstemp(
        prefix=".pwea_model_",
        suffix=".tmp",
        dir=OUTPUT_DIR,
    )

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(header)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary_path, OUTPUT_PATH)

    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise

    native_manifest = {
        "format": "PWEAMDL1",
        "format_version": FORMAT_VERSION,
        "header_bytes": HEADER_SIZE,
        "dtype": "float64-le",
        "arrays": {
            "X_train": {
                "shape": [N_SAMPLES, INPUT_DIM],
                "offset": HEADER_SIZE,
                "bytes": len(X_bytes),
            },
            "dual_coefficients": {
                "shape": [N_SAMPLES, OUTPUT_DIM],
                "offset": HEADER_SIZE + len(X_bytes),
                "bytes": len(C_bytes),
            },
        },
        "payload_bytes": len(payload),
        "payload_sha256": payload_sha.hex(),
        "source_npz_sha256": source_sha,
        "binary_file_sha256": sha256_file(OUTPUT_PATH),
    }

    OUTPUT_MANIFEST.write_text(
        json.dumps(native_manifest, indent=2) + "\n"
    )

    print("NATIVE MODEL EXPORT: PASS")
    print(f"Binary: {OUTPUT_PATH}")
    print(f"Manifest: {OUTPUT_MANIFEST}")
    print(f"Header bytes: {HEADER_SIZE}")
    print(f"Payload bytes: {len(payload)}")
    print(f"Total bytes: {OUTPUT_PATH.stat().st_size}")
    print(f"Payload SHA-256: {payload_sha.hex()}")
    print(f"Source NPZ SHA-256: {source_sha}")


if __name__ == "__main__":
    main()
