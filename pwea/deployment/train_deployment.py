#!/usr/bin/env python3
"""Build the frozen PWEA v1 deployment artifact."""

import hashlib
import json
import os
import platform
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pwea.predictor.ridge import MultiOutputRidge
from pwea.predictor.text_features import transform_texts_variant


ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "pwea/artifacts/prediction/expert_prediction_samples_20261115.jsonl"
SUMMARY = ROOT / "pwea/artifacts/prediction/five_fold_model_selection_summary_20261116.json"
AMENDMENT = ROOT / "pwea/artifacts/prediction/expert_prediction_protocol_amendment_v1_20261116.json"

OUTPUT_DIR = ROOT / "pwea/artifacts/deployment"
MODEL_PATH = OUTPUT_DIR / "pwea_model_v1.npz"
MANIFEST_PATH = OUTPUT_DIR / "pwea_model_v1_manifest.json"

EXPECTED_DATASET_SHA256 = (
    "8a516ca4ff7618be88ca41309943b0c4ff33deab89469ec521bd72afc2ac968a"
)
EXPECTED_AMENDMENT_SHA256 = (
    "c152f7f0d99baff36a2063dbb90cb5cfbe97c408ed9d7a22631f0b259ed98a3e"
)
EXPECTED_SUMMARY_SHA256 = (
    "3e6a09486b1d4501d56c96d00e5840a1a3bd1e8bfb4729d694a325a8938eaa6d"
)

FEATURE_VARIANT = "balanced_hybrid"
ALPHA = 0.1
N_LAYER = 48
N_EXPERT = 128
INPUT_DIM = 32768
OUTPUT_DIM = N_LAYER * N_EXPERT

EXPECTED_CATEGORIES = {
    "debugging": 35,
    "code-explanation": 35,
    "code-transformation": 35,
    "algorithmic-problem-solving": 35,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_and_validate():
    checksums = {
        "dataset": (DATASET, EXPECTED_DATASET_SHA256),
        "amendment": (AMENDMENT, EXPECTED_AMENDMENT_SHA256),
        "selection_summary": (SUMMARY, EXPECTED_SUMMARY_SHA256),
    }

    for label, (path, expected) in checksums.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"{label} checksum mismatch: expected {expected}, got {actual}"
            )

    selection = json.loads(SUMMARY.read_text(encoding="utf-8"))

    if selection.get("status") != "all_five_inner_selection_reports_verified":
        raise RuntimeError("selection summary is not verified")
    if selection.get("outer_test_evaluated") is not False:
        raise RuntimeError("unexpected outer-test evaluation state")

    deployment_fold = selection["folds"]["code-generation"]
    if deployment_fold["selected_feature_variant"] != FEATURE_VARIANT:
        raise RuntimeError("frozen feature variant does not match selection")
    if float(deployment_fold["selected_alpha"]) != ALPHA:
        raise RuntimeError("frozen alpha does not match selection")
    if deployment_fold["verified_text_prompts"] != 140:
        raise RuntimeError("unexpected selected training prompt count")
    if deployment_fold["training_sessions"] != 20:
        raise RuntimeError("unexpected selected training session count")

    samples = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(samples) != 175:
        raise RuntimeError(f"expected 175 source records, got {len(samples)}")

    sample_ids = [sample["sample_id"] for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("duplicate sample IDs in dataset")

    eligible = [
        sample for sample in samples
        if sample.get("prompt_text_status") == "verified"
        and isinstance(sample.get("prompt_text"), str)
        and sample["prompt_text"].strip()
    ]

    if len(eligible) != 140:
        raise RuntimeError(f"expected 140 eligible records, got {len(eligible)}")

    category_counts = dict(Counter(s["category"] for s in eligible))
    if category_counts != EXPECTED_CATEGORIES:
        raise RuntimeError(f"unexpected training categories: {category_counts}")

    session_ids = {s["session_id"] for s in eligible}
    if len(session_ids) != 20:
        raise RuntimeError(f"expected 20 sessions, got {len(session_ids)}")

    censored_count = sum(bool(s.get("generation_censored")) for s in eligible)
    if censored_count != 20:
        raise RuntimeError(
            f"expected 20 generation-censored eligible records, got {censored_count}"
        )

    targets = []
    for sample in eligible:
        matrix = np.asarray(
            sample["target"]["activation_share"], dtype=np.float64
        )
        if matrix.shape != (N_LAYER, N_EXPERT):
            raise RuntimeError(
                f"{sample['sample_id']}: invalid target shape {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"{sample['sample_id']}: non-finite target")

        # Preserve existing evaluation semantics, including zero-filled invalid layers.
        row_sums = matrix.sum(axis=1)
        valid_sums = np.isclose(row_sums, 1.0, atol=1e-8)
        zero_sums = np.isclose(row_sums, 0.0, atol=1e-8)
        if not np.all(valid_sums | zero_sums):
            raise RuntimeError(
                f"{sample['sample_id']}: unexpected activation-share row sums"
            )

        targets.append(matrix.reshape(OUTPUT_DIM))

    texts = [sample["prompt_text"] for sample in eligible]
    X = transform_texts_variant(texts, FEATURE_VARIANT)
    Y = np.asarray(targets, dtype=np.float64)

    if X.shape != (140, INPUT_DIM):
        raise RuntimeError(f"unexpected feature matrix shape: {X.shape}")
    if Y.shape != (140, OUTPUT_DIM):
        raise RuntimeError(f"unexpected target matrix shape: {Y.shape}")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise RuntimeError("training matrices contain non-finite values")

    return eligible, X, Y, category_counts, len(session_ids), censored_count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eligible, X, Y, category_counts, session_count, censored_count = (
        load_and_validate()
    )

    print("Training frozen PWEA deployment model...")
    print("Samples:", len(eligible))
    print("Sessions:", session_count)
    print("Categories:", category_counts)
    print("Generation-censored included:", censored_count)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Feature variant:", FEATURE_VARIANT)
    print("Alpha:", ALPHA)

    model = MultiOutputRidge(alpha=ALPHA).fit(X, Y)

    if model.X_train.shape != (140, INPUT_DIM):
        raise RuntimeError("unexpected fitted X_train shape")
    if model.dual_coefficients.shape != (140, OUTPUT_DIM):
        raise RuntimeError("unexpected dual coefficient shape")

    fd, temporary = tempfile.mkstemp(
        prefix=MODEL_PATH.name + ".", suffix=".tmp", dir=OUTPUT_DIR
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                X_train=model.X_train,
                dual_coefficients=model.dual_coefficients,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, MODEL_PATH)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    artifact_hash = sha256(MODEL_PATH)

    manifest = {
        "artifact_type": "pwea_deployment_model",
        "schema_version": 1,
        "model_version": "pwea_model_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "algorithm": "dual_multi_output_ridge",
            "alpha": ALPHA,
            "feature_variant": FEATURE_VARIANT,
            "feature_dimensions": INPUT_DIM,
            "output_dimensions": OUTPUT_DIM,
            "layers": N_LAYER,
            "experts_per_layer": N_EXPERT,
            "training_samples": len(eligible),
            "training_sessions": session_count,
            "generation_censored_samples_included": censored_count,
            "training_categories": category_counts,
            "dtype": "float64",
        },
        "arrays": {
            "X_train": {
                "shape": list(model.X_train.shape),
                "dtype": str(model.X_train.dtype),
            },
            "dual_coefficients": {
                "shape": list(model.dual_coefficients.shape),
                "dtype": str(model.dual_coefficients.dtype),
            },
        },
        "provenance": {
            "dataset": DATASET.relative_to(ROOT).as_posix(),
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "selection_summary": SUMMARY.relative_to(ROOT).as_posix(),
            "selection_summary_sha256": EXPECTED_SUMMARY_SHA256,
            "protocol_amendment": AMENDMENT.relative_to(ROOT).as_posix(),
            "protocol_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
            "selection_scope": "code-generation outer holdout inner-training cohort",
            "selection_rationale": (
                "The code-generation heldout fold trains on all 140 available "
                "verified-text records from the four other categories."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "serialization": "numpy_npz_no_pickle",
        },
        "model_file": MODEL_PATH.name,
        "model_file_sha256": artifact_hash,
    }

    atomic_json(MANIFEST_PATH, manifest)

    print("\nDEPLOYMENT ARTIFACT CREATED")
    print("Model:", MODEL_PATH)
    print("Manifest:", MANIFEST_PATH)
    print("Model SHA-256:", artifact_hash)
    print("X_train:", model.X_train.shape)
    print("Dual coefficients:", model.dual_coefficients.shape)


if __name__ == "__main__":
    main()
