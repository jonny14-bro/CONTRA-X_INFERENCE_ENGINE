#!/usr/bin/env python3
"""Deterministic word and character n-gram text features."""

import hashlib
import re

import numpy as np


FEATURE_DIMENSIONS = 16384
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _hash_feature(feature, dimensions=FEATURE_DIMENSIONS):
    """Return a deterministic feature index and signed hashing value."""
    digest = hashlib.sha256(feature.encode("utf-8")).digest()

    index = int.from_bytes(digest[:8], "big") % dimensions
    sign = 1.0 if digest[8] & 1 else -1.0

    return index, sign


def _word_features(text):
    tokens = _TOKEN_PATTERN.findall(text.lower())

    for size in range(WORD_NGRAM_RANGE[0], WORD_NGRAM_RANGE[1] + 1):
        for start in range(len(tokens) - size + 1):
            yield "word:" + " ".join(tokens[start:start + size])


def _character_features(text):
    normalized = " ".join(text.lower().split())

    for size in range(CHAR_NGRAM_RANGE[0], CHAR_NGRAM_RANGE[1] + 1):
        for start in range(len(normalized) - size + 1):
            yield "char:" + normalized[start:start + size]


def transform_text(text, dimensions=FEATURE_DIMENSIONS):
    """Convert one prompt into a deterministic L2-normalized feature vector."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt text must be a non-empty string")

    if dimensions <= 0:
        raise ValueError("dimensions must be positive")

    vector = np.zeros(dimensions, dtype=np.float64)

    for feature in (*_word_features(text), *_character_features(text)):
        index, sign = _hash_feature(feature, dimensions)
        vector[index] += sign

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector /= norm

    return vector


def transform_texts(texts, dimensions=FEATURE_DIMENSIONS):
    """Transform multiple prompts into a dense feature matrix."""
    if not texts:
        raise ValueError("texts must not be empty")

    return np.vstack([
        transform_text(text, dimensions)
        for text in texts
    ])


FEATURE_VARIANTS = (
    "word_only",
    "character_only",
    "balanced_hybrid",
)


def _normalized_feature_block(features, dimensions):
    """Hash one feature stream and independently L2-normalize its block."""
    vector = np.zeros(dimensions, dtype=np.float64)

    for feature in features:
        index, sign = _hash_feature(feature, dimensions)
        vector[index] += sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm

    return vector


def transform_text_variant(
    text,
    variant,
    dimensions=FEATURE_DIMENSIONS,
):
    """
    Transform text using word-only, character-only, or balanced features.

    balanced_hybrid concatenates independently normalized word and
    character blocks, each scaled by 1/sqrt(2).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt text must be a non-empty string")

    if variant not in FEATURE_VARIANTS:
        raise ValueError(
            f"variant must be one of {FEATURE_VARIANTS}"
        )

    if not isinstance(dimensions, int) or dimensions <= 0:
        raise ValueError("dimensions must be a positive integer")

    if variant == "word_only":
        return _normalized_feature_block(
            _word_features(text), dimensions
        )

    if variant == "character_only":
        return _normalized_feature_block(
            _character_features(text), dimensions
        )

    word = _normalized_feature_block(
        _word_features(text), dimensions
    )
    character = _normalized_feature_block(
        _character_features(text), dimensions
    )

    scale = 1.0 / np.sqrt(2.0)
    return np.concatenate((word * scale, character * scale))


def transform_texts_variant(
    texts,
    variant,
    dimensions=FEATURE_DIMENSIONS,
):
    """Transform multiple prompts using one explicit feature variant."""
    if not texts:
        raise ValueError("texts must not be empty")

    return np.vstack([
        transform_text_variant(text, variant, dimensions)
        for text in texts
    ])
