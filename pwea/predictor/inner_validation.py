#!/usr/bin/env python3
"""Deterministic category-stratified, session-grouped inner validation."""

from collections import defaultdict


PROMPTS_PER_SESSION = 7
DEFAULT_SPLITS = 5


def build_inner_folds(samples, categories, n_splits=DEFAULT_SPLITS):
    """
    Build category-stratified folds without splitting sessions.

    Each fold holds out the same session ordinal from every category.
    Samples must contain verified prompt text only.

    Returns dictionaries containing sample indices and session IDs.
    """
    categories = tuple(categories)

    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories must be non-empty and unique")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    if not samples:
        raise ValueError("samples must not be empty")

    category_sessions = defaultdict(lambda: defaultdict(list))
    seen_samples = set()

    for index, sample in enumerate(samples):
        category = sample.get("category")
        session_id = sample.get("session_id")
        prompt_id = sample.get("prompt_id")

        if category not in categories:
            raise ValueError(
                f"sample contains an unexpected category: {category}"
            )

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("every sample must have a non-empty session_id")

        if sample.get("prompt_text_status") != "verified":
            raise ValueError(
                f"unverified prompt text in session {session_id}"
            )

        if not isinstance(prompt_id, int) or not 1 <= prompt_id <= PROMPTS_PER_SESSION:
            raise ValueError(
                f"invalid prompt_id in session {session_id}: {prompt_id}"
            )

        sample_id = sample.get("sample_id")
        if sample_id in seen_samples:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_samples.add(sample_id)

        category_sessions[category][session_id].append(
            (index, prompt_id)
        )

    expected_sessions = n_splits

    for category in categories:
        sessions = category_sessions[category]

        if len(sessions) != expected_sessions:
            raise ValueError(
                f"{category}: expected {expected_sessions} sessions, "
                f"found {len(sessions)}"
            )

        for session_id, records in sessions.items():
            prompt_ids = [prompt_id for _, prompt_id in records]

            if len(records) != PROMPTS_PER_SESSION:
                raise ValueError(
                    f"{session_id}: expected exactly "
                    f"{PROMPTS_PER_SESSION} prompts"
                )

            if sorted(prompt_ids) != list(
                range(1, PROMPTS_PER_SESSION + 1)
            ):
                raise ValueError(
                    f"{session_id}: prompt IDs must be exactly 1..7"
                )

    # Sorting makes fold assignment deterministic and auditable.
    ordered_sessions = {
        category: sorted(category_sessions[category])
        for category in categories
    }

    folds = []

    for fold_index in range(n_splits):
        validation_sessions = {
            category: ordered_sessions[category][fold_index]
            for category in categories
        }

        validation_ids = set(validation_sessions.values())

        train_indices = []
        validation_indices = []

        train_session_ids = []
        validation_session_ids = []

        for category in categories:
            for session_id in ordered_sessions[category]:
                records = category_sessions[category][session_id]
                target = (
                    validation_indices
                    if session_id == validation_sessions[category]
                    else train_indices
                )

                target.extend(index for index, _ in records)

                if session_id == validation_sessions[category]:
                    validation_session_ids.append(session_id)
                else:
                    train_session_ids.append(session_id)

        if set(train_indices) & set(validation_indices):
            raise RuntimeError("training and validation indices overlap")

        folds.append({
            "fold": fold_index + 1,
            "train_indices": sorted(train_indices),
            "validation_indices": sorted(validation_indices),
            "train_session_ids": sorted(train_session_ids),
            "validation_session_ids": sorted(validation_session_ids),
            "training_categories": list(categories),
            "validation_categories": list(categories),
        })

    return folds
