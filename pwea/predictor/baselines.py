#!/usr/bin/env python3
"""Training-only empirical baselines for per-prompt expert activation shares."""

import numpy as np


def build_activation_baselines(samples, indices, n_layer=48, n_expert=128):
    """
    Build equal-prompt and pooled-frequency baselines from training samples.

    Equal-prompt:
        Sum each prompt's normalized activation shares and divide by the
        total number of training prompts. Missing experts contribute zero.

    Pooled-frequency:
        Sum activation counts and divide by total events per layer.

    Invalid prompt layers contribute zero to equal-prompt sums and no events
    to pooled-frequency counts. A layer with no training events remains zero.
    """
    if not samples:
        raise ValueError("samples must not be empty")

    indices = list(indices)
    if not indices:
        raise ValueError("training indices must not be empty")

    if len(set(indices)) != len(indices):
        raise ValueError("training indices must be unique")

    if n_layer <= 0 or n_expert <= 0:
        raise ValueError("model dimensions must be positive")

    mean_prompt_sum = np.zeros((n_layer, n_expert), dtype=np.float64)
    pooled_counts = np.zeros((n_layer, n_expert), dtype=np.float64)
    pooled_totals = np.zeros(n_layer, dtype=np.float64)

    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("training indices must be integers")
        if not 0 <= index < len(samples):
            raise IndexError(f"training index out of range: {index}")

        sample = samples[index]
        target = sample["target"]

        shares = np.asarray(target["activation_share"], dtype=np.float64)
        valid = np.asarray(target["valid_layer_mask"], dtype=bool)
        event_counts = np.asarray(
            target["layer_event_counts"], dtype=np.float64
        )

        if shares.shape != (n_layer, n_expert):
            raise ValueError(f"{sample.get('sample_id')}: invalid target shape")
        if valid.shape != (n_layer,) or event_counts.shape != (n_layer,):
            raise ValueError(f"{sample.get('sample_id')}: invalid layer metadata")
        if not np.all(np.isfinite(shares)) or not np.all(np.isfinite(event_counts)):
            raise ValueError("targets and event counts must be finite")
        if np.any(shares < 0) or np.any(event_counts < 0):
            raise ValueError("targets and event counts must be nonnegative")

        for layer in range(n_layer):
            if not valid[layer]:
                if event_counts[layer] != 0:
                    raise ValueError("invalid layer must have zero events")
                continue

            if event_counts[layer] <= 0:
                raise ValueError("valid layer must have positive event count")

            if not np.isclose(shares[layer].sum(), 1.0, atol=1e-10):
                raise ValueError("valid target layer must sum to one")

            mean_prompt_sum[layer] += shares[layer]
            pooled_counts[layer] += shares[layer] * event_counts[layer]
            pooled_totals[layer] += event_counts[layer]

    # Match build_profile.py: divide by all training prompts, including
    # prompts whose corresponding layer is invalid.
    mean_prompt = mean_prompt_sum / len(indices)

    pooled_frequency = np.zeros_like(pooled_counts)
    valid_layers = pooled_totals > 0
    pooled_frequency[valid_layers] = (
        pooled_counts[valid_layers] / pooled_totals[valid_layers, None]
    )

    return {
        "mean_prompt_activation_share": mean_prompt,
        "aggregate_activation_share": pooled_frequency,
        "training_prompt_count": len(indices),
        "layer_event_totals": pooled_totals,
    }
