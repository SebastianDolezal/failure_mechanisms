"""Exact clean -> failed activation patching (Section 22, 25, 27).

For every layer l, the clean-run resid_pre activation at the aligned changed
token positions is spliced into the failed run, and the local margin is
recomputed. Layer-level recovery:

    R_i,l = (m_i,l^patch - m_i^fail) / (|m_i^clean - m_i^fail| + eps)

Granularity starts at layer-level only (Section 27); `component` lets a
later, targeted subset of examples be decomposed into attention vs MLP
contributions without changing the primary pipeline.
"""
from __future__ import annotations

import numpy as np
import torch

from ..inference.model_wrapper import ModelWrapper
from .scoring import continuation_margin

EPS = 1e-6


def exact_layer_patching_profile(
    wrapper: ModelWrapper,
    clean_prefix: str,
    fail_prefix: str,
    correct_continuation: str,
    error_continuation: str,
    changed_clean_positions: list[int],
    changed_fail_positions: list[int],
) -> dict:
    """Returns {"R": np.ndarray[n_layers], "m_clean": float, "m_fail": float,
    "aligned": bool}. When position counts differ (imperfect alignment) the
    shorter list is used and `aligned` is set False so callers can route the
    example to the secondary robustness analysis (Section 25)."""
    n_layers = wrapper.n_layers
    aligned = len(changed_clean_positions) == len(changed_fail_positions) and len(changed_clean_positions) > 0
    k = min(len(changed_clean_positions), len(changed_fail_positions))
    if k == 0:
        return {"R": np.zeros(n_layers), "m_clean": float("nan"), "m_fail": float("nan"), "aligned": False}

    clean_pos = changed_clean_positions[:k]
    fail_pos = changed_fail_positions[:k]

    clean_acts = wrapper.capture_activations(clean_prefix)

    m_clean = continuation_margin(wrapper, clean_prefix, correct_continuation, error_continuation)
    m_fail = continuation_margin(wrapper, fail_prefix, correct_continuation, error_continuation)
    denom = abs(m_clean - m_fail) + EPS

    R = np.zeros(n_layers)
    for l in range(n_layers):
        hs = clean_acts.hidden_states[l]
        max_idx = hs.shape[0] - 1
        valid = [(cp, fp) for cp, fp in zip(clean_pos, fail_pos) if cp <= max_idx]
        if not valid:
            R[l] = 0.0
            continue
        replacement = torch.stack([hs[cp] for cp, _ in valid], dim=0)
        fpos = [fp for _, fp in valid]
        patch = {"layer_idx": l, "positions": fpos, "replacement": replacement}
        m_patch = continuation_margin(wrapper, fail_prefix, correct_continuation, error_continuation, patch=patch)
        R[l] = (m_patch - m_fail) / denom

    return {"R": R, "m_clean": m_clean, "m_fail": m_fail, "aligned": aligned}


def top_k_layers(signature, k: int, prefer_positive: bool = False) -> list[int]:
    """Returns the indices of the k most causally-significant layers.

    Default (prefer_positive=False): ranks by |signature| - "most significant
    in either direction," the original selection rule.

    prefer_positive=True: ranks by the signed value instead, biasing toward
    layers where patching *recovers* correct behavior (positive R/D) rather
    than layers that are merely extreme in magnitude. This matters because a
    large |D| layer is just as often strongly negative (patching there makes
    things worse) as strongly positive - selecting by magnitude alone does
    not preferentially surface "helpful" layers. Callers that specifically
    want intervention locations expected to help (e.g. the intervention-
    transfer "own top layers" upper bound, or the transfer source layers)
    should pass prefer_positive=True.
    """
    import numpy as np
    signature = np.asarray(signature)
    key = signature if prefer_positive else np.abs(signature)
    return np.argsort(-key)[:k].tolist()


def multi_layer_patch_recovery(
    wrapper: ModelWrapper,
    clean_prefix: str,
    fail_prefix: str,
    correct_continuation: str,
    error_continuation: str,
    changed_clean_positions: list[int],
    changed_fail_positions: list[int],
    layers: list[int],
) -> dict:
    """Patches ALL of `layers` simultaneously (clean -> fail, at the aligned
    changed positions) and reports the resulting recovery, R_{source->target}
    of Section 37. Used by script 15 for intervention-location transfer: the
    source failure supplies `layers`, this failure's own clean run supplies
    the replacement activations.
    """
    import contextlib

    n_layers = wrapper.n_layers
    k = min(len(changed_clean_positions), len(changed_fail_positions))
    if k == 0 or not layers:
        return {"R": 0.0, "m_clean": float("nan"), "m_fail": float("nan")}

    clean_pos = changed_clean_positions[:k]
    fail_pos = changed_fail_positions[:k]
    clean_acts = wrapper.capture_activations(clean_prefix)

    m_clean = continuation_margin(wrapper, clean_prefix, correct_continuation, error_continuation)
    m_fail = continuation_margin(wrapper, fail_prefix, correct_continuation, error_continuation)
    denom = abs(m_clean - m_fail) + EPS

    with contextlib.ExitStack() as stack:
        for l in layers:
            if l >= n_layers:
                continue
            hs = clean_acts.hidden_states[l]
            max_idx = hs.shape[0] - 1
            valid = [(cp, fp) for cp, fp in zip(clean_pos, fail_pos) if cp <= max_idx]
            if not valid:
                continue
            replacement = torch.stack([hs[cp] for cp, _ in valid], dim=0)
            fpos = [fp for _, fp in valid]
            stack.enter_context(wrapper.patch_layer(l, fpos, replacement))
        m_patch = continuation_margin(wrapper, fail_prefix, correct_continuation, error_continuation)

    R = (m_patch - m_fail) / denom
    return {"R": R, "m_clean": m_clean, "m_fail": m_fail}
