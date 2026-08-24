"""Attribution Patching (Section 26): a linear, gradient-based approximation
to exact activation patching, used only to check the *reliability* of the
exact-patching instrument (RQ1b), never as the primary causal estimator.

    D_l^AtP ~= (A_clean_l - A_fail_l) . grad_{A_l} m_i

computed with a single forward+backward pass on the failed run (plus one
forward pass on the clean run to obtain A_clean), vs. exact patching's O(L)
forward passes.
"""
from __future__ import annotations

import numpy as np
import torch

from ..inference.model_wrapper import ModelWrapper


def _differentiable_margin(
    wrapper: ModelWrapper,
    prefix: str,
    correct_continuation: str,
    error_continuation: str,
):
    """Forward pass with grad enabled; returns (margin_scalar_tensor,
    list_of_resid_pre_tensors_with_grad, input_ids)."""
    model = wrapper.model
    tok = wrapper.tokenizer

    prefix_ids = tok(prefix, return_tensors="pt").input_ids.to(model.device)

    def seq_logprob(continuation: str):
        full = tok(prefix + continuation, return_tensors="pt").input_ids.to(model.device)
        cont_len = full.shape[1] - prefix_ids.shape[1]
        if cont_len <= 0:
            return None, None

        captured = [None] * wrapper.n_layers
        handles = []

        def make_hook(idx):
            def hook(module, inputs):
                t = inputs[0]
                t.retain_grad()
                captured[idx] = t
            return hook

        for i, layer in enumerate(wrapper._decoder_layers):
            handles.append(layer.register_forward_pre_hook(make_hook(i)))
        try:
            out = model(full)
        finally:
            for h in handles:
                h.remove()

        logits = out.logits[0, prefix_ids.shape[1] - 1: full.shape[1] - 1, :]
        targets = full[0, prefix_ids.shape[1]: full.shape[1]]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        token_lps = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return token_lps.mean(), captured

    lp_correct, captured_correct = seq_logprob(correct_continuation)
    lp_error, captured_error = seq_logprob(error_continuation)
    # Use the correct-continuation forward pass's activations as the
    # reference point for attribution (matches where m_i is most sensitive).
    margin = lp_correct - lp_error.detach()
    return margin, captured_correct, prefix_ids.shape[1]


def attribution_patching_profile(
    wrapper: ModelWrapper,
    clean_prefix: str,
    fail_prefix: str,
    correct_continuation: str,
    error_continuation: str,
    changed_clean_positions: list[int],
    changed_fail_positions: list[int],
) -> np.ndarray:
    n_layers = wrapper.n_layers
    k = min(len(changed_clean_positions), len(changed_fail_positions))
    if k == 0:
        return np.zeros(n_layers)

    clean_acts = wrapper.capture_activations(clean_prefix)

    margin, captured, _ = _differentiable_margin(
        wrapper, fail_prefix, correct_continuation, error_continuation
    )
    wrapper.model.zero_grad(set_to_none=True)
    margin.backward()

    D = np.zeros(n_layers)
    fail_pos = changed_fail_positions[:k]
    clean_pos = changed_clean_positions[:k]
    for l in range(n_layers):
        act = captured[l]
        if act is None or act.grad is None:
            continue
        grad = act.grad[0]  # [seq_len, hidden]
        max_idx = grad.shape[0] - 1
        contrib = 0.0
        for cp, fp in zip(clean_pos, fail_pos):
            if fp > max_idx or cp >= clean_acts.hidden_states[l].shape[0]:
                continue
            diff = (clean_acts.hidden_states[l][cp].float() - act[0, fp].detach().float())
            contrib += float(diff @ grad[fp].float())
        D[l] = contrib
    return D
