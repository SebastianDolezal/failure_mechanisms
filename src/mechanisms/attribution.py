"""Attribution Patching (Section 26): a linear, gradient-based approximation
to exact activation patching, used only to check the *reliability* of the
exact-patching instrument (RQ1b), never as the primary causal estimator.

    D_l^AtP ~= (A_clean_l - A_fail_l) . grad_{A_l} m_i

where m_i = lp_correct - lp_error is the SAME margin exact patching targets
(Section 20-21). Both terms of that margin are differentiated - see the note
in _differentiable_margin below for why that matters - computed with two
forward+backward passes on the failed run (one per continuation) plus one
forward pass on the clean run to obtain A_clean, vs. exact patching's O(L)
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
    """Returns (margin: float, captured_correct: list[Tensor|None],
    grads_correct: list[Tensor|None], grads_error: list[Tensor|None]).

    Exact patching's R (Section 22) measures the effect of an intervention on
    the FULL margin m_i = lp_correct - lp_error - patching one activation
    changes both the correct- and the error-continuation's log-probability,
    and R reflects both. The original attribution-patching implementation
    only differentiated lp_correct (lp_error was detached), so it was a
    linear approximation of a *different, incomplete* quantity - not just a
    noisier estimate of the same one. That mismatch is a first-order
    candidate for why exact and AtP profiles showed only mean_spearman=0.06
    agreement (Gate C).

    Fix: run two separate backward passes, one per continuation, and sum
    their gradients at each layer. This is valid because both forward passes
    share the identical `prefix` text - the prefix-position activations are
    numerically identical between the two passes (causal attention means a
    prefix token's representation cannot depend on what follows it), even
    though they come from two distinct computation graphs and so need two
    separate .backward() calls rather than one.
    """
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

    wrapper.model.zero_grad(set_to_none=True)
    lp_correct.backward()
    grads_correct = [c.grad.clone() if (c is not None and c.grad is not None) else None
                      for c in captured_correct]

    wrapper.model.zero_grad(set_to_none=True)
    (-lp_error).backward()
    grads_error = [c.grad.clone() if (c is not None and c.grad is not None) else None
                    for c in captured_error]

    margin = float((lp_correct - lp_error).detach())
    return margin, captured_correct, grads_correct, grads_error


def attribution_patching_profile(
    wrapper: ModelWrapper,
    clean_prefix: str,
    fail_prefix: str,
    correct_continuation: str,
    error_continuation: str,
    changed_clean_positions: list[int],
    changed_fail_positions: list[int],
) -> np.ndarray:
    """Two-point (trapezoidal) attribution patching.

    Plain single-point AtP evaluates the gradient of the margin only at the
    failed run's activations and assumes it stays constant all the way to the
    clean activation - a first-order Taylor expansion around one endpoint of
    the path exact patching actually interpolates. That assumption is exactly
    what breaks under the non-linearity "When Attribution Patching Lies"
    (2026) identifies as AtP's main error source, and it is a first-order
    candidate for Gate C's low agreement with exact patching (Section 3.5).

    Fix: also evaluate the same margin, and its gradient, at the clean run's
    own activations - the other natural endpoint of the path - and average
    the two gradients before multiplying by the clean-minus-fail difference.
    This is a coarse (two-sample) Riemann approximation to the Integrated
    Gradients path integral (Sundararajan et al., 2017) using the path's
    endpoints rather than a finer interpolation; it does not require the
    O(n_layers) patched forward passes a full multi-step version would, since
    both endpoints are natural (unpatched) forward passes and every layer's
    gradient is still recovered from a single backward pass per endpoint.
    """
    n_layers = wrapper.n_layers
    k = min(len(changed_clean_positions), len(changed_fail_positions))
    if k == 0:
        return np.zeros(n_layers)

    _, captured_fail, grads_correct_fail, grads_error_fail = _differentiable_margin(
        wrapper, fail_prefix, correct_continuation, error_continuation
    )
    _, captured_clean, grads_correct_clean, grads_error_clean = _differentiable_margin(
        wrapper, clean_prefix, correct_continuation, error_continuation
    )

    D = np.zeros(n_layers)
    fail_pos = changed_fail_positions[:k]
    clean_pos = changed_clean_positions[:k]
    for l in range(n_layers):
        # shape [1, seq_len, hidden] (retained-grad tensor is the layer's raw
        # input, batch dim included) - drop the batch dim before indexing.
        gc_f = grads_correct_fail[l][0] if grads_correct_fail[l] is not None else None
        ge_f = grads_error_fail[l][0] if grads_error_fail[l] is not None else None
        gc_c = grads_correct_clean[l][0] if grads_correct_clean[l] is not None else None
        ge_c = grads_error_clean[l][0] if grads_error_clean[l] is not None else None
        if gc_f is None and ge_f is None and gc_c is None and ge_c is None:
            continue
        max_idx_f = (gc_f.shape[0] if gc_f is not None else ge_f.shape[0]) - 1
        max_idx_c = (gc_c.shape[0] if gc_c is not None else ge_c.shape[0]) - 1
        contrib = 0.0
        for cp, fp in zip(clean_pos, fail_pos):
            if fp > max_idx_f or cp > max_idx_c:
                continue
            grad_at_fp = 0.0
            if gc_f is not None:
                grad_at_fp = grad_at_fp + gc_f[fp]
            if ge_f is not None:
                grad_at_fp = grad_at_fp + ge_f[fp]
            grad_at_cp = 0.0
            if gc_c is not None:
                grad_at_cp = grad_at_cp + gc_c[cp]
            if ge_c is not None:
                grad_at_cp = grad_at_cp + ge_c[cp]
            avg_grad = (grad_at_fp.float() + grad_at_cp.float()) / 2.0
            diff = (captured_clean[l][0, cp].detach().float() - captured_fail[l][0, fp].detach().float())
            contrib += float(diff @ avg_grad)
        D[l] = contrib
    return D
