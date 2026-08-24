"""Local causal target: the correct-vs-error continuation margin
(Section 20-21).

m_i = ell(c_i^+ | P_i) - ell(c_i^- | P_i)

where P_i is the failed problem + reasoning prefix up to (not including) the
first externally observable error, c_i^+ is a minimal correct continuation,
and c_i^- is the observed erroneous continuation. `ell` is a length-
normalized sequence log-probability (Section 21) so continuation length
cannot contaminate the metric.
"""
from __future__ import annotations

from ..inference.model_wrapper import ModelWrapper


def continuation_margin(
    wrapper: ModelWrapper,
    prefix: str,
    correct_continuation: str,
    error_continuation: str,
    patch: dict | None = None,
) -> float:
    lp_correct = wrapper.sequence_logprob(prefix, correct_continuation, patch=patch)
    lp_error = wrapper.sequence_logprob(prefix, error_continuation, patch=patch)
    return lp_correct - lp_error


def build_prefix(problem_prompt: str, trace_steps: list[dict], error_step_index: int) -> str:
    """P_i = failed problem + reasoning prefix strictly before the first
    externally observable error step (Section 20)."""
    prefix_steps = [s for s in trace_steps if s["index"] < error_step_index]
    body = "\n".join(f"{s['index']}. {s['text']}" for s in prefix_steps)
    return problem_prompt + ("\n" + body if body else "")


def question_token_offset(reasoning_prompt: str, tokenizer) -> int:
    """changed_clean_tokens / changed_fail_tokens (Section 10-11, computed by
    src.matching.pairs._aligned_changed_positions) are token indices into the
    bare perturbed question text, tokenized in isolation. Every patching
    stage instead indexes into build_prefix(render_prompt(reasoning_prompt,
    question), ...)'s activations, which prepends this benchmark's
    reasoning-prompt preamble before the question. This returns that
    preamble's token length so callers can rebase the stored positions onto
    the full prefix before using them to index its activations."""
    preamble = reasoning_prompt.split("{question}")[0]
    return len(tokenizer.encode(preamble, add_special_tokens=False))


def offset_positions(positions: list[int], offset: int) -> list[int]:
    return [p + offset for p in positions]
