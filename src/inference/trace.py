"""Prompt rendering and reasoning-trace parsing (Section 9).

Produces the structured, numbered-step reasoning trace and extracts the
final numeric answer, normalized for comparison against gold.
"""
from __future__ import annotations

import re

FINAL_ANSWER_RE = re.compile(r"####\s*([\-0-9,\.]+)")
STEP_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(.*)$")


def render_prompt(template: str, question: str) -> str:
    return template.format(question=question)


def _normalize_number(s: str) -> str | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").rstrip(".")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def parse_final_answer(generated_text: str) -> str | None:
    matches = list(FINAL_ANSWER_RE.finditer(generated_text))
    if not matches:
        return None
    return _normalize_number(matches[-1].group(1))


def answers_match(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return _normalize_number(a) == _normalize_number(b)


def split_into_steps(generated_text: str) -> list[dict]:
    """Splits a generated trace into {"index": int, "text": str} steps using
    the "N. ..." numbering the prompt requests. Falls back to line splitting
    if the model did not number its steps (recorded via a low parse-quality
    flag downstream in script 02)."""
    body = generated_text.split("####")[0]
    lines = [l for l in body.splitlines() if l.strip()]
    steps = []
    for line in lines:
        m = STEP_RE.match(line)
        if m:
            steps.append({"index": int(m.group(1)), "text": m.group(2).strip()})
    if steps:
        return steps
    # Fallback: treat every non-empty line as one step, 1-indexed.
    return [{"index": i + 1, "text": l.strip()} for i, l in enumerate(lines)]


def trace_is_well_structured(generated_text: str) -> bool:
    steps = split_into_steps(generated_text)
    has_final = FINAL_ANSWER_RE.search(generated_text) is not None
    return has_final and len(steps) >= 1
