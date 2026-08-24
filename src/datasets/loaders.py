"""Loading and normalizing raw benchmark data (Section 5, 6).

Both loaders return a list of dicts: {"base_id": str, "question": str,
"gold_answer": str, "gold_solution": str}. Raw HF datasets are cached to
data/raw/<benchmark>/ so the pipeline is reproducible offline after the
first run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import REPO_ROOT, _abs

GSM8K_ANSWER_RE = re.compile(r"####\s*([\-0-9,\.]+)")


def _normalize_number(s: str) -> str:
    return s.strip().replace(",", "")


def load_gsm8k(cfg: dict) -> list[dict]:
    raw_path = _abs(cfg["raw_path"])
    cache_file = raw_path / f"{cfg['split']}.jsonl"
    if cache_file.exists():
        return [json.loads(l) for l in open(cache_file)]

    from datasets import load_dataset

    ds = load_dataset(cfg["hf_dataset"], cfg.get("hf_config", "main"), split=cfg["split"])
    records = []
    for i, ex in enumerate(ds):
        m = GSM8K_ANSWER_RE.search(ex["answer"])
        gold = _normalize_number(m.group(1)) if m else None
        records.append({
            "base_id": f"gsm8k_{i:05d}",
            "question": ex["question"],
            "gold_answer": gold,
            "gold_solution": ex["answer"],
        })
    raw_path.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


def load_svamp(cfg: dict) -> list[dict]:
    raw_path = _abs(cfg["raw_path"])
    cache_file = raw_path / f"{cfg['split']}.jsonl"
    if cache_file.exists():
        return [json.loads(l) for l in open(cache_file)]

    from datasets import load_dataset

    ds = load_dataset(cfg["hf_dataset"], split=cfg["split"])
    records = []
    for i, ex in enumerate(ds):
        body = ex.get("Body") or ex.get("body") or ""
        question = ex.get("Question") or ex.get("question") or ""
        full_q = f"{body} {question}".strip()
        ans = ex.get("Answer", ex.get("answer"))
        gold = _normalize_number(str(ans))
        if gold.endswith(".0"):
            gold = gold[:-2]
        records.append({
            "base_id": f"svamp_{i:05d}",
            "question": full_q,
            "gold_answer": gold,
            "gold_solution": ex.get("Equation", ""),
        })
    raw_path.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records
