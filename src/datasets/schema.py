"""Canonical dataset schema and config loading utilities.

Implements the record format from Section 42 of the design doc. Every
clean/fail pair (and every stable correct->correct control pair, which reuses
the same schema with pair_type == "stable_control") is stored as one
PairRecord, serialized to JSONL under data/pairs/<benchmark>/.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path: str | Path) -> dict:
    with open(_abs(path), "r") as f:
        return yaml.safe_load(f)


def load_experiment_config(path: str | Path = "configs/experiment.yaml") -> dict:
    return load_yaml(path)


def load_benchmark_config(name_or_path: str) -> dict:
    """Accepts either 'gsm8k' / 'svamp' or an explicit yaml path."""
    if name_or_path in ("gsm8k", "svamp"):
        path = f"configs/{name_or_path}.yaml"
    else:
        path = name_or_path
    cfg = load_yaml(path)
    cfg["_config_path"] = str(_abs(path))
    return cfg


def load_model_config(name_or_path: str) -> dict:
    if not name_or_path.endswith(".yaml"):
        path = f"configs/models/{name_or_path}.yaml"
    else:
        path = name_or_path
    cfg = load_yaml(path)
    cfg["_config_path"] = str(_abs(path))
    return cfg


@dataclass
class PairRecord:
    pair_id: str
    benchmark: str
    base_id: str

    clean_question: str
    fail_question: str

    pair_type: str  # "induced_failure" | "rescued_failure" | "stable_control"
    perturbation: str
    token_edit_distance: int

    gold_answer: str
    clean_answer: Optional[str] = None
    fail_answer: Optional[str] = None

    clean_trace: Optional[str] = None
    fail_trace: Optional[str] = None

    first_observable_error_step: Optional[int] = None
    wrong_span: Optional[str] = None
    corrected_span: Optional[str] = None

    semantic_description: Optional[str] = None
    semantic_coarse: Optional[str] = None
    semantic_mid: Optional[str] = None
    semantic_fine: Optional[str] = None

    semantic_embedding: Optional[list[float]] = None
    judge_confidence: Optional[float] = None

    changed_clean_tokens: list[int] = field(default_factory=list)
    changed_fail_tokens: list[int] = field(default_factory=list)

    raw_causal_profile: Optional[list[float]] = None
    stable_pair_baseline: Optional[list[float]] = None
    failure_excess_signature: Optional[list[float]] = None

    causal_profile_atp: Optional[list[float]] = None

    # free-form bag for covariates, model name, generation metadata, etc.
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PairRecord":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in known}
        return cls(**clean)

    def content_hash(self) -> str:
        payload = json.dumps(
            {"clean_question": self.clean_question, "fail_question": self.fail_question,
             "pair_type": self.pair_type, "perturbation": self.perturbation},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_jsonl(records: list[PairRecord] | list[dict], path: str | Path) -> None:
    path = _abs(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            d = r.to_dict() if isinstance(r, PairRecord) else r
            f.write(json.dumps(d) + "\n")


def read_jsonl(path: str | Path, as_records: bool = True):
    path = _abs(path)
    out = []
    if not path.exists():
        return out
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(PairRecord.from_dict(d) if as_records else d)
    return out


def append_jsonl(record: PairRecord | dict, path: str | Path) -> None:
    path = _abs(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = record.to_dict() if isinstance(record, PairRecord) else record
    with open(path, "a") as f:
        f.write(json.dumps(d) + "\n")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(_abs(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
