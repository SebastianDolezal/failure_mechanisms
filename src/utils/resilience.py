"""Per-item fault tolerance for long-running batch loops (trace generation,
activation patching, intervention transfer, ...).

A single bad example - a malformed trace, misaligned token positions, a
transient CUDA OOM on one forward pass - should not lose the rest of an
hours-long run. `ErrorTracker` wraps one loop iteration at a time: on
failure it logs the item id, the error, and a short traceback to a JSONL
file and lets the loop move on; on success it's a no-op.

This is deliberately scoped to *within* a script. scripts/run_pipeline.py's
stage-to-stage transitions are NOT wrapped this way and should keep failing
hard - stages are dependent on each other's output (08 needs 07's frozen
taxonomy, 11 needs 08's signatures, ...), so "stage crashed, run the next
one anyway" would either hard-fail downstream or silently run against stale
leftover output, which is worse than just stopping. Likewise the Gate B /
Gate C checks in run_pipeline.py are an intentional halt, not an error to
be swallowed.
"""
from __future__ import annotations

import contextlib
import json
import logging
import time
import traceback
from pathlib import Path

log = logging.getLogger("resilience")


class ErrorTracker:
    def __init__(self, out_path: str | Path):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.n_attempted = 0
        self.n_failed = 0
        self._fh = open(self.out_path, "a")

    @contextlib.contextmanager
    def guard(self, item_id: str, context: str = ""):
        """Use as:  with tracker.guard(item_id, context="..."):  <risky work>

        Any exception raised inside the block is caught, logged, and
        swallowed so the enclosing loop continues to its next iteration.
        Whatever the block would have appended/written simply doesn't
        happen for that item - there is no partial/corrupt state to clean
        up as long as side effects (list.append, file writes) happen at the
        end of the guarded block rather than incrementally through it.
        """
        self.n_attempted += 1
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - intentionally broad: keep the batch alive
            self.n_failed += 1
            entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "context": context,
                "item_id": item_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=6),
            }
            self._fh.write(json.dumps(entry) + "\n")
            self._fh.flush()
            log.warning("[%s] skipping %s after %s: %s", context or "item", item_id, type(exc).__name__, exc)

    def summary(self) -> dict:
        return {
            "n_attempted": self.n_attempted,
            "n_failed": self.n_failed,
            "n_succeeded": self.n_attempted - self.n_failed,
            "error_log_path": str(self.out_path),
        }

    def close(self) -> dict:
        self._fh.close()
        s = self.summary()
        if s["n_failed"]:
            log.warning("%d/%d items failed and were skipped - see %s",
                        s["n_failed"], s["n_attempted"], s["error_log_path"])
        else:
            log.info("%d/%d items succeeded, 0 failures.", s["n_succeeded"], s["n_attempted"])
        return s
