#!/usr/bin/env python
"""Master orchestrator - runs scripts 00-15 (and optionally 16) in the order
described in README.md / Section 44, in one command.

This changes nothing about the science: it's a thin subprocess driver over
the exact same scripts you'd run by hand. It still reads the Gate B
(script 06) and Gate C (script 10) verdicts from their JSON reports and
stops there by default - pass --force to continue past a failed gate anyway
(the failure is still logged loudly). SVAMP (script 16 + its own 01-04 data
generation) only runs if you pass --with-svamp, since it's meant to happen
only after you've looked at and finalized the GSM8K results.

Human annotation (Section 15/19's optional strengthening of Gate B) is
interactive and can't be scripted - this orchestrator sticks to the default
two-model-judge annotation flow (judge_a / judge_b) end to end. If you want
a human pass, run the `05_annotate_failures.py --annotator-type human`
commands from README.md yourself, add the annotator id to
`--annotators` here (or edit configs/experiment.yaml), and resume with
--start-stage 06_semantic_reliability.

Usage:
    # Full GSM8K discovery run, from the model pilot through intervention transfer
    python scripts/run_pipeline.py --model qwen2.5-3b-instruct

    # Skip the pilot (you already know which model you're using)
    python scripts/run_pipeline.py --model qwen2.5-3b-instruct --skip-pilot

    # Also run the frozen SVAMP replication at the end
    python scripts/run_pipeline.py --model qwen2.5-3b-instruct --with-svamp

    # Resume after fixing something by hand, starting from a given stage
    python scripts/run_pipeline.py --model qwen2.5-3b-instruct --start-stage 08_exact_patching

    # List stages without running anything
    python scripts/run_pipeline.py --model qwen2.5-3b-instruct --list-stages

    # Don't stop on a failed gate (still prints a loud warning)
    python scripts/run_pipeline.py --model qwen2.5-3b-instruct --force
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_pipeline")


class Stage:
    def __init__(self, name: str, script: str, args_fn, gate_report: str | None = None, gate_key: str | None = None):
        self.name = name
        self.script = script
        self.args_fn = args_fn          # cfg -> list[str]
        self.gate_report = gate_report  # relative path under results/ to check after running
        self.gate_key = gate_key        # boolean field name inside that JSON report


def _measured_flip_rate(model_name: str) -> float | None:
    """Reads results/reports/pilot_report.json (if it exists, i.e. the pilot
    stage already ran) and returns the measured flip rate for `model_name`,
    or None if unavailable."""
    report_path = REPO_ROOT / "results" / "reports" / "pilot_report.json"
    if not report_path.exists():
        return None
    try:
        with open(report_path) as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    for r in report.get("reports", []):
        if r.get("model") == model_name and r.get("flip_rate", 0) > 0:
            return r["flip_rate"]
    return None


def _estimate_base_problem_limit(cfg: argparse.Namespace, target_n: int) -> int:
    """01_make_variants.py / 02_generate_traces.py aren't scoped by
    --target-n - they process however many base problems they're given, in
    full, before 03 ever gets a chance to stop early once it has enough
    pairs. So --target-n alone does NOT bound the (expensive) generation
    stage; the number of base problems fed into 01/02 does. This estimates
    that number from the flip rate: if only flip_rate of perturbed problems
    actually flip correct->incorrect, you need roughly target_n / flip_rate
    base problems to end up with target_n usable pairs. Prefers the REAL
    flip rate measured by the pilot stage (cheap, ~30 examples) over the
    --assumed-flip-rate guess, since this number directly controls how long
    the run takes.
    """
    if cfg.limit:
        return cfg.limit
    flip_rate = _measured_flip_rate(cfg.model)
    if flip_rate is not None:
        log.info("Using pilot-measured flip_rate=%.3f for %s to size --limit (run the pilot instead of "
                  "--skip-pilot to get this real number rather than the --assumed-flip-rate guess).",
                  flip_rate, cfg.model)
    else:
        flip_rate = cfg.assumed_flip_rate
        log.warning("No pilot flip-rate measurement available (pilot was skipped or hasn't run yet) - "
                    "sizing --limit from --assumed-flip-rate=%.3f instead. This is a guess; if it's too "
                    "low the generation stage will run far longer than expected.", flip_rate)
    # +30% safety margin: the pilot's own flip rate is itself a noisy
    # estimate from a small sample.
    return max(target_n, math.ceil(target_n / max(flip_rate, 0.02) * 1.3))


def build_stages(cfg: argparse.Namespace) -> list[Stage]:
    # NOTE: stage arg-builders below read `c.model` (the cfg passed into
    # run_stage at *execution* time), not a name closed over here at
    # build-list time. That's what lets _apply_pilot_selection() (called
    # right after the 00_pilot stage runs, or before the loop if resuming
    # past it) retarget every downstream stage at the pilot's actual
    # selected_model instead of silently keeping whatever --model was typed.
    ja, jb = cfg.judge_a_id, cfg.judge_b_id
    ja_cfg, jb_cfg = cfg.judge_a, cfg.judge_b

    stages = [
        Stage("00_pilot", "00_model_pilot.py", lambda c: []),

        Stage("01_variants_gsm8k", "01_make_variants.py",
              lambda c: ["--benchmark", "gsm8k", "--variants", str(c.n_variants),
                         "--limit", str(_estimate_base_problem_limit(c, c.target_n))]),
        Stage("02_traces_gsm8k", "02_generate_traces.py",
              lambda c: ["--benchmark", "gsm8k", "--model", c.model, "--resume"]),
        Stage("03_pairs_gsm8k", "03_select_pairs.py",
              lambda c: ["--benchmark", "gsm8k", "--model", c.model, "--prefer-token-aligned",
                         "--target-n", str(c.target_n)]),
        Stage("04_stable_gsm8k", "04_make_stable_controls.py",
              lambda c: ["--benchmark", "gsm8k", "--model", c.model]),

        Stage("05_annotate_judge_a", "05_annotate_failures.py",
              lambda c: ["--benchmark", "gsm8k", "--annotator-type", "model",
                         "--annotator-id", ja, "--model", ja_cfg]),
        Stage("05_annotate_judge_b", "05_annotate_failures.py",
              lambda c: ["--benchmark", "gsm8k", "--annotator-type", "model",
                         "--annotator-id", jb, "--model", jb_cfg]),
        Stage("06_semantic_reliability", "06_semantic_reliability.py",
              lambda c: ["--benchmark", "gsm8k", "--annotators", ja, jb],
              gate_report="reports/semantic_reliability.json", gate_key="gate_b_passed"),

        Stage("07_freeze_taxonomy", "07_freeze_taxonomy.py",
              lambda c: ["--benchmark", "gsm8k", "--primary-annotator", ja]),

        Stage("08_exact_patching", "08_compute_exact_patching.py",
              lambda c: ["--benchmark", "gsm8k", "--model", c.model]),
        Stage("09_attribution_patching", "09_compute_attribution_patching.py",
              lambda c: ["--benchmark", "gsm8k", "--model", c.model]),
        Stage("10_mechanistic_reliability", "10_mechanistic_reliability.py",
              lambda c: ["--benchmark", "gsm8k"],
              gate_report="reports/mechanistic_reliability.json", gate_key="gate_c_passed"),

        Stage("11_main_correspondence", "11_main_correspondence_test.py",
              lambda c: ["--benchmark", "gsm8k", "--permutations", str(c.permutations)]),
        Stage("12_prediction_test", "12_prediction_test.py",
              lambda c: ["--benchmark", "gsm8k"]),
        Stage("13_resolution_analysis", "13_resolution_analysis.py",
              lambda c: ["--benchmark", "gsm8k"]),
        Stage("14_split_merge", "14_split_merge_analysis.py",
              lambda c: ["--benchmark", "gsm8k"]),
        Stage("15_intervention_transfer", "15_intervention_transfer.py",
              lambda c: ["--benchmark", "gsm8k", "--model", c.model, "--n-pairs", str(c.n_transfer_pairs),
                         "--top-k", str(c.top_k)]),
    ]

    if cfg.with_svamp:
        stages += [
            Stage("01_variants_svamp", "01_make_variants.py",
                  lambda c: ["--benchmark", "svamp", "--variants", str(c.n_variants),
                             "--limit", str(_estimate_base_problem_limit(c, c.target_n_svamp))]),
            Stage("02_traces_svamp", "02_generate_traces.py",
                  lambda c: ["--benchmark", "svamp", "--model", c.model, "--resume"]),
            Stage("03_pairs_svamp", "03_select_pairs.py",
                  lambda c: ["--benchmark", "svamp", "--model", c.model, "--prefer-token-aligned",
                             "--target-n", str(c.target_n_svamp)]),
            Stage("04_stable_svamp", "04_make_stable_controls.py",
                  lambda c: ["--benchmark", "svamp", "--model", c.model]),
            Stage("16_replicate_svamp", "16_replicate_svamp.py",
                  lambda c: ["--model", c.model, "--judge-model", ja_cfg]),
        ]

    if cfg.skip_pilot:
        stages = [s for s in stages if s.name != "00_pilot"]

    return stages


def _reset_results() -> None:
    """Deletes everything under results/{reports,figures,tables,logs} (but
    keeps the directories and their .gitkeep, so the tree stays intact for
    git). Does NOT touch data/ (raw downloads, variants, pairs, annotations,
    taxonomy, mechanisms) - those are separate, expensive-to-regenerate
    pipeline artifacts, not run reports. Use this to clear out stale/broken
    reports (e.g. a pilot_report.json written before a bug fix) so a fresh
    run starts from a clean slate."""
    removed = 0
    for sub in ("reports", "figures", "tables", "logs"):
        d = REPO_ROOT / "results" / sub
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.name == ".gitkeep":
                continue
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
            removed += 1
    log.info("--reset-results: removed %d item(s) under results/{reports,figures,tables,logs}.", removed)


def _apply_pilot_selection(cfg: argparse.Namespace) -> None:
    """Overrides cfg.model with the Phase-0 pilot's measurement-based
    selected_model (results/reports/pilot_report.json), so every downstream
    stage actually uses the model the pilot picked instead of silently
    keeping whatever --model the caller happened to type. Called right after
    the 00_pilot stage runs, or before the stage loop starts if resuming
    past it (--start-stage) from a pilot that already ran in a prior
    invocation. Skipped entirely when --skip-pilot is passed - that's the
    explicit "I already know my model" escape hatch."""
    report_path = REPO_ROOT / "results" / "reports" / "pilot_report.json"
    if not report_path.exists():
        raise RuntimeError(
            "No pilot_report.json found and --skip-pilot was not passed - can't resolve which model the "
            "pilot selected. Run the pipeline from 00_pilot (the default), or pass --skip-pilot if you're "
            "deliberately choosing --model yourself."
        )
    with open(report_path) as f:
        report = json.load(f)
    selected = report.get("selected_model")
    if not selected:
        msg = ("Pilot ran but selected_model is null in pilot_report.json - no candidate in "
               "configs/experiment.yaml:pilot.models cleared the measurement gates.")
        if cfg.force:
            log.warning(msg + " Continuing with --model=%s anyway (--force).", cfg.model)
            return
        raise RuntimeError(msg + " Re-run with --force to proceed with --model anyway, or add/adjust "
                                  "candidates in configs/experiment.yaml:pilot.models.")
    if selected != cfg.model:
        log.info("Pilot selected '%s' (you passed --model %s) - using the pilot's pick for every "
                  "downstream stage.", selected, cfg.model)
    cfg.model = selected


def _check_gate(report_path: Path, gate_key: str) -> bool | None:
    if not report_path.exists():
        log.warning("Expected gate report %s not found - treating gate as unknown/failed.", report_path)
        return None
    with open(report_path) as f:
        report = json.load(f)
    return report.get(gate_key)


def run_stage(stage: Stage, cfg: argparse.Namespace) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / stage.script)] + stage.args_fn(cfg)
    log.info("=" * 70)
    log.info("STAGE %s :: %s", stage.name, " ".join(cmd[1:]))
    log.info("=" * 70)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Stage {stage.name} exited with code {result.returncode}")

    if stage.gate_report is not None:
        passed = _check_gate(REPO_ROOT / "results" / stage.gate_report, stage.gate_key)
        if passed is True:
            log.info("Stage %s: gate '%s' PASSED.", stage.name, stage.gate_key)
        else:
            msg = f"Stage {stage.name}: gate '{stage.gate_key}' FAILED (or unknown) - see {stage.gate_report}."
            if cfg.force:
                log.warning(msg + " Continuing anyway (--force).")
            else:
                raise RuntimeError(msg + " Re-run with --force to continue past this, or fix the underlying "
                                          "issue (e.g. add human annotation passes, adjust thresholds in "
                                          "configs/experiment.yaml) and resume with --start-stage.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="target model config name, e.g. qwen2.5-3b-instruct")
    ap.add_argument("--judge-a", default="configs/models/mistral-7b-instruct-v0.3.yaml")
    ap.add_argument("--judge-b", default="configs/models/phi-3.5-mini-instruct.yaml")
    ap.add_argument("--judge-a-id", default="judge_a")
    ap.add_argument("--judge-b-id", default="judge_b")
    ap.add_argument("--n-variants", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                     help="cap the number of base problems fed into variant generation (01/02) - the expensive "
                          "stage. Default: auto-estimated from target-n and the measured (or assumed) flip rate, "
                          "since --target-n alone does not bound how many problems 01/02 process.")
    ap.add_argument("--assumed-flip-rate", type=float, default=0.25,
                     help="used to size --limit ONLY when the pilot hasn't measured a real flip rate for --model "
                          "(i.e. --skip-pilot was passed, or the pilot report is missing)")
    ap.add_argument("--target-n", type=int, default=200)
    ap.add_argument("--target-n-svamp", type=int, default=150)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-transfer-pairs", type=int, default=150)
    ap.add_argument("--skip-pilot", action="store_true", help="skip the Phase-0 model pilot")
    ap.add_argument("--with-svamp", action="store_true", help="also run the frozen SVAMP replication")
    ap.add_argument("--start-stage", default=None, help="resume from this stage name (see --list-stages)")
    ap.add_argument("--stop-stage", default=None, help="stop after this stage name (inclusive)")
    ap.add_argument("--force", action="store_true", help="continue past a failed reliability gate")
    ap.add_argument("--list-stages", action="store_true", help="print the stage list and exit")
    ap.add_argument("--reset-results", action="store_true",
                     help="delete everything under results/{reports,figures,tables,logs} before running "
                          "(keeps the directories/.gitkeep). Does not touch data/ - use this to clear out "
                          "stale reports (e.g. a pilot_report.json from a run that failed before a bug fix), "
                          "not to redo data generation.")
    args = ap.parse_args()

    if args.reset_results:
        _reset_results()

    stages = build_stages(args)

    if args.list_stages:
        for s in stages:
            gate = f"  [gate: {s.gate_key}]" if s.gate_report else ""
            print(f"  {s.name}{gate}")
        return

    names = [s.name for s in stages]
    start_idx = names.index(args.start_stage) if args.start_stage else 0
    stop_idx = names.index(args.stop_stage) + 1 if args.stop_stage else len(stages)
    if args.start_stage and args.start_stage not in names:
        raise SystemExit(f"Unknown --start-stage '{args.start_stage}'. Run --list-stages to see valid names.")
    if args.stop_stage and args.stop_stage not in names:
        raise SystemExit(f"Unknown --stop-stage '{args.stop_stage}'. Run --list-stages to see valid names.")

    run_names = names[start_idx:stop_idx]
    if not args.skip_pilot and "00_pilot" not in run_names:
        # Pilot must have already run in a previous invocation (we're
        # resuming past it via --start-stage) - resolve its selection from
        # disk before anything downstream reads cfg.model.
        _apply_pilot_selection(args)

    for stage in stages[start_idx:stop_idx]:
        run_stage(stage, args)
        if stage.name == "00_pilot":
            _apply_pilot_selection(args)

    log.info("Pipeline complete. Reports are under results/reports/, figures under results/figures/.")


if __name__ == "__main__":
    main()
