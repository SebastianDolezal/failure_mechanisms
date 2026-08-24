# Failure Mechanisms

**Do semantic failure modes track causal mechanisms in language models?**

This repo implements the full experimental pipeline for testing whether
human-readable descriptions of LLM failures (`T_i`) correspond to
reproducible, failure-specific causal deviations in the model's computation
(`D_i`), beyond what is already explained by the underlying problem and
surface perturbation. See the design doc (conversation history) for the full
statistical/methodological specification (RQ1-RQ4, H1-H7, Sections 1-51).

Nothing here is a toy: every script does real model inference, real
activation patching, and real (permutation / bootstrap / regression)
inference. Nothing here has been run end-to-end in this environment (no GPU,
no downloaded model weights) — what has been verified is that every module
imports cleanly, every script's CLI is well-formed, and the entire
non-model-dependent logic (perturbations, pair selection, matching,
permutation/bootstrap statistics, regression, taxonomy clustering/freezing)
runs correctly on synthetic data (see "Validation" below).

## 1. Setup

```bash
cd failure_mechanisms
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You need a machine that can hold a 2-4B parameter model in bf16 (a single
24GB+ GPU is comfortable; a recent Mac with enough unified memory works via
`device_map="auto"`/CPU fallback, just slower). No API keys are required —
everything runs against locally-downloaded Hugging Face checkpoints
(`Qwen/Qwen2.5-3B-Instruct`, `microsoft/Phi-3.5-mini-instruct` by default;
edit `configs/models/*.yaml` to add others).

## 2. Repository map

```
configs/            model + benchmark + global experiment YAML configs
data/                raw/ variants/ generations/ pairs/ annotations/ taxonomy/ mechanisms/
src/                 the actual library (importable, unit-testable)
  datasets/          canonical PairRecord schema + GSM8K/SVAMP loaders
  perturb/           name_swap, number_format, lexical_conservative + variant-bank builder
  inference/         deterministic model wrapper, prompt rendering, trace parsing
  taxonomy/          description embedding, HDBSCAN clustering, freeze/map
  mechanisms/         continuation-margin scoring, exact + attribution patching,
                      stable-pair baseline subtraction (D_i), similarity metrics
  matching/          clean/fail pair selection, stable-pair construction, covariates
  statistics/        permutation test, bootstrap, reliability (kappa/alpha/F1), regression
  plots/             Figures 1-7
scripts/             00-16 (one per pipeline stage) + run_pipeline.py (runs them all in order)
results/             figures/ tables/ reports/ logs/  (all script outputs land here)
```

Every intermediate artifact is a `.jsonl` file of `PairRecord`s (see
`src/datasets/schema.py`) or a `.json` report — nothing is pickled, so every
stage is inspectable and diffable.

## 3. How to run it

Two ways to do this: one command via the orchestrator, or the phases by hand
if you want to inspect/intervene between steps. Both run the exact same
scripts — the orchestrator is just a subprocess driver over them, it doesn't
change what gets checked.

### 3a. One command

```bash
python scripts/run_pipeline.py --model qwen2.5-3b-instruct
```

Runs the model pilot, all of GSM8K data generation, annotation (the default
two-model-judge flow from Section 4 of the previous discussion),
Gate B, taxonomy freezing, exact + attribution patching, Gate C, and all of
scripts 11-15, in order. It reads the actual Gate B / Gate C JSON reports
after those stages and **stops if a gate fails** — pass `--force` to
continue anyway (still logs the failure loudly). Useful flags:

```bash
python scripts/run_pipeline.py --model qwen2.5-3b-instruct --list-stages          # see stage names
python scripts/run_pipeline.py --model qwen2.5-3b-instruct --skip-pilot           # you already picked a model
python scripts/run_pipeline.py --model qwen2.5-3b-instruct --with-svamp           # also run the frozen SVAMP replication at the end
python scripts/run_pipeline.py --model qwen2.5-3b-instruct --start-stage 08_exact_patching   # resume after fixing something by hand
python scripts/run_pipeline.py --model qwen2.5-3b-instruct --force               # don't stop on a failed gate
```

If you want a human annotation pass (see Phase 2 below), that step is
interactive and can't be scripted — run it yourself between
`05_annotate_judge_b` and `06_semantic_reliability`, add the annotator id to
`configs/experiment.yaml:annotation.annotators`, then resume the
orchestrator with `--start-stage 06_semantic_reliability`.

### 3b. Phase by phase

This mirrors Section 44 of the design doc and is useful if you want to
inspect outputs, re-annotate, or add a human pass between stages. **Do not
skip the gates** — they exist so a null result later on is interpretable
rather than just "the measurement was too noisy."

### Phase 0 — model pilot (selects the primary model on measurement grounds only)

```bash
python scripts/00_model_pilot.py --config configs/experiment.yaml
```

Runs ~30 GSM8K clean/fail flips through each candidate model in
`configs/experiment.yaml:pilot.models`, checks flip availability, trace
structure, non-degenerate patching profiles, and exact-vs-attribution-patching
reliability, and writes `results/reports/pilot_report.json` naming the
selected model. **Do not** look at semantic correspondence results when
choosing here — the script doesn't compute any.

Take the `selected_model` name from that report and use it as `--model` in
every script below (this README uses `qwen2.5-3b-instruct` as the example).

### Phase 1 — data generation (per benchmark)

```bash
python scripts/01_make_variants.py --benchmark gsm8k --variants 8
python scripts/02_generate_traces.py --benchmark gsm8k --model qwen2.5-3b-instruct
python scripts/03_select_pairs.py --benchmark gsm8k --model qwen2.5-3b-instruct \
    --prefer-token-aligned --target-n 200
python scripts/04_make_stable_controls.py --benchmark gsm8k --model qwen2.5-3b-instruct
```

Produces `data/pairs/gsm8k/pairs_primary.jsonl` (clean/fail pairs, target
N≈200 unique problems), `pairs_secondary_unaligned.jsonl` (robustness-only,
imperfect token alignment), and `pairs_stable.jsonl` (correct→correct
controls for the same problems).

### Phase 2 — semantic measurement

Annotators are never the target model itself (Section 15: "the target model
should not be the only judge of its own failures"). The default here is two
independent judge models from different families — neither a pilot/target
candidate — so cross-judge agreement reflects category stability rather
than one model's phrasing habits, without requiring hand-annotation:

```bash
python scripts/05_annotate_failures.py --benchmark gsm8k \
    --annotator-type model --annotator-id judge_a \
    --model configs/models/smollm2-1.7b-instruct.yaml

python scripts/05_annotate_failures.py --benchmark gsm8k \
    --annotator-type model --annotator-id judge_b \
    --model configs/models/tinyllama-1.1b-chat.yaml

python scripts/06_semantic_reliability.py --benchmark gsm8k \
    --annotators judge_a judge_b
```

Optionally strengthen this into a human-verified reliability check by adding
one or two human passes (uncomment the human entries in
`configs/experiment.yaml:annotation.annotators` first):

```bash
python scripts/05_annotate_failures.py --benchmark gsm8k \
    --annotator-type human --annotator-id annotator_1          # full set, interactive CLI, resumable

python scripts/05_annotate_failures.py --benchmark gsm8k \
    --annotator-type human --annotator-id annotator_2 --subset-frac 0.3   # stratified subset

python scripts/06_semantic_reliability.py --benchmark gsm8k \
    --annotators judge_a judge_b annotator_1 annotator_2
```

`06` writes `results/reports/semantic_reliability.json` with Cohen's kappa,
Krippendorff's alpha, first-error-step agreement, and raw-description
embedding stability, and prints **Gate B PASSED/FAILED**. Only continue if it
passes (threshold in `configs/experiment.yaml:gates.gate_b_min_kappa`). Note
that with only model judges, Gate B demonstrates *cross-model* semantic
reliability, not human-verified reliability — a limitation worth stating
explicitly if you skip the human passes.

```bash
python scripts/07_freeze_taxonomy.py --benchmark gsm8k --primary-annotator judge_a
```

(`--primary-annotator` accepts any annotator id already in
`data/annotations/gsm8k/` — swap in a human id here if you added one.)

Freezes `data/taxonomy/taxonomy_v1.json` (+ `.sha256`) and writes
`data/pairs/gsm8k/pairs_annotated.jsonl` — the taxonomy is immutable for
every analysis from here on.

### Phase 3 — mechanistic measurement

```bash
python scripts/08_compute_exact_patching.py --benchmark gsm8k --model qwen2.5-3b-instruct \
    --site resid_pre --metric error_boundary_margin

python scripts/09_compute_attribution_patching.py --benchmark gsm8k --model qwen2.5-3b-instruct

python scripts/10_mechanistic_reliability.py --benchmark gsm8k
```

`08` computes exact layer-level clean→fail patching profiles `R_i` for both
failure and stable pairs, then subtracts a covariate-matched stable-pair
baseline to get the failure-excess signature `D_i` (Section 23) — this is
the primary mechanistic object everything downstream uses. `09` recomputes a
gradient-based Attribution Patching estimate on a ~55-item stratified
subset; `10` compares the two and prints **Gate C PASSED/FAILED**
(`results/reports/mechanistic_reliability.json`).

### Phase 4 — main analyses

```bash
python scripts/11_main_correspondence_test.py --benchmark gsm8k --permutations 10000
python scripts/12_prediction_test.py --benchmark gsm8k --category-level mid
python scripts/13_resolution_analysis.py --benchmark gsm8k
python scripts/14_split_merge_analysis.py --benchmark gsm8k --category-level mid
```

- `11` is the primary confirmatory test (H2/H3): continuous
  `S_ij^sem -> C_ij^mech` regression (Figure 3) + the matched
  same-vs-different-category `Delta` statistic with a stratified block
  permutation test and a problem-level bootstrap CI (Figure 4).
- `12` tests H4: does adding `D_i` to nonmechanistic covariates improve
  held-out prediction of semantic type?
- `13` tests H5: sweeps taxonomy resolution (coarse/mid/fine) and plots the
  mechanistic-coherence curve (Figure 5).
- `14` tests H6: clusters `D_i` independently of any semantic label and
  reports splits/merges against the frozen taxonomy (Figure 6, Sankey HTML).

### Phase 5 — functional validation

```bash
python scripts/15_intervention_transfer.py --benchmark gsm8k --model qwen2.5-3b-instruct \
    --n-pairs 150 --top-k 5
```

Tests H7: does mechanistic similarity `C_ij` predict whether an intervention
location found on failure A transfers to failure B (Figure 7,
`functional_validity_model` regression)?

### Phase 6 — frozen replication (SVAMP)

Only after every GSM8K analysis above is finalized:

```bash
python scripts/01_make_variants.py --benchmark svamp --variants 8
python scripts/02_generate_traces.py --benchmark svamp --model qwen2.5-3b-instruct
python scripts/03_select_pairs.py --benchmark svamp --model qwen2.5-3b-instruct \
    --prefer-token-aligned --target-n 150
python scripts/04_make_stable_controls.py --benchmark svamp --model qwen2.5-3b-instruct

python scripts/16_replicate_svamp.py --model qwen2.5-3b-instruct \
    --judge-model configs/models/smollm2-1.7b-instruct.yaml
```

`--model` is the target model under study (must match whichever model
produced the SVAMP generations above); `--judge-model` is the independent
annotator, kept separate here too so replication doesn't quietly reintroduce
self-judging. `16` re-annotates SVAMP failures with the same frozen prompt, maps
descriptions onto the **existing** GSM8K taxonomy (nearest-centroid, with
"novel/unmapped" for anything that doesn't fit — no re-clustering), recomputes
`D_i` with the same patching pipeline, and reruns the same continuous +
categorical correspondence statistics. Nothing about the taxonomy, matching
logic, or statistical tests is retuned here.

## 4. What lands where

| Output | Produced by |
|---|---|
| `results/reports/pilot_report.json` | 00 |
| `data/pairs/<bench>/pairs_primary.jsonl`, `pairs_stable.jsonl` | 03, 04 |
| `data/taxonomy/taxonomy_v1.json` (+`.sha256`) | 07 |
| `data/pairs/<bench>/pairs_annotated.jsonl` (has `D_i`, taxonomy labels) | 07, 08 |
| `results/reports/mechanistic_reliability.json` | 10 |
| `results/figures/fig3_*`, `fig4_*`, `results/reports/main_correspondence_*.json` | 11 |
| `results/reports/prediction_test_*.json` | 12 |
| `results/figures/fig5_resolution_curve_*.png` | 13 |
| `results/figures/fig6_splits_merges_*.html`, `results/reports/split_merge_*.json` | 14 |
| `results/figures/fig7_*`, `results/tables/intervention_transfer_*.csv` | 15 |
| `results/reports/replication_svamp.json` | 16 |

## 5. Reliability gates (Section 41) — read before trusting any RQ2 result

`configs/experiment.yaml:gates` holds the thresholds. A null result from
script 11 is only scientifically meaningful if Gate A (N≥120), Gate B
(annotator agreement — with the default two-model-judge setup this measures
*cross-model* reliability, not human-verified reliability; add the human
annotator passes described in Phase 2 to strengthen this), Gate C (exact-vs-AtP
agreement exceeds cross-failure agreement), Gate D (excess signal beyond the
stable-pair baseline — implicit in using `D_i` rather than raw `R_i`
everywhere), and Gate E (taxonomy isn't trivially predictable from covariates
alone — check via `12`'s baseline model) all pass. If any gate fails, treat
the correspondence result as
inconclusive, not as evidence against the hypothesis.

## 6. Validation performed in this environment

- `python3 -m py_compile` over every file in `src/` and `scripts/`: clean.
- Every script's `--help` (module-level imports) executes cleanly.
- A synthetic end-to-end smoke test exercised (with fabricated data, no
  model calls): perturbation generation, clean/fail + stable pair selection,
  covariate-matched candidate pools, `Delta`/permutation/bootstrap
  statistics, Cohen's kappa / Krippendorff's alpha / macro-F1, the
  continuous-correspondence and incremental-prediction regressions,
  HDBSCAN taxonomy clustering + freeze + nearest-centroid mapping, and all
  seven figure-generation functions.
- Model-dependent code paths (generation, activation patching, attribution
  patching) follow standard HF `transformers` hook patterns but have **not**
  been exercised against real weights here — run Phase 0 first and inspect
  `pilot_report.json` before trusting anything downstream.
