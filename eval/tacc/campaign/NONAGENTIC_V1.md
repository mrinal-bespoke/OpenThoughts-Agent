# Non-agentic auto-eval — versioned suites

Minimal wrapper around the proven `hpc/vllm/smoke_grug_vllm_tacc.sbatch`
(Marin-vLLM + Evalchemy/lm-eval). It evaluates only an **explicit allowlist**
of models and is dry-run by default.

## Profiles

### `midtrain9-v1-seed42` (default, compatibility)

The original nine-benchmark handoff at seed 42. It remains the default so the
in-flight full9 run and its artifact layout do not change.

| Task | Protocol |
|---|---|
| MATH500 | zero-shot, 8,192 output tokens |
| IFEval | zero-shot, 2,048 output tokens |
| GSM8K | 5-shot, 512 output tokens |
| MMLU | 5-shot loglikelihood |
| HellaSwag | 10-shot loglikelihood |
| HumanEvalPlus | zero-shot pass@1, unsafe-code opt-in |
| MBPPPlus | zero-shot pass@1, unsafe-code opt-in |
| GPQADiamond | zero-shot; evaluator's three repeats |
| AIME24 | zero-shot; evaluator's ten repeats |

The result identity is `midtrain9-v1-seed42`; older seed-1234 V1 artifacts are
preserved but do not satisfy this suite.

### `marin-policy-production-v1` (opt-in)

The production-runnable non-agentic set from Marin policy issue #7958:

| Capability | Benchmarks |
|---|---|
| Math/reasoning | MATH500, AIME24, GPQA Diamond, GSM8K |
| Code | HumanEvalPlus, MBPPPlus |
| Knowledge/NLP | MMLU, HellaSwag, ARC-Challenge, ARC-Easy, PIQA, WinoGrande, OpenBookQA, BoolQ, TruthfulQA-MC2, LAMBADA, TriviaQA, NQ-Open, DROP |

Every benchmark is invoked once externally at seed 42, making 19 task+seed
legs. Evalchemy performs AIME24's ten repetitions and GPQA Diamond's three
repetitions inside those single invocations; the wrapper does not multiply
them with external reruns. Each leg has an isolated path, for example
`math500/seed-42`, and resume requests only missing legs.

The following policy benchmarks are coverage-only and are **not launched** by
this profile: OlympiadBench, MMLU-Pro, and CruxEval (known issues), plus
FinanceBench, IFBench, and MRCR (in development). IFEval remains in the legacy
full9 profile; it is not treated as an alias for IFBench.

Not included: discovery, capacity management, retry policy, daemon mode, DB
registration, leaderboard wiring, or pass@k.

## Dry run (this is also the status command)

    cd $SCRATCH/OpenThoughts-Agent
    python eval/tacc/campaign/nonagentic_autoeval.py \
      --allowlist eval/tacc/campaign/nonagentic_allowlist.json \
      --results-base $SCRATCH/experiments/nonagentic \
      --state-file $SCRATCH/experiments/nonagentic/state.json

Prints one block per model: SKIP or SUBMIT, the reason, the result root, and
the exact sbatch command. Nothing is submitted and no state is written.

To inspect the policy profile, add:

    --suite marin-policy-production-v1

## Launch

Same command plus `--submit`. Only then does it shell out to sbatch and record
job ids in the state file.

## Status / recovery

There is no separate status command: re-run the dry run. It reads the result
artifacts on disk, so it always reports current truth:

* `SKIP ... all benchmarks have valid results` — done.
* `SKIP ... already submitted as job N` — in flight. Check `squeue -j N`.
* `SUBMIT ... needs AIME24 (already done: ...)` — resume; only the missing
  benchmark will be requested.

A benchmark counts as complete only when a `results_*.json` under its task
directory contains a **numeric metric**. Evalchemy exits 0 and writes
`{"results": {}}` when the engine dies, so file existence alone is not proof.
Malformed, empty and metric-less artifacts are all treated as incomplete and
will be rerun.

**Recovery after a failed job:** clear the stale marker so the target is
eligible again, then dry-run to confirm what it wants to do:

    python - <<'PY'
    import json, pathlib
    p = pathlib.Path("$SCRATCH/experiments/nonagentic/state.json")
    s = json.loads(p.read_text())
    s["targets"].pop("<slug>", None)
    p.write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")
    PY

Valid results already on disk are kept, so recovery reruns only what is missing.

## sbatch knobs this relies on

`RUN_DIR` (stable result root across retries), `NONAGENTIC_TASKS_SEMICOLON`
(semicolon list of tasks to run), `NONAGENTIC_SUITE`, and `EVAL_SEED`. The
semicolon transport avoids colliding with Slurm's comma-delimited `--export`
syntax. The sbatch still accepts the older comma-delimited `NONAGENTIC_TASKS`
variable for direct/manual launches. The Python wrapper pins the suite identity
and seed and requests a 47:59 wall time.
