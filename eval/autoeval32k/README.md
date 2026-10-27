# Explicit 32K automatic evaluation

This opt-in path accepts a published dense Qwen3 checkpoint and a reviewed frozen
task inventory. It does not use the existing listener, resume old runs, register
scores, upload traces, build snapshots, or discover models automatically.

Status: locally tested implementation, **not live-qualified on TACC yet**. TACC
access is renewed; a synthetic smoke is still required. Concurrency is
locked to 1. Jupiter, automatic checkpoint watching, MoE/multi-node serving,
pass@8, and BPB are not implemented here.

## Protocol

32768 total tokens: at most 24576 serialized input/history tokens and 8192 output
tokens. The adapter uses the serving tokenizer to check the entire request.
No history summarization or terminal-output truncation is allowed. Overflow is
an explicit failure. Temperature is 0.6; thinking mode and tokenizer template
hash are explicit. One trial per task, no trial/generation/sandbox-create retry.
Harbor's underlying command/file-transfer retry behavior is unchanged.

The strict transport retains the Terminus-2 parser/loop, but it bypasses hidden
LLM retries. It is a new evaluation profile, not a claim of byte-for-byte parity
with the old campaign. The adapter rejects Harbor versions/source fingerprints
other than the inspected implementation. Changing its supported pin requires
review and requalification.

## Prepare inputs

The suite specification is a JSON object with `version: 1`, `suite`, `tasks`,
`runtime`, and `budgets`. It deliberately has no model yet.

- Each task supplies `id`, absolute `path`, `sha256` from
  `eval.autoeval32k.protocol.tree_hash(path)`, existing snapshot `snapshot` name,
  and immutable `snapshot_id`. Task IDs must match their staged directory names.
  Snapshot compatibility, task setup and grading patch correctness require a
  reviewed inventory; the pipeline does not guess or create environments.
- Runtime supplies `harbor_commit`, `vllm_version`, and `agent: "terminus-2"`.
  The only currently inspected Harbor commit is
  `84c67493d28fdd56d6d1476da1d95485d4e41b37`. This is not a statement that the
  same runtime is already installed on your TACC account.
- Budgets supply positive `startup_seconds`, `request_seconds`, `agent_seconds`,
  `verifier_seconds`, and `environment_seconds`. Slurm allocation time is a
  separate hard cap. An allocation must cover model download/startup, tasks,
  verification and cleanup. Long downloads can exhaust it; there is no retry.
  The startup limit is per GPU-probe/serving-readiness phase, not a global
  download deadline. Agent setup is explicitly capped by `environment_seconds`;
  Harbor's outer environment setup/healthchecks also use the frozen task's
  timeout settings. Slurm bounds the whole execution, including these phases.

Freeze checkpoint metadata into a fresh private cache (no weights or GPU work):

```sh
python -m eval.autoeval32k.intake \
  --checkpoint Qwen/Qwen3-8B --revision EXACT_40_HEX_COMMIT \
  --thinking false --spec /absolute/suite.json \
  --cache-dir /absolute/new-intake-cache --output /absolute/new-manifest.json
```

Deployment settings supply `cluster: "tacc"`, your `account` and `partition`,
absolute `python`, `serve_python`, `repo_root`, `harbor_root`, `run_root`,
`state_root`, `walltime_seconds`, and `concurrency: 1`. Optionally supply
`checkpoint_cache` for read-only reuse of existing Hugging Face cache bytes.
Every selected weight/tokenizer file is checked against immutable-revision
upstream metadata before private references are staged. Corruption or missing
files fails closed. No fallback download occurs when this option is set.
Optionally supply
`credential_file` pointing to an existing private env file. Only
`DAYTONA_API_KEY`/`DAYTONA_DATA_API_KEY` assignments are read; the file is never
executed, copied, or printed. Do not put credential values in JSON or argv.
Use one authoritative state root for this pipeline, not a new ledger per run.

## Plan, then submit

Run from the deployed checkout on the TACC login host:

```sh
python -m eval.autoeval32k --manifest /absolute/new-manifest.json \
  --settings /absolute/tacc.json
```

Without `--submit`, this only prints the plan and creates no files or jobs. After
live qualification and task approval, the same command with `--submit` freezes
source/config, checks all your jobs against the cap of 20 nonheld jobs, persists
an exclusive intent and submits once. Held jobs are untouched; pending failed
dependencies count toward the cap. Queue-visibility lag cannot free reservations.

Submission uncertainty stops the pipeline. Do not delete the intent, clear a
writer lock or change the state root to retry. Inspect scheduler evidence, then
use `queue.reconcile_existing` with exactly one matching run/job record. Missing
queue entries alone are not terminal evidence.

## Execution and evidence

Each fresh run has an immutable control bundle and a separate new output folder.
The compute runner downloads the exact checkpoint into a private cache or
verifies the configured read-only cache. Preparation runs in serving Python
under a startup deadline, reusing its tokenizer dependencies. It checks the template,
runtime, BF16 GPU operation and generated text, stages verified task copies, then
executes Harbor using only the supplied snapshots.

`audit.json` distinguishes planned, numeric, zero, unscored and missing tasks.
`mean_numeric` is explicitly conditional on numeric records;
`successes_over_planned` retains the full denominator without imputing nulls.
Verifier rewards and controlled raw configuration fields must agree. Scored
timeouts remain scores. Raw result hashes/paths remain attached to outcomes.

`cleanup-audit.json` and per-trial creation/deletion evidence report scoped
environment cleanup. SIGTERM triggers best-effort cleanup, but a hard kill or
lost connection can still leave unresolved cleanup. Such runs are not healthy
and require exact-ID reconciliation; never retry the evaluation to clean it up.
Idle expiry is a backstop, not proof of deletion. No success claim may rely on
Slurm COMPLETED or a serving HTTP response alone.

## Validation sequence

1. TACC synthetic single-trial smoke: generated text, exact tokenizer count,
   Daytona command, agent artifact, verifier, and cleanup.
2. Small separately approved real-task pilot with frozen snapshots and no replay
   of existing scored tasks.
3. Measure concurrency 1 -> 2 -> 4 before enabling higher admission: productive
   trials/hour, request latency, timeout/overflow rate, GPU usage and cleanup.
4. Add Jupiter with queue-aware proxy lifetime and repeat all compute-side gates.

The old SWE and BPB campaign monitors stay paused throughout this work.
