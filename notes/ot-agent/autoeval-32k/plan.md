# 32K automatic evaluation

Owner request (2026-10-27): TACC first, improve TACC concurrency, then Jupiter.
Branch: `codex/autoeval-32k`, isolated managed worktree. Legacy launchers are unchanged.

## Contract

Explicit published checkpoint + immutable revision + frozen task manifest -> fresh
run -> raw outcomes and an audited coverage report. Initial support: dense Qwen3,
Terminus-2, BF16 TP1, 32768 total tokens, 24576 history/input, 8192 output,
temperature 0.6, no summarization, one attempt, no automatic retries.

No existing campaign is resumed. No score, frozen bundle, shared runtime, SSH
master, registry, credential source, or snapshot is changed. Snapshot-only task
execution must fail closed. Zero is a score; null is not zero. Unknown submission
outcome requires reconciliation, never another submission. Input must be explicit;
there is no broad model discovery or daemon activation in this change.

## Stages and gates

0. **Protocol and audit**: strict manifest, complete run identity, deterministic
   configuration and coverage accounting. CPU tests first; legacy files unchanged.
1. **TACC**: durable submission, explicit owner/runtime settings, pinned serving,
   exact task staging, snapshot-only Harbor execution, cleanup and reports. Gate:
   CPU integration tests, runtime schema validation, bounded synthetic live smoke.
2. **TACC concurrency**: bounded admission, measured promotion, no automatic
   retry. Gate: concurrency/visibility-lag tests; live 1 -> 2 -> 4 qualification
   before claiming higher throughput or enabling a higher level.
3. **Jupiter**: same protocol and audit, queue-aware external proxy admission.
   Gate: expired proxy rejected before submission and again on allocation; bounded
   compute serving + Daytona + agent + cleanup proof. Do not extrapolate TACC proof.

Stages are not complete until their live gates pass. Existing synthetic Jupiter
smokes did not validate end-to-end execution. No new benchmark pilot or snapshot
build is implied by the synthetic validation gate. A real pilot needs a frozen,
approved task/snapshot inventory that does not replay existing scored work.

## Reuse and exclusions

Borrow argv conventions from `eval/tacc/eval_harbor.sbatch`, not its entrypoint:
it registers jobs, resumes runs, retries and uploads. Use no existing listener
defaults. Harbor runtime API/schema must be checked against its exact source pin.
Vista is one GH200 per node, no GRES request; total nonheld user job cap is 20.
DependencyNeverSatisfied jobs count. Held jobs remain untouched.

## Status

Stage 0 CPU gate passed. Stage 1 implementation/testing in progress. TACC live
gate blocked by rejected SSH authentication; reported once, no retry pending
human renewal. No cluster writes or launches from this implementation yet.

Stage 1 local checkpoint: intake, queue, runner, strict adapters and audit have
155 passing focused/neighboring tests. Live qualification is outstanding. Full
eval regression is blocked by missing dependency/source imports outside this
change. Concurrency remains 1; Jupiter is rejected by deployment validation.
