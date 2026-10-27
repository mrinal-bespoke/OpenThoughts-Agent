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

Stage 0 CPU gate passed. Stage 1 implementation/testing in progress. Human
renewed TACC access October 27; bounded reads now succeed. A private pinned
Harbor checkout reuses existing dependencies without changing the installed
environment. Smoke job 1104859 was submitted exactly once with a fifteen-minute
allocation limit and a five-minute queue limit. It remained pending Priority
and was cancelled at the queue deadline, with zero elapsed runtime and no
compute directory or Daytona creation. Live execution is outstanding.

Stage 1 local checkpoint: intake, queue, runner, strict adapters and audit have
164 passing focused/neighboring tests. Live qualification is outstanding. Full
eval regression is blocked by missing dependency/source imports outside this
change. Concurrency remains 1; Jupiter is rejected by deployment validation.

After the minimal TACC end-to-end gate passes, the human requested moving the
reusable implementation, tests and docs to open-thoughts/open-thoughts-next.
Do not migrate an unvalidated implementation or copy credentials, site-specific
deployment settings, old results or campaign monitors. Concurrency and Jupiter
follow that move.

Human approved a fresh continuation with a longer queue allowance. Job 1104938
was submitted once at 2026-10-27T20:43:58Z with six hours allowed in queue and
the unchanged fifteen-minute execution cap. Initial state: PENDING Priority.
Follow-up `tacc-32k-smoke-and-pipeline-migration` tracks the live gate and the
subsequent migration. The policy will be normative; readiness stays in logs.

Job 1104938 failed after 89 seconds before serving, with checkpoint verification
passed and no sandbox creation. A wheel/source version-string mismatch was
identified and fixed in bfc35a4; 185 tests pass. Fresh synthetic job 1105033 was
submitted once at 2026-10-27T21:10:02Z, six-hour queue deadline
2026-10-28T03:10:02Z and unchanged 900-second execution cap. Migration remains
gated on the full live pass. Prior evidence and journals remain intact.

1105033 failed after 241 seconds: GPU and serving passed, environment setup timed
out waiting for a Daytona session command. Agent/verifier were not reached.
Cleanup independently confirmed. A separate config-mutation audit bug is fixed
locally with 186 tests passing. Diagnose session transport before another smoke;
no new allocation is pending and migration remains gated.
