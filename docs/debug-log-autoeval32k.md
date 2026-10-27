# Debugging the TACC automatic evaluation path

## Runtime compatibility

The deployed harness has Harbor 0.8.1 but different source fingerprints from the
locally reviewed adapter, and it has no transformers distribution. The serving
environment already has transformers 5.12.1 and the proven vLLM build.

Use a private exact Harbor checkout with the existing harness dependencies;
perform tokenizer preparation in serving Python. No package installation or
shared runtime modification is needed if the import/schema gate passes.

## Cache reuse

A revision-named cache directory is not content proof. Verify each selected
model/tokenizer file against published immutable-revision metadata, using LFS
SHA256 or git blob SHA1 as appropriate. Stage private references only after all
hashes pass; reject corrupt bytes before starting serving. Preparation has an
explicit subprocess deadline. Four regression tests pass; combined gate 159.

## Intake deadline

The first metadata-only intake exited 124 at its internal 32-second bound; no
job was submitted. A staged read-only probe showed HF import at 0.41 seconds,
published revision metadata at 0.58 seconds, and transformers import at 15.04
seconds. Remove that unnecessary GPU-stack import from intake and download only
config/template metadata. Serving Python still verifies the actual loaded
tokenizer against the frozen hash before execution. Retain the partial intake
cache and use a fresh metadata-attempt directory; never overwrite evidence.

The existing TACC batch recipe also loads gcc/13.2.0 and cuda/12.8 and uses
enforce-eager. Carry these into the isolated allocation instead of introducing
an unqualified JIT/compiler configuration.

## Scheduler receipt

The first submission from commit 87e96c58da9a757365e4e85eb69e3aebfd5561cf
returned text rejected by the strict receipt parser. The original implementation
did not retain the text, so its exact format is unknown. Scheduler records showed
one exact-name job, 1104859, pending Priority. Reconciled that job through the
existing queue API without resubmission. Calling the same CLI request then
returned its existing receipt, confirming duplicate suppression live.

Save scheduler stdout before parsing in future submissions. If the response is
ambiguous, reconcile only one new exact-name scheduler row; reject zero or
multiple matches. Reject an existing exact-name job before dispatch as well.
Three regression tests cover those cases. Combined local gate: 164 passed.
This fix is not deployed into the frozen submitted job.

## CUDA wheel version metadata

Job 1104938 failed after 89 seconds with ValueError after checkpoint integrity
passed and before serving. No gpu.json survived, so this run does not establish
a recorded GPU pass. No Daytona creation was attempted. The exact exception
message was not retained; the failed evidence remains unchanged.

Read-only inspection found distribution version
0.20.1rc1.dev0+g506029539.cu128 but imported source version
0.20.1rc1.dev0+g506029539. The runner compared the latter against the former.
This deterministic mismatch is consistent with the observed failure point.

Require the exact distribution pin and separately require the imported version
to match it, permitting only the terminal CUDA wheel suffix difference. Record
both versions and GPU evidence before checking. Do not change the environment.
Regression-first test failed on the missing helper; after the fix, 185 focused
and neighboring tests pass. Ruff passes with caching disabled; the repository
has no infra/pre-commit.py entry point. A fresh synthetic run is still required.

## Schema mutation and environment setup timeout

Job 1105033 failed in 241 seconds. Checkpoint integrity, GPU BF16, both vLLM
version checks and an actual completion passed. Daytona snapshot lookup and
creation succeeded. Environment startup timed out at 60 seconds in ensure_dirs,
waiting for get_session_command through the async toolbox client. Agent and
verifier never started; reward is null. The stack identifies the waiting call,
not why it stalled. Do not increase the timeout or infer an authentication fault.

Result processing separately rejected the stored config: JobConfig's compatibility
validator mutated the submitted dictionary, moving orchestrator.retry and
orchestrator.n_concurrent_trials to top-level fields. Raw trial protocol fields
and task hashes match. Validate a deep copy so the frozen input stays unchanged;
retain the effective-field assertions. A regression using the real schema was
red before the helper and passes after the fix. Combined suite: 186 passed.

Cleanup for exact sandbox 13980649-df31-4009-b9d6-b4ca1327a29a recorded not_found
and an independent SDK lookup also returned DaytonaNotFoundError. No active
allocation remains. Preserve failed raw outputs, config and journals. Next,
isolate the session-command path with bounded diagnostics before another model
smoke; no end-to-end pass or repository migration is established.

## Session API isolation

Installed Daytona and async toolbox client are both 0.203.0; aiohttp is 3.14.5.
get_session_command supports an explicit request_timeout but Harbor calls it
without one. The outer 60-second environment cap did stop the stalled request;
that is not evidence that a longer timeout would help. Prepare one login-node
diagnostic comparing direct SDK execution with session create/execute/get/logs.
Body cap 24s, cleanup cap 10s, one existing-snapshot environment, no GPU job.
This checks API behavior on login only; compute remains a separate gate.

First diagnostic exited in 0.16s after successful snapshot lookup, before any
creation intent. Its evidence writer's positional name argument collided with
the snapshot name field. Rename that parameter; preserve the failed diagnostic
directory and use a new v2 directory. This attempt tested no command transport.

Corrected diagnostic bbf2616 passed on the TACC login node in 2.38s. Snapshot
lookup, create, direct command, session create/execute/get/logs all passed;
exact output and deletion absence confirmed. Sandbox ID
8e1599b2-069b-4e46-b5f8-f9182fcda353. No model or allocation was used. This
does not reproduce or explain the compute stall. Next: the same short diagnostic
from compute before loading another model; retain the original failed trial.

The compute diagnostic 1105146 passed the same direct/session operations in
5.418s on c642-011; allocation elapsed39s, COMPLETED0:0. Exact temporary sandbox
76beb1fc-b795-45dc-b466-b9bfa7b788c2 independently returned DaytonaNotFoundError.
This rules out a persistent inability to use these APIs from that compute node,
not a transient failure or a difference in Harbor's session polling/setup path.
Next compare those paths before another model smoke. Do not increase timeout.

## Real Harbor setup isolation

The passing SDK probe omitted Harbor's root-user shell wrapper, log-directory
mount targets, shared-client configuration, and its session polling loop. Test
the unchanged SnapshotOnly.start path with the same hello task, mount targets
and 60-second environment budget, recording entry/return/error for each setup
operation. One fresh login-node diagnostic, no model; hard process cap210s
includes existing failure cleanup. No command retry or transport alteration is
added by the instrumentation. Follow with one exact-output exec and deletion.
The original 1105146 allocation has now disappeared from the live queue.

The real Harbor login diagnostic reproduced the failure: create/session submit
passed, then 761 status queries never returned an exit code. Setup stopped at
60 seconds; total 61.375s including cleanup. Exact environment
d8314a82-2543-41b3-bfcb-3bfc4c6885a2 independently returned NotFound. Results
remain in autoeval32k-tacc-harbor-setup-20261027; no model was loaded.

A fresh 8.601s wrapper isolation identified the user mismatch. The existing
snapshot starts as uid1001(daytona). Plain bash printed WRAPPER_OK and exited0;
the otherwise identical su-root wrapper had no exit after5s, with su still in
the process tree. No API request timeout or authentication exception occurred.
Temporary environment5b242de3-a492-4ffa-8151-65af9e7d8ab0 was deleted. This
supports a root-user switch blocking, not a generally unavailable transport.

Set the disposable environment's explicit os_user=root when creating from the
existing snapshot, matching Harbor's privileged directory preparation. Reject
nonroot create-parameter drift before intent. Do not modify the snapshot,
shared Harbor/runtime, shell wrappers, polling, or60s budget. Two regression
checks were red before this change;134 autoeval tests now pass and Ruff passes.
Live setup verification with this fix is still required before another model run.

The os_user fix did not change the actual toolbox process user. In fresh v2,
the API reported root but id still returned uid1001(daytona), and the identical
su command remained running. Preserve that failed attempt and commit7d68250;
remove the ineffective creation override. A bounded direct sudo -n id -u in
that exact owned environment returned0, establishing noninteractive privilege
availability without changing authentication settings.

Override only explicit root execution in the private adapter: sudo -n, no
interactive fallback, preserving shell/env/cwd and the inherited timeout,
session submission and polling. Other execution remains delegated unchanged.
Behavior tests execute the produced command against a disposable fake privilege
boundary and verify cwd/env preservation and denial exit1; both failed before
the fix.135 autoeval tests pass; live verification remains required.

Fresh v3 at 3503979 passed real Harbor setup, exact-output command and cleanup
in 3.678s. The environment deadline remains 60s. Exact environment
53505d4a-3dfa-42f5-918f-890f1ed803bf independently returned NotFound before
the full smoke was submitted. This qualifies login setup only. Both failed
setup attempts and the wrapper test are preserved, with independent cleanup.

Fresh full smoke 1105363/run ae32-a5204d2ec269b61beaf5147f8bb2b951 was
submitted once at 2026-10-27T23:46:20Z from 3503979. Initial PENDING Priority;
900s allocation/Requeue0, queue deadline 2026-10-28T05:46:20Z. Prior scored
results and failures are untouched. Full agent/reward/audit proof remains pending.
