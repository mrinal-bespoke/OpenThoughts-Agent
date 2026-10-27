# Stage 1 — TACC execution

Use `eval/autoeval32k` only; do not call the legacy listener or batch entrypoint.
Implement explicit settings, read-only planning, immutable bundle, exclusive
submission journal, all-user queue capacity, fresh run directory and strict
Harbor adapter. Stage task copies and verify their byte hashes. Bind the exact
checkpoint, tokenizer template and runtime versions. Preserve raw results and
check verifier evidence. Every created environment needs cleanup evidence.

CPU gates: interrupted submission cannot duplicate; missing scheduler visibility
does not free capacity; malformed or drifting files reject; old run untouched;
missing snapshot cannot build; serialized token overflow rejects before request;
effective Harbor schema matches. Live gate: separately owned synthetic smoke
with GPU BF16, generated response, compute Daytona command, agent and cleanup.

Not complete until live gate passes. Access renewed and verified October 27.
Reuse proven TACC serving dependencies, pinned model cache (content verified)
and the existing daytona-small snapshot. One synthetic task, five-minute queue
wait cap and fifteen-minute allocation cap for the original attempt; never
submit an automatic retry.
Concurrency remains 1. Jupiter is rejected by the CLI.

October 27 outcome: job 1104859 stayed pending and was cancelled after the
five-minute queue budget. Intake, exact-source/schema checks and live duplicate
suppression passed; compute, agent, verifier and cleanup gates were not reached.
No successful end-to-end claim or migration is authorized by this evidence.

Human-approved continuation: job 1104938, separate run
ae32-7761e93294d7687a1ddb10418c90e538. Allow six hours queued from
2026-10-27T20:43:58Z (deadline 2026-10-28T02:43:58Z); retain the hard
900-second allocation cap. Same model, task and existing snapshot; no benchmark.
Require synthetic verifier reward 1, actual agent activity, audit and confirmed
cleanup before migration. Policy must contain rules only, not readiness status.
