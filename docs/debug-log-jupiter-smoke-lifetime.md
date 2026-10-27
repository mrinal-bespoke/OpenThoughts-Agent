# Jupiter smoke proxy lifetime

## Initial evidence

Job 2540486 queued 6:29:34, then failed the proxy-lifetime guard in 1:04.
The proxy had a two-hour lifetime. GPU/serving/Daytona were unreached; cleanup
was audited. Earlier diagnostic 2540374 passed tokenizer/CUDA/BF16 in 129s.

## Hypothesis and bounded change

The startup guard is correct. A nine-hour proxy lifetime covers an eight-hour
queue allowance, a thirty-minute allocation, and cleanup. Reject execution
after the queue deadline even if the follow-up could not cancel the job.
Bind to the actual login node's unique internal address, not the old host.
Keep the proxy authenticated, internal-only and destination-allowlisted.

## Scope and gates

Local source: scripts/smoke/jupiter32k. Remote fresh directory:
/e/scratch/reformo/kumar37/jupiter-smoke-v3-20261027. Preserve v1/v2 records.
Stage 1 CPU: validate the 6.5-hour regression, queue deadline and malformed times.
Stage 2 one live synthetic allocation: GPU, serving, compute HTTPS, exact
existing snapshot, SDK command, real agent artifact, cleanup, terminal audit.
No benchmark, new snapshot, shared runtime edit or TACC adapter change.
Default-off: this standalone script has no effect on any existing launcher.

## Results

Access succeeded at 2026-10-27T20:48:17Z as kumar37. No active user jobs;
30 booster nodes idle. No reformo reservation was listed; do not invent one.
Live submission and gates remain outstanding.

CPU checks: 12 lifecycle tests passed; combined TACC and neighboring tests
total 176 passed. Ruff passed. Repository infra/pre-commit.py is absent, so
its required checks could not run; Ruff is supplementary, not a substitute.
The first SDK snapshot query hit its 30s cap. A bounded import-only diagnostic
completed in 3.73s; a direct 15s HTTPS metadata request returned HTTP200 and
the expected active snapshot ID. The initial SDK timeout's cause is unresolved.
The compute SDK gate remains mandatory and will retain its own timeout.

Proxy preparation stopped before credentials/process creation: getaddrinfo of
the short hostname resolves loopback on this login node. hostname -I exposes
10.128.1.3 plus unrelated interfaces. Select the unique 10.128/16 interface from
that inventory and regression-test absent/ambiguous addresses. Preserve the v3
preparation intent; use fresh v3a for the corrected attempt. No Slurm submission
or proxy existed in v3.
