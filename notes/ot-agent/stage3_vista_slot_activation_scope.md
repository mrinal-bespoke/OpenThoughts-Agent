# Stage 3 — Vista slot-policy activation

Date: 2026-09-01
Status: scoped GO
Companion: `vista-eval-slot-pool-plan.md`

## Why this is next

The tested policy must replace the live cap-20 flawed_summ process without disturbing
its running eval legs, then begin admitting newer evaluations only into free capacity.

## Change-set

- Controller process arguments on Vista: global cap 20, campaign cap 8.
- Durable leaderboard-pool tmux process: total 12, agentic 8, non-agentic 4.
- No eval-job mutation.

## GO / NO-GO gate

GO when the old controller exits cleanly, the replacement holds the same durable
state lock and owned-job state, all pre-existing eval job IDs remain live, status
shows the requested caps, and a dry-run/live sweep never admits job 21. If no new
leaderboard work is eligible, the scheduler remains idle without fabricating work.

## Composes with / depends on

Stages 1–2 green.
