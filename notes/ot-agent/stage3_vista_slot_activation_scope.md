# Stage 3 — Vista slot-policy activation

Date: 2026-09-01
Status: DONE on Vista — activated 2026-09-01 11:57 CDT
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

## Activation evidence

- Before/after Slurm job IDs were identical: eight RUNNING and two Priority
  PENDING flawed_summ/A3 jobs; two `JobHeldUser` jobs remained excluded.
- Replacement flawed_summ process is live with `--cap 20 --campaign-cap 8`.
- First sweep reported `queue runnable=10/20; campaign=10/8` and made no
  submission. Existing jobs are draining naturally toward eight.
- `leaderboard-eval-pool` is live with pool 12, nominal agentic 8 and
  nonagentic 4. Its reviewed queue is currently empty, so it submitted nothing.
