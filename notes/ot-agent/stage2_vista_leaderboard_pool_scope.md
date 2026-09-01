# Stage 2 — leaderboard FCFS pool

Date: 2026-09-01
Status: DONE locally — 25 focused tests green
Companion: `vista-eval-slot-pool-plan.md`

## Why this is next

Once flawed_summ owns at most eight slots, the remaining twelve need one admission
policy so independent agentic and non-agentic launchers cannot race past the global
cap or strand capacity behind a failing kind.

## Change-set

- `eval/tacc/campaign/leaderboard_pool.py`
- `tests/eval/test_leaderboard_pool.py`
- `eval/tacc/campaign/README.md`

## GO / NO-GO gate

GO when deterministic CPU simulations prove: total <=12; nominal 8/4 while both
kinds have backlog; either kind borrows unused slots; oldest eligible item wins;
three consecutive failures open only that kind's circuit; a later success or cooldown
restores it; and no live job is cancelled.

## Composes with / depends on

Stage 1 green; uses the same 20-job global accounting contract.
