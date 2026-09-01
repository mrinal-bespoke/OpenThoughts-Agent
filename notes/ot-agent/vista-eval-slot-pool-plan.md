# Vista evaluation slot-pool plan

Date: 2026-09-01
Status: activated on Vista
Target: `/Users/mrinal/dev/OpenThoughts-Agent`
Working copy: live operator workspace; preserve unrelated dirty and untracked work
Branch: `pr/c-rescue-tacc-campaign-scripts`
Evidence: `agent_logs/2026-09-01_vista-eval-slot-pools.md`

## Goal

Keep Vista's 20 runnable production-job allowance split non-preemptively into at
most eight flawed_summ jobs and at most twelve newer leaderboard-evaluation jobs.
Within the leaderboard pool, target eight agentic and four non-agentic jobs. The
12-slot pool must remain work-conserving: when one kind has no eligible work or
opens a repeated-failure circuit, the other kind may borrow its unused slots.
Queue order is oldest eligible work first. Existing jobs are never cancelled.

## Mechanism

Separate the current overloaded global `--cap` into a hard global ceiling and a
campaign-owned ceiling. Add a small pure admission-policy layer for the 12-slot
leaderboard pool, with per-kind soft quotas, FCFS ordering, and a bounded failure
circuit. The policy returns submission allowances; the established agentic and
non-agentic launch paths remain responsible for constructing jobs and validating
results.

## Stage map

| Stage | Title | What | Layer | Cost | Gate |
|---|---|---|---|---:|---|
| 1 ✅ | flawed_summ ownership cap | Add `--campaign-cap 8` alongside global cap 20 | controller | CPU | 15/15 focused tests green |
| 2 ✅ | leaderboard FCFS policy | 12 total, 8/4 soft quotas, borrowing, failure circuit | scheduler | CPU | 25/25 combined focused tests green |
| 3 ✅ | Vista activation | Restart controller with new flags and start pool scheduler | operations | live read-only + controller restart | IDs unchanged; first sweep held at flawed 10/8 and global 10/20 |

## Global invariants

- No existing RUNNING or PENDING eval job is cancelled or held by this change.
- `JobHeldUser` remains excluded from runnable-cap accounting.
- The hard global runnable-job cap remains 20.
- The flawed_summ controller counts only its durable `owned_job_ids` against the
  new campaign cap; unrelated work cannot suppress its reserved eight slots.
- The leaderboard 8/4 split is soft, while 12 total is hard.
- A failure circuit suppresses new submissions only; it never kills a live job.
- Default/off behavior remains compatible: omitting `--campaign-cap` preserves
  the prior global-cap-only controller behavior.
- Minimal diff: serving, benchmark definitions, result validation, upload, and DB
  registration paths are not changed.

## Borrow map

- `eval/tacc/campaign/priority_backfill.py`: `Controller.sweep`, durable
  `owned_job_ids`, `runnable_job_count`, and CLI cap parsing. Anchors drift;
  reconfirm before each edit.
- `eval/tacc/campaign/nonagentic_autoeval.py`: structured non-agentic submission
  and durable result-root identity; do not duplicate its command construction.
- `eval/tacc/campaign/autoeval_snowball.sh`: historical two-kind retry behavior
  and job naming; superseded only after the new scheduler dry-run is green.
- `hpc/vllm/smoke_grug_vllm_tacc.sbatch`: proven common agentic/non-agentic job
  wrapper; left unchanged.
- `tests/eval/test_priority_backfill.py`: controller regression anchors.

## Safety and validation discipline

Stage 1 first proves the old no-campaign-cap behavior, then the owned-cap behavior.
Stage 2 is a pure CPU policy simulation: no scheduler, network, DB, or result writes.
Stage 3 starts with dry-run/status output, then performs only controller-process
replacement and scheduler startup. Live jobs drain naturally; no cancellation is
permitted. A total runnable count above 20 or an owned flawed_summ count above eight
after convergence is a NO-GO.
