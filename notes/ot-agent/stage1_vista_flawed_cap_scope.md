# Stage 1 — flawed_summ ownership cap

Date: 2026-09-01
Status: DONE locally — 15 focused tests green
Companion: `vista-eval-slot-pool-plan.md`

## Why this is next

The current `--cap` counts every user's runnable job. Setting it to eight would
starve flawed_summ as soon as unrelated leaderboard work exists, so ownership-aware
capacity is the prerequisite for a real 8/12 split.

## Change-set

- `eval/tacc/campaign/priority_backfill.py`
- `tests/eval/test_priority_backfill.py`
- `eval/tacc/campaign/README.md`

## GO / NO-GO gate

GO when omitting `--campaign-cap` preserves the old slot calculation, configuring
eight prevents new flawed_summ submissions at eight owned runnable jobs, unrelated
jobs do not consume the eight, and the global cap still wins. CPU-only focused tests.

Gate result: GO. `python3 -m pytest -q tests/eval/test_priority_backfill.py`
passed 15/15.

## Composes with / depends on

No prior stage.
