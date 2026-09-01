# TACC campaign automation

The loop that drives the flawed_summ re-eval campaign on Vista, plus its health
and hygiene helpers.

These scripts ran for weeks from `$SCRATCH/spike/` and existed **nowhere else** —
not in this repo, not on any branch, not on any operator's machine. `$SCRATCH` is
a purged filesystem, so a routine cleanup would have destroyed the automation and
`eval/refresh_flawed_summ_list.py` with it, mid-campaign. They are committed here
verbatim so that can't happen again.

## What runs what

`refire_dev2.sh` is the front door: one benchmark, one listener, `--once` per
sweep, re-fired on a loop. Looping `--once` (rather than a long-lived listener)
is what POLICY prescribes, and it self-heals when the login node reaps tmux —
which silently stalled the campaign twice, once for ~23h.

Each sweep runs, in order:

| Script | Why it exists |
| --- | --- |
| `detect_stalls.sh` | Kills legs writing no files for 30 min |
| `health_check.sh` | Stall, wall-risk and per-leg failure-rate check |
| `prune_partial_trials.sh` | Removes trial dirs lacking `config.json`. A cancelled job leaves these, every resume then dies in ~3 min, and the loop retries forever — 144 failed jobs and 5h of zero progress before this was added |
| `refresh_flawed_summ_list.py` | Regenerates the priority list from Supabase |
| `drop_inflight.sh` | Drops models already queued, so the loop doesn't double-submit |

`refire_tb2.sh` is the earlier tb2 variant, kept because tb2 may need refilling.
`run_listener*.sh` / `run_refresher.sh` are the single-shot equivalents.
`backfill_traces.sh` repairs null `hf_traces_link` rows after the fact.

## Priority-aware backfill controller

`priority_backfill.py` supersedes the unconditional `refire_*.sh` loops. It
polls every five minutes and:

- caps total runnable jobs at 20 and can separately cap controller-owned
  flawed_summ jobs, so unrelated leaderboard work does not consume the
  campaign's reserved share;
- can optionally pause submissions for an explicitly guarded pipeline;
- builds the priority list from the gate-aware reconciler, so a below-gate
  `Finished` row remains a warm-resume candidate;
- selects the highest-priority enabled tier with pending work (`a1`, then
  `a3`), advancing benchmarks within that tier while already-running
  legs drain so empty job slots stay filled; and
- refreshes a durable generated `flawed-summ-STATE.{json,md}` every 30 minutes.

The current operator policy is to run without guards and cap new flawed_summ
admissions at eight. Jobs that predate this policy drain naturally; the
controller does not cancel or hold them. Guards remain available as an optional
mechanism:

```bash
bash eval/tacc/campaign/run_priority_backfill.sh guard snowball \
  --pattern '(snowball|grug)'
```

Future jobs matching a configured guard pause new submissions. Remove it with
`unguard`; without guards, the controller continuously refills to its cap.

Run one sweep, inspect state, or start the durable loop:

```bash
bash eval/tacc/campaign/run_priority_backfill.sh once
bash eval/tacc/campaign/run_priority_backfill.sh status --live
tmux new-session -d -s flawed-summ-backfill \
  'bash eval/tacc/campaign/run_priority_backfill.sh --cap 20 --campaign-cap 8 --quiet-seconds 0 run 2>&1 | tee -a $SCRATCH/flawed_summ_backfill/tmux.log'
```

Only one controller can run at a time; a lock in its state directory rejects a
second instance. Stop the old `evalloop` before enabling this controller.

## Before reusing these

**They carry hardcoded paths** — `/scratch/11694/mkumar73` and a shared conda
prefix under another operator's home. They are committed **as-is**, matching what
actually ran, rather than parameterized in the same change: rewriting paths while
rescuing them would mean committing something never executed in that form.

Parameterizing by `$SCRATCH`/`$USER` is the same work as PRs #85 and #91 and
belongs in a follow-up, tested against a real sweep.

The priority controller's global cap counts **all runnable queued/running jobs**.
`--campaign-cap` counts only durable controller-owned jobs. The standing Vista
policy uses `--cap 20 --campaign-cap 8`, leaving twelve slots for the leaderboard
pool without cancelling jobs that predate the policy. The older `refire_*.sh`
scripts retain their historical `CAP=16` value and are retired while this
controller owns refills.

## Leaderboard evaluation pool

`leaderboard_pool.py` owns admission for the other twelve slots. Its hard limits
are 20 global runnable jobs and 12 jobs named in its reviewed queue. Within the
pool it first targets eight agentic and four non-agentic jobs, then lets either
kind borrow unused capacity. Queue entries remain FCFS by `enqueued_at`. Three
consecutive terminal failures open a six-hour circuit for only that kind, so the
healthy kind can keep using the pool. The controller never cancels jobs.

The queue is an operator-reviewed JSON list. Each command must be an argv list
that submits exactly one Slurm job; shell strings and duplicate job names are
rejected. Example:

```json
[
  {
    "id": "model-revision-tb2",
    "kind": "agentic",
    "enqueued_at": "2026-09-01T16:00:00Z",
    "job_name": "lb-model-tb2",
    "command": ["sbatch", "--job-name=lb-model-tb2", "path/to/job.sbatch"]
  }
]
```

Dry-run one sweep before enabling submissions:

```bash
python eval/tacc/campaign/leaderboard_pool.py once \
  --queue "$SCRATCH/leaderboard_eval_pool/queue.json"
tmux new-session -d -s leaderboard-eval-pool \
  'python eval/tacc/campaign/leaderboard_pool.py run --submit 2>&1 | tee -a $SCRATCH/leaderboard_eval_pool/tmux.log'
```

The queue is intentionally admission-only: existing agentic and non-agentic
launchers still construct pinned, benchmark-specific commands and validate their
own outputs. Adding an entry is the reviewed handoff from discovery to execution.
