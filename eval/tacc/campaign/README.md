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

## Before reusing these

**They carry hardcoded paths** — `/scratch/11694/mkumar73` and a shared conda
prefix under another operator's home. They are committed **as-is**, matching what
actually ran, rather than parameterized in the same change: rewriting paths while
rescuing them would mean committing something never executed in that form.

Parameterizing by `$SCRATCH`/`$USER` is the same work as PRs #85 and #91 and
belongs in a follow-up, tested against a real sweep.

`CAP=16` in the refire scripts counts **all** queued jobs, not just this
campaign's, so co-tenant jobs (e.g. snowball/grug evals) correctly reserve slots.
