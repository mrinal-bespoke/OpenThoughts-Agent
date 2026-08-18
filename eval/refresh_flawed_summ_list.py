#!/usr/bin/env python
"""Regenerate the flawed_summ priority list from Supabase ground truth.

The eval listener hot-reloads its --priority-file every iteration, so rewriting this
file is how a long-running listener learns what is still outstanding.

WHY THIS EXISTS: flawed_summ re-evals need --force-reeval, because every target model
already has an OLD (pre-fix) Finished row and the listener would otherwise skip it.
But --force-reeval also bypasses the "already finished" guard, so a persistent listener
would re-submit a model forever once it completes. Pruning the list here is what stops
that loop: a model drops out as soon as it has a Finished tb2 row dated >= CUTOFF.

Run it on a timer (see the refresher tmux window). Idempotent, read-only against
Supabase, only writes the list file.
"""
import os, re, sys, datetime
from huggingface_hub import hf_hub_download
from supabase import create_client

CUTOFF = "2026-08-01"          # a Finished row at/after this date counts as re-evaled
BENCH  = sys.argv[1] if len(sys.argv) > 1 else "terminal_bench_2"
FAMILY = "a1-"                 # priority order is a1 -> a3 -> g
OUT    = os.path.join(os.path.dirname(__file__), "lists",
                      sys.argv[2] if len(sys.argv) > 2 else "mrinal_flawed_summ_a1_tb2.txt")
# Stranded on Leonardo at 96-97% valid by warm resume; resume them there, never redo here.
STRANDED = {"DCAgent/a1-nl2bash", "DCAgent/a1-nemotron_csharp", "DCAgent/a1-stack_dockerfile"}

c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
bs = {b["id"]: b["name"] for b in c.table("benchmarks").select("id,name").execute().data}
tb2 = {k for k, v in bs.items() if v == BENCH}
models = {m["name"]: m["id"] for m in c.table("models").select("id,name").execute().data}

p = hf_hub_download("penfever/flawed-summ-evals", "STATE.md", repo_type="dataset",
                    token=os.environ["HF_TOKEN"])
sec, cand = None, []
for line in open(p):
    h = re.match(r"^#{2,4}\s+(.*)", line.strip())
    if h:
        sec = h.group(1).split("—")[0].strip(); continue
    m = re.match(r"^\|\s*(r\d+)\s*\|\s*([^|]+?)\s*\|\s*pending\s*\|", line)
    if m and sec == BENCH and FAMILY in m.group(2):
        cand.append(m.group(2).strip())

keep, done = [], []
for n in cand:
    if n in STRANDED:
        continue
    mid = models.get(n)
    if not mid:
        keep.append(n); continue
    rows = [x for x in c.table("sandbox_jobs")
              .select("job_status,created_at,benchmark_id").eq("model_id", mid).execute().data
            if x.get("benchmark_id") in tb2 and x["job_status"] == "Finished"
            and str(x["created_at"])[:10] >= CUTOFF]
    (done if rows else keep).append(n)

# ---- ordering: code/SWE first -------------------------------------------------------------
# POLICY.md sets no topic priority (its Goal is "every DCAgent/a1-<benchmark>"), so this is an
# operator choice: terminal_bench_2 is a coding/terminal benchmark, so the code and SWE arms are
# the ones whose scores actually inform the a1 data-mix question. Non-code arms still run, just
# last. The listener consumes this file top-down, so order == priority.
TIER1 = re.compile(r"(swegym|r2egym|manybugs|pr_mining|issue_tasks|magicoder|code_contests|taco|"
                   r"exercism|stack_(pytest|junit|jest|phpunit|rspec|go|rust|ruby|csharp|selfdoc)|"
                   r"unitsyn|pymethods2test|nemotron_(cpp|junit|pytest|rspec|csharp|rust)|"
                   r"crosscodeeval|staqc|codereview|repo_scaffold)", re.I)
TIER2 = re.compile(r"(nl2bash|dockerfile|stackexchange_(unix|superuser)|bash_withtests)", re.I)
def _tier(n):
    s = n.split("/")[-1]
    return 0 if TIER1.search(s) else 1 if TIER2.search(s) else 2
keep.sort(key=lambda n: (_tier(n), n))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(keep) + ("\n" if keep else ""))
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{ts}] tiers: t1={sum(1 for n in keep if _tier(n)==0)} t2={sum(1 for n in keep if _tier(n)==1)} t3={sum(1 for n in keep if _tier(n)==2)}")
print(f"[{ts}] candidates={len(cand)} outstanding={len(keep)} completed_since_{CUTOFF}={len(done)} -> {OUT}")
if not keep:
    print(f"[{ts}] LIST EMPTY — a1/{BENCH} backlog is drained. Move to the next family (a3).")
