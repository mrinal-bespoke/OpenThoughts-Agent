#!/usr/bin/env python3
"""Scheduler-aware flawed_summ backfill controller for TACC Vista.

The controller keeps the flawed_summ campaign moving without crowding out work
that the operator has explicitly marked as higher priority.  A guard is armed
before launching a priority pipeline.  While a matching job is queued/running,
and for a quiet period after it disappears, the controller submits no new
flawed_summ jobs and holds only pending jobs that it submitted itself.

Examples (from the OpenThoughts-Agent repository root):

    python eval/tacc/campaign/priority_backfill.py guard snowball \
      --pattern '(snowball|grug)'
    python eval/tacc/campaign/priority_backfill.py run
    python eval/tacc/campaign/priority_backfill.py status --live
    python eval/tacc/campaign/priority_backfill.py unguard snowball

The state file is durable on $SCRATCH.  The controller is single-instance via
an advisory lock and never deletes Supabase rows or cancels running jobs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scripts.database.reconcile_flawed_summ_state import priority_tier


ACTIVE_STATES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}
IGNORED_PENDING_REASONS = {"JobHeldUser", "DependencyNeverSatisfied"}

BENCHMARKS = {
    "swebench": {
        "family": "swebench-verified-random-100-folders",
        "preset": "swebench",
        "suffix": "swe",
    },
    "v2": {"family": "dev_set_v2", "preset": "v2", "suffix": "v2"},
    "tb2": {
        "family": "terminal_bench_2",
        "preset": "tb2",
        "suffix": "tb2",
    },
}
DEFAULT_BENCHMARK_ORDER = ["swebench", "v2", "tb2"]
DEFAULT_TIER_ORDER = ["a1", "a3"]


@dataclass(frozen=True)
class QueueJob:
    job_id: str
    name: str
    state: str
    reason: str
    partition: str

    @property
    def runnable(self) -> bool:
        return (
            self.state in ACTIVE_STATES and self.reason not in IGNORED_PENDING_REASONS
        )


def utc_now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def parse_squeue(output: str) -> list[QueueJob]:
    jobs: list[QueueJob] = []
    for line in output.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        job_id, name, state, reason, partition = (part.strip() for part in parts)
        jobs.append(
            QueueJob(
                job_id=job_id,
                name=name,
                state=state,
                reason=reason.strip("()"),
                partition=partition,
            )
        )
    return jobs


def query_queue(user: str) -> list[QueueJob]:
    result = run_command(["squeue", "-h", "-u", user, "-o", "%i|%200j|%T|%R|%P"])
    return parse_squeue(result.stdout)


def default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "guards": {},
        "last_priority_seen_at": None,
        "owned_job_ids": [],
        "held_job_ids": [],
        "benchmark": "swebench",
        "benchmark_order": DEFAULT_BENCHMARK_ORDER,
        "tier": "a1",
        "tier_order": DEFAULT_TIER_ORDER,
        "last_reconcile_at": None,
        "last_reconcile_counts": {},
        "last_submission_at": None,
    }


def load_state(path: Path) -> dict[str, Any]:
    state = default_state()
    if path.exists():
        loaded = json.loads(path.read_text())
        state.update(loaded)
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def matching_priority_jobs(
    jobs: Iterable[QueueJob], guards: dict[str, dict[str, Any]]
) -> dict[str, list[QueueJob]]:
    matches: dict[str, list[QueueJob]] = {}
    for label, guard in guards.items():
        pattern = re.compile(guard["pattern"], re.IGNORECASE)
        found = [job for job in jobs if job.runnable and pattern.search(job.name)]
        if found:
            matches[label] = found
    return matches


def priority_pause_remaining(
    state: dict[str, Any],
    matches: dict[str, list[QueueJob]],
    now: int,
    quiet_seconds: int,
) -> tuple[bool, int]:
    if matches:
        state["last_priority_seen_at"] = now
        return True, quiet_seconds
    last_seen = state.get("last_priority_seen_at")
    if last_seen is None:
        return False, 0
    remaining = max(0, quiet_seconds - (now - int(last_seen)))
    return remaining > 0, remaining


def runnable_job_count(jobs: Iterable[QueueJob]) -> int:
    return sum(job.runnable for job in jobs)


def owned_runnable_job_count(
    jobs: Iterable[QueueJob], owned_job_ids: Iterable[str]
) -> int:
    """Count runnable jobs owned by this durable controller."""
    owned = set(owned_job_ids)
    return sum(job.runnable and job.job_id in owned for job in jobs)


def available_submission_slots(
    jobs: Iterable[QueueJob],
    owned_job_ids: Iterable[str],
    *,
    global_cap: int,
    campaign_cap: int | None,
) -> int:
    """Return safe new slots under both global and campaign limits.

    ``campaign_cap=None`` preserves the historical behavior where ``--cap``
    was the only limit. A configured campaign cap is ownership-aware, so
    unrelated leaderboard jobs consume global capacity but not flawed_summ's
    reserved share.
    """
    jobs = list(jobs)
    global_free = max(0, global_cap - runnable_job_count(jobs))
    if campaign_cap is None:
        return global_free
    campaign_free = max(
        0,
        campaign_cap - owned_runnable_job_count(jobs, owned_job_ids),
    )
    return min(global_free, campaign_free)


def pending_models(
    summary: dict[str, Any], benchmark: str, tier: str = "a1"
) -> list[str]:
    family = BENCHMARKS[benchmark]["family"]
    candidates = [
        row
        for row in summary["rows"]
        if row["benchmark"] == family
        and row["state"] == "pending"
        and priority_tier(row["model"]) == tier
    ]

    def sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
        # A below-gate row is a warm resume and policy priority 1.  A genuinely
        # new row follows.  Source row number is the stable tie-breaker.
        reason = row.get("reason") or ""
        is_new = reason == "no owned gate-passing eval"
        row_number = int(str(row["row"]).lstrip("r") or 0)
        return (1 if is_new else 0, row_number, row["model"])

    ordered: list[str] = []
    seen: set[str] = set()
    for row in sorted(candidates, key=sort_key):
        if row["model"] not in seen:
            ordered.append(row["model"])
            seen.add(row["model"])
    return ordered


def active_families(summary: dict[str, Any]) -> set[str]:
    active_states = {"slurm-active-unverified", "db-active-unverified"}
    return {
        row["benchmark"] for row in summary["rows"] if row["state"] in active_states
    }


def select_benchmark(state: dict[str, Any], summary: dict[str, Any]) -> str | None:
    current = state.get("benchmark", "swebench")
    benchmark_order = state.get("benchmark_order", DEFAULT_BENCHMARK_ORDER)
    for tier in state.get("tier_order", DEFAULT_TIER_ORDER):
        if pending_models(summary, current, tier):
            state["tier"] = tier
            return current
        for benchmark in benchmark_order:
            if pending_models(summary, benchmark, tier):
                state["tier"] = tier
                state["benchmark"] = benchmark
                return benchmark
    return None


def submitted_job_ids(output: str) -> list[str]:
    patterns = (
        r"Submitted batch job\s+(\d+)",
        r"->\s*Submitted(?:\s+batch\s+job)?\s+(\d+)",
        r"SLURM job\s+(\d+)",
    )
    found: list[str] = []
    for pattern in patterns:
        for job_id in re.findall(pattern, output, flags=re.IGNORECASE):
            if job_id not in found:
                found.append(job_id)
    return found


class Controller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo = Path(args.repo).resolve()
        self.state_path = Path(args.state_file).resolve()
        self.state_dir = self.state_path.parent
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state = load_state(self.state_path)

    def log(self, message: str) -> None:
        stamp = datetime.now().astimezone().strftime("%F %T %Z")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with (self.state_dir / "controller.log").open("a") as handle:
            handle.write(line + "\n")

    def save(self) -> None:
        save_state(self.state_path, self.state)

    def refresh_owned_jobs(self, jobs: list[QueueJob]) -> None:
        live_ids = {job.job_id for job in jobs}
        self.state["owned_job_ids"] = [
            job_id
            for job_id in self.state.get("owned_job_ids", [])
            if job_id in live_ids
        ]
        self.state["held_job_ids"] = [
            job_id
            for job_id in self.state.get("held_job_ids", [])
            if job_id in live_ids
        ]

    def hold_owned_pending(self, jobs: list[QueueJob]) -> None:
        owned = set(self.state.get("owned_job_ids", []))
        held = set(self.state.get("held_job_ids", []))
        to_hold = [
            job.job_id
            for job in jobs
            if job.job_id in owned
            and job.state == "PENDING"
            and job.reason not in IGNORED_PENDING_REASONS
            and job.job_id not in held
        ]
        if to_hold:
            self.log(
                f"holding controller-owned pending backfill jobs: {','.join(to_hold)}"
            )
        if self.args.dry_run:
            return
        for job_id in to_hold:
            result = run_command(["scontrol", "hold", job_id], check=False)
            if result.returncode != 0:
                self.log(f"hold warning for {job_id}: {result.stdout.strip()}")
                continue
            # Persist each successful hold immediately. A later scheduler or
            # controller error must not leave an untracked JobHeldUser job.
            held.add(job_id)
            self.state["held_job_ids"] = sorted(held)
            self.save()

        # Reconcile against actual scheduler state. This also captures the
        # unlikely case where Slurm applied a hold despite a non-zero command.
        try:
            refreshed = query_queue(self.args.user)
        except (OSError, subprocess.SubprocessError) as exc:
            self.log(f"hold reconciliation warning: {exc}")
            return
        actual_held = {
            job.job_id
            for job in refreshed
            if job.job_id in owned
            and job.state == "PENDING"
            and job.reason == "JobHeldUser"
        }
        self.state["held_job_ids"] = sorted(actual_held)
        self.save()

    def release_owned(self) -> None:
        held = list(self.state.get("held_job_ids", []))
        if not held:
            return
        self.log(f"releasing controller-held backfill jobs: {','.join(held)}")
        if self.args.dry_run:
            return
        remaining = set(held)
        for job_id in held:
            result = run_command(["scontrol", "release", job_id], check=False)
            if result.returncode != 0:
                self.log(f"release warning for {job_id}: {result.stdout.strip()}")
                continue
            remaining.discard(job_id)
            self.state["held_job_ids"] = sorted(remaining)
            self.save()

        # A failed release can still have changed scheduler state. Conversely,
        # a successful command must not make us forget a job Slurm kept held.
        try:
            refreshed = query_queue(self.args.user)
        except (OSError, subprocess.SubprocessError) as exc:
            self.log(f"release reconciliation warning: {exc}")
            return
        owned = set(self.state.get("owned_job_ids", []))
        actual_held = {
            job.job_id
            for job in refreshed
            if job.job_id in owned
            and job.state == "PENDING"
            and job.reason == "JobHeldUser"
        }
        self.state["held_job_ids"] = sorted(actual_held)
        self.save()

    def cluster_snapshot(self) -> str:
        result = run_command(
            ["sinfo", "-h", "-p", self.args.partition, "-o", "%t|%D"], check=False
        )
        if result.returncode != 0:
            return "unknown"
        counts: dict[str, int] = {}
        for line in result.stdout.splitlines():
            try:
                state, count = line.strip().split("|", 1)
                counts[state] = counts.get(state, 0) + int(count)
            except ValueError:
                continue
        return (
            ",".join(f"{key}={value}" for key, value in sorted(counts.items()))
            or "unknown"
        )

    def reconcile(self) -> dict[str, Any]:
        command = [
            sys.executable,
            "scripts/database/reconcile_flawed_summ_state.py",
            "--slurm-user",
            self.args.user,
            "--active-max-age-hours",
            "0",
            "--format",
            "json",
        ]
        result = run_command(command, cwd=self.repo, env=self.launch_env())
        summary = json.loads(result.stdout)
        self.state["last_reconcile_at"] = utc_now_epoch()
        self.state["last_reconcile_counts"] = summary.get("counts", {})
        (self.state_dir / "flawed-summ-STATE.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        try:
            from scripts.database.reconcile_flawed_summ_state import render_markdown

            report = render_markdown(summary)
            (self.state_dir / "flawed-summ-STATE.md").write_text(report)
            if self.args.state_report:
                report_path = Path(self.args.state_report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(report)
        except Exception as exc:  # State JSON remains authoritative.
            self.log(f"markdown state refresh warning: {exc}")
        return summary

    def launch_env(self) -> dict[str, str]:
        env = os.environ.copy()
        harbor_src = self.args.harbor_src
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join(
            part for part in (harbor_src, str(self.repo), existing) if part
        )
        env["EVAL_UPLOAD_HF_ORG"] = self.args.hf_org
        return env

    def run_hooks(self) -> None:
        hooks = (
            ["bash", "eval/tacc/campaign/detect_stalls.sh", "no"],
            ["bash", "eval/tacc/campaign/health_check.sh"],
            ["bash", "eval/tacc/campaign/prune_partial_trials.sh"],
        )
        for command in hooks:
            result = run_command(
                command, cwd=self.repo, env=self.launch_env(), check=False
            )
            if result.stdout.strip():
                self.log(result.stdout.strip())
            if result.returncode != 0:
                raise RuntimeError(f"pre-submit hook failed: {shlex.join(command)}")

    def write_priority_list(
        self, benchmark: str, tier: str, models: list[str]
    ) -> Path:
        suffix = BENCHMARKS[benchmark]["suffix"]
        path = (
            self.repo
            / "eval"
            / "lists"
            / f"flawed_summ_backfill_{tier}_{suffix}.txt"
        )
        path.write_text("\n".join(models) + ("\n" if models else ""))
        return path

    def launch(
        self, benchmark: str, tier: str, slots: int, models: list[str]
    ) -> list[str]:
        priority_file = self.write_priority_list(benchmark, tier, models)
        relative_priority = priority_file.relative_to(self.repo)
        before_ids = {job.job_id for job in query_queue(self.args.user)}
        command = [
            sys.executable,
            "-m",
            "hpc.launch",
            "--job_type",
            "eval_listener",
            "--cluster-config",
            "tacc",
            "--preset",
            BENCHMARKS[benchmark]["preset"],
            "--require-priority-list",
            "--priority-file",
            str(relative_priority),
            "--config-yaml",
            "dcagent_eval_config_no_override.yaml",
            "--force-reeval",
            "--once",
            "--verbose",
            "--slurm-time",
            "47:59:00",
            "--max-jobs-submitted",
            str(slots),
        ]
        self.log(
            f"launching up to {slots} {benchmark}/{tier} backfills from "
            f"{len(models)} gate-pending rows"
        )
        if self.args.dry_run:
            self.log("DRY_RUN: " + shlex.join(command))
            return []
        result = run_command(command, cwd=self.repo, env=self.launch_env(), check=False)
        (self.state_dir / "last_listener.log").write_text(result.stdout)
        for line in result.stdout.splitlines():
            if re.search(r"submitted|no eligible|error|fatal", line, re.IGNORECASE):
                self.log(line)
        if result.returncode != 0:
            raise RuntimeError(f"eval listener exited {result.returncode}")
        parsed_ids = submitted_job_ids(result.stdout)
        new_eval_ids = [
            job.job_id
            for job in query_queue(self.args.user)
            if job.job_id not in before_ids
            and (job.name.startswith("eval_") or job.name.startswith("res_"))
        ]
        return list(dict.fromkeys([*parsed_ids, *new_eval_ids]))

    def sweep(self) -> None:
        now = utc_now_epoch()
        jobs = query_queue(self.args.user)
        self.refresh_owned_jobs(jobs)
        matches = matching_priority_jobs(jobs, self.state.get("guards", {}))
        paused, remaining = priority_pause_remaining(
            self.state, matches, now, self.args.quiet_seconds
        )
        runnable = runnable_job_count(jobs)
        campaign_runnable = owned_runnable_job_count(
            jobs, self.state.get("owned_job_ids", [])
        )
        campaign_status = (
            f"{campaign_runnable}/{self.args.campaign_cap}"
            if self.args.campaign_cap is not None
            else f"{campaign_runnable}/unbounded"
        )
        self.log(
            f"queue runnable={runnable}/{self.args.cap}; "
            f"campaign={campaign_status}; gh={self.cluster_snapshot()}; "
            f"priority={','.join(matches) if matches else 'none'}"
        )

        if paused:
            self.hold_owned_pending(jobs)
            if matches:
                details = ", ".join(
                    f"{label}:[{','.join(job.job_id for job in guarded_jobs)}]"
                    for label, guarded_jobs in matches.items()
                )
                self.log(f"backfill paused for protected pipeline(s): {details}")
            else:
                self.log(f"backfill quiet window: {remaining}s remaining")
            self.save()
            return

        self.release_owned()
        # Release changes the runnable count; query again before calculating slots.
        jobs = query_queue(self.args.user)
        self.refresh_owned_jobs(jobs)
        runnable = runnable_job_count(jobs)
        campaign_runnable = owned_runnable_job_count(
            jobs, self.state.get("owned_job_ids", [])
        )
        slots = available_submission_slots(
            jobs,
            self.state.get("owned_job_ids", []),
            global_cap=self.args.cap,
            campaign_cap=self.args.campaign_cap,
        )

        reconcile_due = self.state.get("last_reconcile_at") is None or (
            now - int(self.state["last_reconcile_at"]) >= self.args.reconcile_seconds
        )
        summary_path = self.state_dir / "flawed-summ-STATE.json"
        if slots <= 0:
            if reconcile_due or not summary_path.exists():
                self.reconcile()
            if runnable >= self.args.cap:
                self.log("backfill at global runnable-job cap; no submission")
            elif (
                self.args.campaign_cap is not None
                and campaign_runnable >= self.args.campaign_cap
            ):
                self.log("backfill at campaign-owned job cap; no submission")
            else:
                self.log("backfill has no safe submission slots")
            self.save()
            return

        # Always reconcile immediately before a submission. Reusing a cached
        # list here can double-submit rows whose prior listener just created a
        # Pending DB row.
        summary = self.reconcile()
        benchmark = select_benchmark(self.state, summary)
        if benchmark is None:
            self.log(
                "no gate-pending rows remain in configured priority tiers"
            )
            self.save()
            return
        tier = self.state.get("tier", "a1")
        models = pending_models(summary, benchmark, tier)
        if not models:
            self.log(f"no gate-pending {tier} rows for {benchmark}")
            self.save()
            return

        self.run_hooks()
        job_ids = self.launch(benchmark, tier, slots, models)
        if job_ids:
            owned = set(self.state.get("owned_job_ids", []))
            owned.update(job_ids)
            self.state["owned_job_ids"] = sorted(owned)
            self.state["last_submission_at"] = now
            self.log(f"recorded controller-owned jobs: {','.join(job_ids)}")
        elif not self.args.dry_run:
            self.log(
                "listener returned successfully but no submitted job IDs were found"
            )
        self.save()


def build_parser() -> argparse.ArgumentParser:
    scratch = os.environ.get("SCRATCH", str(Path.home()))
    repo = os.environ.get("DCFT", str(Path(scratch) / "OpenThoughts-Agent"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=repo)
    parser.add_argument(
        "--state-file",
        default=str(Path(scratch) / "flawed_summ_backfill" / "state.json"),
    )
    parser.add_argument("--state-report")
    parser.add_argument("--user", default=os.environ.get("USER", ""))
    parser.add_argument("--harbor-src", default=str(Path(scratch) / "harbor" / "src"))
    parser.add_argument("--hf-org", default="laion")
    parser.add_argument("--partition", default="gh")
    parser.add_argument("--cap", type=int, default=20)
    parser.add_argument(
        "--campaign-cap",
        type=int,
        default=None,
        help=(
            "optional maximum runnable jobs owned by this controller; "
            "unrelated jobs still count only against --cap"
        ),
    )
    parser.add_argument("--quiet-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--reconcile-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("once")
    subparsers.add_parser("run")
    status = subparsers.add_parser("status")
    status.add_argument("--live", action="store_true")
    guard = subparsers.add_parser("guard")
    guard.add_argument("label")
    guard.add_argument("--pattern", required=True)
    unguard = subparsers.add_parser("unguard")
    unguard.add_argument("label")
    benchmark = subparsers.add_parser("set-benchmark")
    benchmark.add_argument("benchmark", choices=sorted(BENCHMARKS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cap < 0:
        raise SystemExit("--cap must be non-negative")
    if args.campaign_cap is not None and args.campaign_cap < 0:
        raise SystemExit("--campaign-cap must be non-negative")
    state_path = Path(args.state_file).resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "guard":
        re.compile(args.pattern)
        state = load_state(state_path)
        state.setdefault("guards", {})[args.label] = {
            "pattern": args.pattern,
            "armed_at": utc_now_epoch(),
        }
        state["last_priority_seen_at"] = utc_now_epoch()
        save_state(state_path, state)
        print(f"armed guard {args.label!r}: {args.pattern}")
        return 0
    if args.command == "unguard":
        state = load_state(state_path)
        state.setdefault("guards", {}).pop(args.label, None)
        save_state(state_path, state)
        print(f"removed guard {args.label!r}")
        return 0
    if args.command == "set-benchmark":
        state = load_state(state_path)
        state["benchmark"] = args.benchmark
        save_state(state_path, state)
        print(f"active benchmark set to {args.benchmark}")
        return 0
    if args.command == "status":
        state = load_state(state_path)
        payload: dict[str, Any] = {"state": state}
        if args.live:
            jobs = query_queue(args.user)
            payload["queue"] = [asdict(job) for job in jobs]
            payload["priority_matches"] = {
                label: [asdict(job) for job in guarded_jobs]
                for label, guarded_jobs in matching_priority_jobs(
                    jobs, state.get("guards", {})
                ).items()
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    lock_path = state_path.parent / "controller.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "another priority_backfill controller is already running",
                file=sys.stderr,
            )
            return 2
        controller = Controller(args)
        if args.command == "once":
            controller.sweep()
            return 0
        while True:
            try:
                controller.sweep()
            except Exception as exc:
                controller.log(
                    f"sweep failed safely (no submission): {type(exc).__name__}: {exc}"
                )
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
