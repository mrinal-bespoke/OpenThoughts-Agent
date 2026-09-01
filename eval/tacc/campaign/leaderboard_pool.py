#!/usr/bin/env python3
"""Work-conserving Vista admission controller for leaderboard eval jobs.

Queue entries are operator-reviewed JSON objects with an ``id``, ``kind``
(``agentic`` or ``nonagentic``), ISO-8601 ``enqueued_at``, unique Slurm
``job_name``, and a structured ``command`` argv list that submits exactly one
job.  The controller never invokes a shell, cancels a job, or writes results.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from eval.tacc.campaign.priority_backfill import (
        QueueJob,
        query_queue,
        runnable_job_count,
    )
except ModuleNotFoundError:  # Direct ``python eval/tacc/campaign/...`` invocation.
    from priority_backfill import QueueJob, query_queue, runnable_job_count


KINDS = ("agentic", "nonagentic")
TERMINAL_SUCCESS = {"COMPLETED"}
TERMINAL_FAILURE = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}
JOB_ID_RE = re.compile(r"Submitted batch job\s+(\d+)")


@dataclass(frozen=True)
class WorkItem:
    id: str
    kind: str
    enqueued_at: str
    job_name: str
    command: tuple[str, ...]


def utc_now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def load_queue(path: Path) -> list[WorkItem]:
    raw = json.loads(path.read_text()) if path.exists() else []
    if not isinstance(raw, list):
        raise ValueError("queue must be a JSON list")
    items: list[WorkItem] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("each queue entry must be an object")
        item_id = str(row.get("id", "")).strip()
        kind = str(row.get("kind", "")).strip()
        enqueued_at = str(row.get("enqueued_at", "")).strip()
        job_name = str(row.get("job_name", "")).strip()
        command = row.get("command")
        if not item_id or item_id in seen_ids:
            raise ValueError(f"missing or duplicate queue id: {item_id!r}")
        if kind not in KINDS:
            raise ValueError(f"invalid kind for {item_id}: {kind!r}")
        try:
            datetime.fromisoformat(enqueued_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"invalid enqueued_at for {item_id}") from error
        if not job_name or job_name in seen_names:
            raise ValueError(f"missing or duplicate job_name: {job_name!r}")
        if not isinstance(command, list) or not command or not all(
            isinstance(arg, str) and arg for arg in command
        ):
            raise ValueError(f"command for {item_id} must be a non-empty argv list")
        items.append(WorkItem(item_id, kind, enqueued_at, job_name, tuple(command)))
        seen_ids.add(item_id)
        seen_names.add(job_name)
    return sorted(items, key=lambda item: (item.enqueued_at, item.id))


def default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "items": {},
        "health": {
            kind: {"consecutive_failures": 0, "circuit_open_until": 0}
            for kind in KINDS
        },
    }


def load_state(path: Path) -> dict[str, Any]:
    state = default_state()
    if path.exists():
        loaded = json.loads(path.read_text())
        state.update(loaded)
        for kind in KINDS:
            state.setdefault("health", {}).setdefault(
                kind, {"consecutive_failures": 0, "circuit_open_until": 0}
            )
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def active_pool_counts(
    jobs: Iterable[QueueJob], items: Iterable[WorkItem]
) -> tuple[int, dict[str, int]]:
    kind_by_name = {item.job_name: item.kind for item in items}
    counts = {kind: 0 for kind in KINDS}
    total = 0
    for job in jobs:
        kind = kind_by_name.get(job.name)
        if kind and job.runnable:
            counts[kind] += 1
            total += 1
    return total, counts


def circuit_is_open(state: dict[str, Any], kind: str, now: int) -> bool:
    return int(state["health"][kind].get("circuit_open_until", 0)) > now


def select_admissions(
    pending: Iterable[WorkItem],
    *,
    global_active: int,
    pool_counts: dict[str, int],
    global_cap: int,
    pool_cap: int,
    quotas: dict[str, int],
    open_kinds: set[str] | None = None,
) -> list[WorkItem]:
    """Select oldest work under hard caps and soft, borrowable quotas."""
    open_kinds = open_kinds or set()
    ordered = [item for item in pending if item.kind not in open_kinds]
    free = min(max(0, global_cap - global_active), max(0, pool_cap - sum(pool_counts.values())))
    selected: list[WorkItem] = []
    counts = dict(pool_counts)

    # First honor the nominal split, preserving FCFS among eligible entries.
    for item in ordered:
        if len(selected) >= free:
            break
        if counts[item.kind] < quotas[item.kind]:
            selected.append(item)
            counts[item.kind] += 1

    # Then borrow every still-idle slot from the other kind, again FCFS.
    selected_ids = {item.id for item in selected}
    for item in ordered:
        if len(selected) >= free:
            break
        if item.id not in selected_ids:
            selected.append(item)
            selected_ids.add(item.id)
            counts[item.kind] += 1
    return sorted(selected, key=lambda item: (item.enqueued_at, item.id))


def adopt_live_jobs(
    state: dict[str, Any], items: Iterable[WorkItem], jobs: Iterable[QueueJob]
) -> None:
    """Adopt matching live jobs before considering any queue item for submit."""
    live_by_name: dict[str, QueueJob] = {}
    for job in jobs:
        if job.runnable:
            live_by_name.setdefault(job.name, job)
    for item in items:
        job = live_by_name.get(item.job_name)
        record = state["items"].get(item.id, {})
        if job and record.get("status") not in {"active", "done"}:
            state["items"][item.id] = {
                "status": "active",
                "attempts": max(1, int(record.get("attempts", 0))),
                "job_id": job.job_id,
                "adopted": True,
            }


def query_accounting(job_ids: Iterable[str]) -> dict[str, str]:
    ids = sorted(set(job_ids))
    if not ids:
        return {}
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", ",".join(ids), "-o", "JobIDRaw,State"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in ids:
            states[fields[0]] = fields[1].split("+")[0]
    return states


def reconcile(
    state: dict[str, Any],
    items: Iterable[WorkItem],
    jobs: Iterable[QueueJob],
    *,
    now: int,
    failure_threshold: int,
    circuit_seconds: int,
) -> None:
    live_ids = {job.job_id for job in jobs if job.runnable}
    active_records = {
        item.id: state["items"].get(item.id, {})
        for item in items
        if state["items"].get(item.id, {}).get("status") == "active"
    }
    accounting = query_accounting(
        record["job_id"]
        for record in active_records.values()
        if record.get("job_id") not in live_ids
    )
    by_id = {item.id: item for item in items}
    for item_id, record in active_records.items():
        job_id = record.get("job_id")
        terminal = accounting.get(job_id)
        if terminal in TERMINAL_SUCCESS:
            record.update(status="done", terminal_state=terminal, ended_at=now)
            state["health"][by_id[item_id].kind] = {
                "consecutive_failures": 0,
                "circuit_open_until": 0,
            }
        elif terminal in TERMINAL_FAILURE:
            record.update(status="failed", terminal_state=terminal, ended_at=now)
            health = state["health"][by_id[item_id].kind]
            health["consecutive_failures"] = int(health.get("consecutive_failures", 0)) + 1
            if health["consecutive_failures"] >= failure_threshold:
                health["circuit_open_until"] = now + circuit_seconds


def pending_items(
    items: Iterable[WorkItem], state: dict[str, Any], max_attempts: int
) -> list[WorkItem]:
    pending: list[WorkItem] = []
    for item in items:
        record = state["items"].get(item.id, {})
        if record.get("status") in {"active", "done"}:
            continue
        if int(record.get("attempts", 0)) >= max_attempts:
            record["status"] = "exhausted"
            state["items"][item.id] = record
            continue
        pending.append(item)
    return pending


def submit(item: WorkItem) -> tuple[str, str]:
    result = subprocess.run(
        list(item.command),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    matches = JOB_ID_RE.findall(result.stdout)
    if len(matches) != 1:
        raise RuntimeError(
            f"{item.id} command must submit exactly one Slurm job; output={result.stdout!r}"
        )
    return matches[0], result.stdout.strip()


def sweep(args: argparse.Namespace) -> dict[str, Any]:
    items = load_queue(args.queue)
    state = load_state(args.state)
    now = utc_now_epoch()
    jobs = query_queue(args.user)
    adopt_live_jobs(state, items, jobs)
    reconcile(
        state,
        items,
        jobs,
        now=now,
        failure_threshold=args.failure_threshold,
        circuit_seconds=args.circuit_seconds,
    )
    pool_total, pool_counts = active_pool_counts(jobs, items)
    open_kinds = {kind for kind in KINDS if circuit_is_open(state, kind, now)}
    selected = select_admissions(
        pending_items(items, state, args.max_attempts),
        global_active=runnable_job_count(jobs),
        pool_counts=pool_counts,
        global_cap=args.global_cap,
        pool_cap=args.pool_cap,
        quotas={"agentic": args.agentic_quota, "nonagentic": args.nonagentic_quota},
        open_kinds=open_kinds,
    )
    report: dict[str, Any] = {
        "global": {"active": runnable_job_count(jobs), "cap": args.global_cap},
        "pool": {"active": pool_total, "cap": args.pool_cap, **pool_counts},
        "circuits_open": sorted(open_kinds),
        "selected": [asdict(item) for item in selected],
        "submitted": [],
    }
    if args.submit:
        for item in selected:
            # Requery before every mutation so a concurrent external submission
            # cannot push this controller through the hard global ceiling.
            current = query_queue(args.user)
            current_pool, _ = active_pool_counts(current, items)
            if runnable_job_count(current) >= args.global_cap or current_pool >= args.pool_cap:
                break
            job_id, output = submit(item)
            old = state["items"].get(item.id, {})
            state["items"][item.id] = {
                "status": "active",
                "attempts": int(old.get("attempts", 0)) + 1,
                "job_id": job_id,
                "submitted_at": utc_now_epoch(),
            }
            report["submitted"].append({"id": item.id, "job_id": job_id, "output": output})
            save_state(args.state, state)
    save_state(args.state, state)
    return report


def parser() -> argparse.ArgumentParser:
    default_root = Path(os.environ.get("SCRATCH", "/tmp")) / "leaderboard_eval_pool"
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("once", "run", "status"), nargs="?", default="once")
    result.add_argument("--queue", type=Path, default=default_root / "queue.json")
    result.add_argument("--state", type=Path, default=default_root / "state.json")
    result.add_argument("--lock", type=Path, default=default_root / "controller.lock")
    result.add_argument("--user", default=os.environ.get("USER", ""))
    result.add_argument("--global-cap", type=int, default=20)
    result.add_argument("--pool-cap", type=int, default=12)
    result.add_argument("--agentic-quota", type=int, default=8)
    result.add_argument("--nonagentic-quota", type=int, default=4)
    result.add_argument("--failure-threshold", type=int, default=3)
    result.add_argument("--circuit-seconds", type=int, default=6 * 60 * 60)
    result.add_argument("--max-attempts", type=int, default=4)
    result.add_argument("--poll-seconds", type=int, default=300)
    result.add_argument("--submit", action="store_true", help="submit selected work; default is dry-run")
    return result


def main() -> int:
    args = parser().parse_args()
    if min(args.global_cap, args.pool_cap, args.agentic_quota, args.nonagentic_quota) < 0:
        raise SystemExit("caps and quotas must be non-negative")
    if args.agentic_quota + args.nonagentic_quota != args.pool_cap:
        raise SystemExit("agentic and nonagentic quotas must sum to pool cap")
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("w") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another leaderboard-pool controller holds the lock")
        while True:
            print(json.dumps(sweep(args), indent=2, sort_keys=True))
            if args.mode in {"once", "status"}:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
