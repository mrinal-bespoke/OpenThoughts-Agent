"""Fail-closed, write-ahead scheduler receipts; no scheduler or network imports.

Callbacks are the only scheduler boundary. An uncertain dispatch is never retried.
Missing queue entries remain reserved until explicit scheduler reconciliation.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Callable


class QueueSafetyError(RuntimeError):
    """State or scheduler evidence is insufficient to submit safely."""


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
_ACTIVE = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESV_DEL_HOLD",
    "SIGNALING",
    "SPECIAL_EXIT",
    "STAGE_OUT",
    "STOPPED",
}
_TERMINAL = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
_HELD = {"JobHeldUser", "JobHeldAdmin"}


def _run_id(value: str) -> str:
    if not isinstance(value, str) or not _RUN_ID.fullmatch(value):
        raise QueueSafetyError("Invalid run_id")
    return value


def _job_id(value: object) -> str:
    if not isinstance(value, str) or not _JOB_ID.fullmatch(value):
        raise QueueSafetyError("Expected a positive numeric scheduler job ID")
    return value


def _command(value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(part, str) or not part or "\x00" in part for part in value
        )
    ):
        raise QueueSafetyError("Expected a nonempty command argument list")
    return value


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path: Path, data: dict) -> None:
    # Exclusive creation means partial writes remain visible and fail closed.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)


def _read(path: Path) -> dict:
    if path.is_symlink():
        raise QueueSafetyError(f"Symlinked scheduler state: {path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QueueSafetyError(f"Unreadable scheduler state: {path.name}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise QueueSafetyError(f"Invalid scheduler state: {path.name}")
    return data


@contextmanager
def _locked(root: Path):
    root = Path(root)
    if root.is_symlink():
        raise QueueSafetyError("Scheduler state root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise QueueSafetyError(
            "Writer lock exists; never clear it automatically"
        ) from exc
    try:
        _sync_dir(root)
        runs = root / "runs"
        if runs.is_symlink():
            raise QueueSafetyError("Scheduler runs directory must not be a symlink")
        runs.mkdir(exist_ok=True)
        _sync_dir(root)
        yield runs
    finally:
        lock.rmdir()
        _sync_dir(root)


def _rows(rows: object) -> dict[str, dict]:
    if not isinstance(rows, list):
        raise QueueSafetyError("Queue response must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise QueueSafetyError("Malformed queue row")
        jid = _job_id(row.get("job_id"))
        if jid in result:
            raise QueueSafetyError("Duplicate scheduler job ID")
        if row.get("state") not in _ACTIVE | _TERMINAL or not isinstance(
            row.get("reason"), str
        ):
            raise QueueSafetyError("Unknown scheduler state or malformed reason")
        result[jid] = row
    return result


def _load_runs(runs: Path) -> dict[str, tuple[dict, dict | None, dict | None]]:
    result = {}
    ids = set()
    for directory in sorted(runs.iterdir()):
        name = _run_id(directory.name)
        if directory.is_symlink() or not directory.is_dir():
            raise QueueSafetyError("Invalid run state directory")
        intent = _read(directory / "intent.json")
        if intent.get("run_id") != name:
            raise QueueSafetyError("Intent identity mismatch")
        _command(intent.get("command"))
        receipt_path = directory / "receipt.json"
        receipt = (
            _read(receipt_path)
            if receipt_path.exists() or receipt_path.is_symlink()
            else None
        )
        terminal_path = directory / "terminal.json"
        terminal = (
            _read(terminal_path)
            if terminal_path.exists() or terminal_path.is_symlink()
            else None
        )
        if receipt:
            jid = _job_id(receipt.get("job_id"))
            if receipt.get("run_id") != name or jid in ids:
                raise QueueSafetyError("Duplicate or mismatched receipt identity")
            ids.add(jid)
        if terminal:
            if (
                not receipt
                or terminal.get("run_id") != name
                or terminal.get("job_id") != receipt["job_id"]
            ):
                raise QueueSafetyError("Terminal evidence identity mismatch")
            if terminal.get("state") not in _TERMINAL:
                raise QueueSafetyError("Invalid terminal evidence")
        result[name] = intent, receipt, terminal
    return result


def submit_once(
    state_root: Path,
    run_id: str,
    command: list[str],
    *,
    max_nonheld: int = 20,
    query_queue: Callable[[], list[dict]],
    submit: Callable[[list[str]], str],
) -> dict:
    """Submit once after capacity validation; replay only an existing receipt.

    query_queue must cover ALL of the user's jobs, not only this campaign.
    submit must use a bounded invocation; an exception leaves durable uncertainty.
    Commands must contain no credentials: they are retained as non-secret evidence.
    """
    _run_id(run_id)
    command = list(_command(command))
    if type(max_nonheld) is not int or max_nonheld < 1:
        raise QueueSafetyError("max_nonheld must be a positive integer")
    with _locked(state_root) as runs:
        existing = _load_runs(runs)
        if run_id in existing:
            intent, receipt, _ = existing[run_id]
            if intent["command"] != command:
                raise QueueSafetyError("Run ID already belongs to another command")
            if receipt:
                return receipt
            raise QueueSafetyError(
                "Dispatch outcome uncertain; reconcile, never resubmit"
            )
        queue = _rows(query_queue())
        if any(row.get("run_id") == run_id for row in queue.values()):
            raise QueueSafetyError(
                "Matching scheduler job already exists; reconcile first"
            )
        count = sum(
            row["state"] in _ACTIVE
            and not (row["state"] == "PENDING" and row["reason"] in _HELD)
            for row in queue.values()
        )
        for _, receipt, terminal in existing.values():
            if terminal:
                if (
                    receipt["job_id"] in queue
                    and queue[receipt["job_id"]]["state"] in _ACTIVE
                ):
                    raise QueueSafetyError(
                        "Terminal evidence conflicts with live queue"
                    )
                continue
            if receipt is None or receipt["job_id"] not in queue:
                count += 1
        if count >= max_nonheld:
            raise QueueSafetyError(
                f"Capacity exhausted: {count}/{max_nonheld} reserved nonheld jobs"
            )
        directory = runs / run_id
        directory.mkdir()
        _sync_dir(runs)
        intent = {"schema_version": 1, "run_id": run_id, "command": command}
        _write_new(directory / "intent.json", intent)
        output = submit(list(command))
        if not isinstance(output, str):
            raise QueueSafetyError("Scheduler returned non-text receipt")
        _write_new(
            directory / "submission-output.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "stdout": output,
            },
        )
        match = re.fullmatch(
            r"(?:Submitted batch job )?([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*", output
        )
        reconciled = False
        if match:
            jid = match[1]
        else:
            # Site banners may accompany --parsable output. Do not guess an ID
            # from arbitrary text: require one new exact-name scheduler record.
            after = _rows(query_queue())
            candidates = [row for row in after.values() if row.get("run_id") == run_id]
            if len(candidates) != 1:
                raise QueueSafetyError(
                    "Scheduler returned an ambiguous receipt; do not retry"
                )
            jid = candidates[0]["job_id"]
            reconciled = True
        if jid in queue or any(
            receipt and receipt["job_id"] == jid for _, receipt, _ in existing.values()
        ):
            raise QueueSafetyError(
                "Scheduler returned an already known job ID; do not retry"
            )
        receipt = {"schema_version": 1, "run_id": run_id, "job_id": jid}
        if reconciled:
            receipt["reconciled"] = True
        _write_new(directory / "receipt.json", receipt)
        return receipt


def reconcile_existing(
    state_root: Path, run_id: str, job_id: str, evidence: list[dict]
) -> dict:
    """Record exact scheduler evidence without ever issuing a new submission.

    Caller supplies authoritative scheduler rows filtered by exact run identity.
    Exactly one row, with exact run_id and job_id, is required. Terminal evidence
    releases a reservation; missing/ambiguous scheduler evidence never does.
    """
    _run_id(run_id)
    _job_id(job_id)
    rows = _rows(evidence)
    if len(rows) != 1 or job_id not in rows or rows[job_id].get("run_id") != run_id:
        raise QueueSafetyError(
            "Reconciliation requires one exact matching scheduler record"
        )
    row = rows[job_id]
    with _locked(state_root) as runs:
        existing = _load_runs(runs)
        if run_id not in existing:
            raise QueueSafetyError("Cannot reconcile a run without a dispatch intent")
        _, receipt, terminal = existing[run_id]
        if receipt and receipt["job_id"] != job_id:
            raise QueueSafetyError("Reconciliation conflicts with existing receipt")
        if any(
            name != run_id and other and other["job_id"] == job_id
            for name, (_, other, _) in existing.items()
        ):
            raise QueueSafetyError("Job ID already belongs to another run")
        if terminal and terminal["state"] != row["state"]:
            raise QueueSafetyError("Reconciliation conflicts with terminal evidence")
        directory = runs / run_id
        if receipt is None:
            receipt = {
                "schema_version": 1,
                "run_id": run_id,
                "job_id": job_id,
                "reconciled": True,
            }
            _write_new(directory / "receipt.json", receipt)
        if row["state"] in _TERMINAL and terminal is None:
            _write_new(
                directory / "terminal.json",
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "job_id": job_id,
                    "state": row["state"],
                    "reason": row["reason"],
                },
            )
        return receipt
