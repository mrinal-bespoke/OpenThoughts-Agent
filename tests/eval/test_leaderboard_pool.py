import json

import pytest

from eval.tacc.campaign import leaderboard_pool
from eval.tacc.campaign.leaderboard_pool import WorkItem, load_queue, select_admissions
from eval.tacc.campaign.priority_backfill import QueueJob


def item(number: int, kind: str) -> WorkItem:
    return WorkItem(
        id=f"item-{number:02d}",
        kind=kind,
        enqueued_at=f"2026-09-01T00:{number:02d}:00Z",
        job_name=f"lb-{number:02d}",
        command=("sbatch", f"job-{number}.sbatch"),
    )


def choose(pending, *, global_active=0, counts=None, open_kinds=None):
    return select_admissions(
        pending,
        global_active=global_active,
        pool_counts=counts or {"agentic": 0, "nonagentic": 0},
        global_cap=20,
        pool_cap=12,
        quotas={"agentic": 8, "nonagentic": 4},
        open_kinds=open_kinds,
    )


def test_nominal_split_is_eight_and_four():
    selected = choose(
        [item(n, "agentic") for n in range(12)]
        + [item(n + 20, "nonagentic") for n in range(12)]
    )
    assert sum(row.kind == "agentic" for row in selected) == 8
    assert sum(row.kind == "nonagentic" for row in selected) == 4


def test_agentic_borrows_all_unused_nonagentic_slots():
    selected = choose([item(n, "agentic") for n in range(15)])
    assert len(selected) == 12
    assert {row.kind for row in selected} == {"agentic"}


def test_nonagentic_borrows_when_agentic_circuit_is_open():
    pending = [item(n, "agentic") for n in range(6)] + [
        item(n + 20, "nonagentic") for n in range(15)
    ]
    selected = choose(pending, open_kinds={"agentic"})
    assert len(selected) == 12
    assert {row.kind for row in selected} == {"nonagentic"}


def test_global_cap_limits_selection_and_preserves_fcfs():
    pending = [item(n, "agentic") for n in range(12)]
    selected = choose(pending, global_active=18)
    assert [row.id for row in selected] == ["item-00", "item-01"]


def test_existing_pool_jobs_reduce_each_nominal_allowance():
    pending = [item(n, "agentic") for n in range(12)] + [
        item(n + 20, "nonagentic") for n in range(12)
    ]
    selected = choose(pending, counts={"agentic": 7, "nonagentic": 3})
    assert len(selected) == 2
    assert [row.kind for row in selected] == ["agentic", "nonagentic"]


def test_live_job_with_matching_name_is_adopted_not_resubmitted():
    row = item(1, "agentic")
    state = leaderboard_pool.default_state()
    jobs = [QueueJob("123", row.job_name, "RUNNING", "node", "gh")]
    leaderboard_pool.adopt_live_jobs(state, [row], jobs)
    assert state["items"][row.id] == {
        "status": "active",
        "attempts": 1,
        "job_id": "123",
        "adopted": True,
    }
    assert leaderboard_pool.pending_items([row], state, max_attempts=4) == []


def test_queue_rejects_shell_command_string(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            [{
                "id": "x",
                "kind": "agentic",
                "enqueued_at": "2026-09-01T00:00:00Z",
                "job_name": "lb-x",
                "command": "sbatch x.sbatch",
            }]
        )
    )
    with pytest.raises(ValueError, match="argv list"):
        load_queue(queue)


def test_queue_loader_orders_oldest_first(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            [
                {**as_json(item(2, "agentic"))},
                {**as_json(item(1, "nonagentic"))},
            ]
        )
    )
    assert [row.id for row in load_queue(queue)] == ["item-01", "item-02"]


def test_three_terminal_failures_open_only_that_kind_circuit(monkeypatch):
    items = [item(n, "agentic") for n in range(3)]
    state = leaderboard_pool.default_state()
    for n, row in enumerate(items):
        state["items"][row.id] = {"status": "active", "job_id": str(100 + n)}
    monkeypatch.setattr(
        leaderboard_pool,
        "query_accounting",
        lambda job_ids: {str(100 + n): "FAILED" for n in range(3)},
    )
    leaderboard_pool.reconcile(
        state,
        items,
        [],
        now=1000,
        failure_threshold=3,
        circuit_seconds=600,
    )
    assert state["health"]["agentic"]["circuit_open_until"] == 1600
    assert state["health"]["nonagentic"]["circuit_open_until"] == 0


def test_later_success_resets_kind_health(monkeypatch):
    row = item(1, "agentic")
    state = leaderboard_pool.default_state()
    state["health"]["agentic"] = {
        "consecutive_failures": 3,
        "circuit_open_until": 1600,
    }
    state["items"][row.id] = {"status": "active", "job_id": "101"}
    monkeypatch.setattr(
        leaderboard_pool, "query_accounting", lambda job_ids: {"101": "COMPLETED"}
    )
    leaderboard_pool.reconcile(
        state,
        [row],
        [],
        now=1200,
        failure_threshold=3,
        circuit_seconds=600,
    )
    assert state["health"]["agentic"] == {
        "consecutive_failures": 0,
        "circuit_open_until": 0,
    }


def as_json(row: WorkItem):
    return {
        "id": row.id,
        "kind": row.kind,
        "enqueued_at": row.enqueued_at,
        "job_name": row.job_name,
        "command": list(row.command),
    }
