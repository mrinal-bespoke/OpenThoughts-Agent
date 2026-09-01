import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import eval.tacc.campaign.priority_backfill as priority_backfill
from eval.tacc.campaign.priority_backfill import (
    Controller,
    QueueJob,
    available_submission_slots,
    matching_priority_jobs,
    owned_runnable_job_count,
    parse_squeue,
    pending_models,
    priority_pause_remaining,
    runnable_job_count,
    select_benchmark,
    submitted_job_ids,
)
from scripts.database.reconcile_flawed_summ_state import priority_tier


def test_parse_queue_and_exclude_held_jobs_from_capacity():
    jobs = parse_squeue(
        "1|snowball-r4|PENDING|Priority|gh\n"
        "2|eval_a1-rust_swe|PENDING|JobHeldUser|gh\n"
        "3|eval_a1-bash_swe|RUNNING|c123|gh\n"
    )

    assert jobs[0] == QueueJob("1", "snowball-r4", "PENDING", "Priority", "gh")
    assert runnable_job_count(jobs) == 2


def test_campaign_cap_counts_only_owned_runnable_jobs():
    jobs = [
        QueueJob("1", "eval_a3-one", "RUNNING", "node-a", "gh"),
        QueueJob("2", "lb-agentic-two", "RUNNING", "node-b", "gh"),
        QueueJob("3", "eval_a3-held", "PENDING", "JobHeldUser", "gh"),
    ]

    assert owned_runnable_job_count(jobs, {"1", "3"}) == 1
    assert available_submission_slots(
        jobs,
        {"1", "3"},
        global_cap=20,
        campaign_cap=8,
    ) == 7


def test_campaign_cap_and_global_cap_both_apply():
    campaign_jobs = [
        QueueJob(str(i), f"eval_a3-{i}", "RUNNING", f"node-{i}", "gh")
        for i in range(1, 9)
    ]
    unrelated = [
        QueueJob(str(i), f"lb-agentic-{i}", "RUNNING", f"node-{i}", "gh")
        for i in range(9, 20)
    ]

    assert available_submission_slots(
        campaign_jobs + unrelated,
        {str(i) for i in range(1, 9)},
        global_cap=20,
        campaign_cap=8,
    ) == 0
    assert available_submission_slots(
        campaign_jobs[:4] + unrelated,
        {str(i) for i in range(1, 9)},
        global_cap=20,
        campaign_cap=8,
    ) == 4


def test_omitting_campaign_cap_preserves_global_cap_only_behavior():
    jobs = [
        QueueJob(str(i), f"job-{i}", "RUNNING", f"node-{i}", "gh")
        for i in range(1, 7)
    ]

    assert available_submission_slots(
        jobs,
        {"1", "2"},
        global_cap=20,
        campaign_cap=None,
    ) == 14


def test_guard_pauses_while_matching_job_is_live_and_for_quiet_window():
    jobs = [QueueJob("1", "snowball-r4", "PENDING", "Priority", "gh")]
    state = {"last_priority_seen_at": None}
    guards = {"snowball": {"pattern": "snowball|grug"}}
    matches = matching_priority_jobs(jobs, guards)

    assert set(matches) == {"snowball"}
    assert priority_pause_remaining(state, matches, now=1000, quiet_seconds=1800) == (
        True,
        1800,
    )
    assert state["last_priority_seen_at"] == 1000
    assert priority_pause_remaining(state, {}, now=2000, quiet_seconds=1800) == (
        True,
        800,
    )
    assert priority_pause_remaining(state, {}, now=2800, quiet_seconds=1800) == (
        False,
        0,
    )


def test_pending_models_put_below_gate_warm_resumes_first():
    summary = {
        "rows": [
            {
                "row": "r9",
                "model": "DCAgent/a1-new",
                "benchmark": "swebench-verified-random-100-folders",
                "state": "pending",
                "reason": "no owned gate-passing eval",
            },
            {
                "row": "r10",
                "model": "DCAgent/a1-resume",
                "benchmark": "swebench-verified-random-100-folders",
                "state": "pending",
                "reason": "valid-complete 270/300 below 90%",
            },
            {
                "row": "r11",
                "model": "DCAgent/a3-not-yet",
                "benchmark": "swebench-verified-random-100-folders",
                "state": "pending",
                "reason": "valid-complete 270/300 below 90%",
            },
        ]
    }

    assert pending_models(summary, "swebench") == [
        "DCAgent/a1-resume",
        "DCAgent/a1-new",
    ]
    assert pending_models(summary, "swebench", "a3") == [
        "DCAgent/a3-not-yet"
    ]


def test_benchmark_advances_while_previous_family_is_active():
    state = {"benchmark": "swebench", "benchmark_order": ["swebench", "v2", "tb2"]}
    summary = {
        "rows": [
            {
                "row": "r1",
                "model": "DCAgent/a1-active",
                "benchmark": "swebench-verified-random-100-folders",
                "state": "slurm-active-unverified",
            },
            {
                "row": "r2",
                "model": "DCAgent/a1-next",
                "benchmark": "dev_set_v2",
                "state": "pending",
                "reason": "no owned gate-passing eval",
            },
        ]
    }

    assert select_benchmark(state, summary) == "v2"
    assert state["benchmark"] == "v2"


def test_benchmark_keeps_current_pending_family_while_off_family_is_active():
    state = {"benchmark": "swebench", "benchmark_order": ["swebench", "v2", "tb2"]}
    summary = {
        "rows": [
            {
                "row": "r1",
                "model": "DCAgent/a1-current",
                "benchmark": "swebench-verified-random-100-folders",
                "state": "pending",
                "reason": "no owned gate-passing eval",
            },
            {
                "row": "r2",
                "model": "DCAgent/a1-off-family",
                "benchmark": "dev_set_v2",
                "state": "slurm-active-unverified",
            },
        ]
    }

    assert select_benchmark(state, summary) == "swebench"


def test_controller_advances_to_a3_while_completed_a1_jobs_drain():
    state = {
        "benchmark": "v2",
        "benchmark_order": ["swebench", "v2", "tb2"],
        "tier": "a1",
        "tier_order": ["a1", "a3"],
    }
    summary = {
        "rows": [
            {
                "row": "r1",
                "model": "DCAgent/a1-draining",
                "benchmark": "dev_set_v2",
                "state": "slurm-active-unverified",
            },
            {
                "row": "r2",
                "model": "DCAgent/a3-next",
                "benchmark": "dev_set_v2",
                "state": "pending",
                "reason": "no owned gate-passing eval",
            },
        ]
    }

    assert select_benchmark(state, summary) == "v2"
    assert state["tier"] == "a3"


def test_controller_returns_to_a1_when_a1_work_reappears():
    state = {
        "benchmark": "v2",
        "benchmark_order": ["swebench", "v2", "tb2"],
        "tier": "a3",
        "tier_order": ["a1", "a3"],
    }
    summary = {
        "rows": [
            {
                "row": "r1",
                "model": "DCAgent/a3-current",
                "benchmark": "dev_set_v2",
                "state": "pending",
                "reason": "no owned gate-passing eval",
            },
            {
                "row": "r2",
                "model": "DCAgent/a1-repair",
                "benchmark": "terminal_bench_2",
                "state": "pending",
                "reason": "valid-complete 230/267 below 90%",
            },
        ]
    }

    assert select_benchmark(state, summary) == "tb2"
    assert state["tier"] == "a1"


def test_listener_job_ids_are_recorded():
    output = "Submitted batch job 123\n[listener] -> Submitted 124\n"
    assert submitted_job_ids(output) == ["123", "124"]


def test_priority_tier_uses_leading_tier_for_derived_model_names():
    assert priority_tier("DCAgent/g1_a1_top16_from_rollouts") == "g1"
    assert priority_tier("DCAgent/a3-rl-SankalpKJ") == "a3"
    assert priority_tier("penfever/a1-stack_bash") == "a1"


def test_successful_hold_is_persisted_before_reconciliation_error(
    tmp_path: Path, monkeypatch
):
    args = SimpleNamespace(
        repo=str(tmp_path),
        state_file=str(tmp_path / "state.json"),
        dry_run=False,
        user="mkumar73",
    )
    controller = Controller(args)
    controller.state["owned_job_ids"] = ["123"]

    monkeypatch.setattr(
        priority_backfill,
        "run_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, ""),
    )

    def fail_to_reconcile(user: str):
        raise subprocess.CalledProcessError(1, ["squeue", user])

    monkeypatch.setattr(priority_backfill, "query_queue", fail_to_reconcile)
    controller.hold_owned_pending(
        [QueueJob("123", "eval_a3", "PENDING", "Priority", "gh")]
    )

    persisted = json.loads((tmp_path / "state.json").read_text())
    assert persisted["held_job_ids"] == ["123"]


def test_hold_reconciles_an_untracked_existing_user_hold(tmp_path: Path, monkeypatch):
    args = SimpleNamespace(
        repo=str(tmp_path),
        state_file=str(tmp_path / "state.json"),
        dry_run=False,
        user="mkumar73",
    )
    controller = Controller(args)
    controller.state["owned_job_ids"] = ["123"]
    held_job = QueueJob("123", "eval_a3", "PENDING", "JobHeldUser", "gh")
    monkeypatch.setattr(priority_backfill, "query_queue", lambda user: [held_job])

    controller.hold_owned_pending([held_job])

    assert controller.state["held_job_ids"] == ["123"]


def test_failed_release_retains_job_that_scheduler_reports_held(
    tmp_path: Path, monkeypatch
):
    args = SimpleNamespace(
        repo=str(tmp_path),
        state_file=str(tmp_path / "state.json"),
        dry_run=False,
        user="mkumar73",
    )
    controller = Controller(args)
    controller.state["owned_job_ids"] = ["123"]
    controller.state["held_job_ids"] = ["123"]
    held_job = QueueJob("123", "eval_a3", "PENDING", "JobHeldUser", "gh")
    monkeypatch.setattr(
        priority_backfill,
        "run_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "still held"),
    )
    monkeypatch.setattr(priority_backfill, "query_queue", lambda user: [held_job])

    controller.release_owned()

    assert controller.state["held_job_ids"] == ["123"]
