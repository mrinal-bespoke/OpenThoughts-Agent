"""Scheduler boundary tests using real durable state and no scheduler processes."""

import json

import pytest

from eval.autoeval32k.queue import QueueSafetyError, reconcile_existing, submit_once


COMMAND = ["sbatch", "--parsable", "fresh.sbatch"]


def row(job_id="100", state="RUNNING", reason="None", **extra):
    return dict(job_id=job_id, state=state, reason=reason, **extra)


def launch(
    root, run_id="run-1", *, rows=None, output="123", max_nonheld=20, submit=None
):
    return submit_once(
        root,
        run_id,
        COMMAND,
        max_nonheld=max_nonheld,
        query_queue=lambda: [] if rows is None else rows,
        submit=submit or (lambda command: output),
    )


def test_intent_is_persisted_before_dispatch_and_receipt_replays_without_dispatch(
    tmp_path,
):
    calls = []

    def scheduler(command):
        intent = json.loads((tmp_path / "runs/run-1/intent.json").read_text())
        assert intent["command"] == COMMAND
        assert intent["run_id"] == "run-1"
        assert not (tmp_path / "runs/run-1/receipt.json").exists()
        calls.append(command)
        return "123;vista\n"

    receipt = launch(tmp_path, submit=scheduler)
    assert receipt["job_id"] == "123"
    assert launch(tmp_path, submit=scheduler) == receipt
    assert calls == [COMMAND]
    assert json.loads((tmp_path / "runs/run-1/receipt.json").read_text()) == receipt


def test_timeout_after_dispatch_blocks_retry_and_reserves_capacity(tmp_path):
    calls = []

    def timeout(command):
        calls.append(command)
        raise TimeoutError("Scheduler response lost after dispatch")

    with pytest.raises(TimeoutError):
        launch(tmp_path, submit=timeout)
    with pytest.raises(QueueSafetyError, match="uncertain"):
        launch(tmp_path, submit=timeout)
    with pytest.raises(QueueSafetyError, match="Capacity"):
        launch(tmp_path, "another", max_nonheld=1, submit=timeout)
    assert len(calls) == 1
    assert (tmp_path / "runs/run-1/intent.json").exists()
    assert not (tmp_path / "runs/run-1/receipt.json").exists()


@pytest.mark.parametrize("output", ["0", "-1", "123\n456", "warning\n123", "", None])
def test_ambiguous_receipt_preserves_uncertainty(tmp_path, output):
    with pytest.raises(QueueSafetyError):
        launch(tmp_path, output=output)
    with pytest.raises(QueueSafetyError, match="uncertain"):
        launch(tmp_path)


@pytest.mark.parametrize(
    "output", ["123", "123\n", "123;vista\n", "Submitted batch job 123\n"]
)
def test_supported_scheduler_receipt_formats(tmp_path, output):
    assert launch(tmp_path, output=output)["job_id"] == "123"


def test_banner_receipt_is_reconciled_from_exact_new_scheduler_identity(tmp_path):
    queue = []

    def dispatch(command):
        queue.append(row("123", run_id="run-1"))
        return "site notice\n123\n"

    receipt = submit_once(
        tmp_path, "run-1", COMMAND, query_queue=lambda: queue, submit=dispatch
    )
    assert receipt["job_id"] == "123" and receipt["reconciled"]
    assert (
        json.loads((tmp_path / "runs/run-1/submission-output.json").read_text())[
            "stdout"
        ]
        == "site notice\n123\n"
    )
    assert (
        launch(tmp_path, submit=lambda _: pytest.fail("must not dispatch again"))
        == receipt
    )


def test_two_matching_scheduler_records_cannot_resolve_receipt(tmp_path):
    queue = []

    def dispatch(command):
        queue.extend([row("123", run_id="run-1"), row("124", run_id="run-1")])
        return "site notice\n123\n"

    with pytest.raises(QueueSafetyError, match="ambiguous"):
        submit_once(
            tmp_path, "run-1", COMMAND, query_queue=lambda: queue, submit=dispatch
        )
    assert not (tmp_path / "runs/run-1/receipt.json").exists()


def test_existing_exact_name_blocks_an_unjournaled_duplicate(tmp_path):
    with pytest.raises(QueueSafetyError, match="already exists"):
        launch(
            tmp_path,
            rows=[row("123", run_id="run-1")],
            submit=lambda _: pytest.fail("must not dispatch"),
        )


def test_same_identity_cannot_be_reused_for_a_changed_command(tmp_path):
    launch(tmp_path)
    with pytest.raises(QueueSafetyError, match="another command"):
        submit_once(
            tmp_path,
            "run-1",
            ["sbatch", "different"],
            query_queue=lambda: [],
            submit=lambda _: "124",
        )


def test_queue_visibility_lag_keeps_receipt_reserved(tmp_path):
    launch(tmp_path, max_nonheld=1)
    with pytest.raises(QueueSafetyError, match="Capacity"):
        launch(tmp_path, "run-2", max_nonheld=1)


def test_visible_receipt_is_not_double_counted(tmp_path):
    launch(tmp_path)
    receipt = launch(tmp_path, "run-2", rows=[row("123")], output="124", max_nonheld=2)
    assert receipt["job_id"] == "124"


@pytest.mark.parametrize(
    "state,reason,allowed",
    [
        ("PENDING", "JobHeldUser", True),
        ("PENDING", "JobHeldAdmin", True),
        ("PENDING", "DependencyNeverSatisfied", False),
        ("PENDING", "Priority", False),
        ("RUNNING", "JobHeldUser", False),
        ("COMPLETING", "None", False),
    ],
)
def test_only_pending_explicitly_held_jobs_are_excluded(
    tmp_path, state, reason, allowed
):
    kwargs = dict(rows=[row(state=state, reason=reason)], max_nonheld=1)
    if allowed:
        assert launch(tmp_path, **kwargs)["job_id"] == "123"
    else:
        with pytest.raises(QueueSafetyError, match="Capacity"):
            launch(tmp_path, **kwargs)
        assert not (tmp_path / "runs/run-1").exists()


@pytest.mark.parametrize(
    "rows",
    [
        {},
        [None],
        [row(job_id="1_2")],
        [row(job_id=123)],
        [row(state="UNKNOWN")],
        [row(reason=None)],
        [row(), row()],
    ],
)
def test_malformed_or_duplicate_queue_fails_before_dispatch(tmp_path, rows):
    with pytest.raises(QueueSafetyError):
        launch(tmp_path, rows=rows)
    assert not (tmp_path / "runs/run-1").exists()


def test_second_writer_is_blocked_while_scheduler_callback_runs(tmp_path):
    def scheduler(command):
        with pytest.raises(QueueSafetyError, match="lock"):
            launch(tmp_path, "run-2", output="124")
        return "123"

    launch(tmp_path, submit=scheduler)
    assert not (tmp_path / "runs/run-2").exists()
    assert not (tmp_path / ".writer.lock").exists()


def test_stale_lock_is_preserved(tmp_path):
    (tmp_path / ".writer.lock").mkdir()
    with pytest.raises(QueueSafetyError, match="lock"):
        launch(tmp_path)
    assert (tmp_path / ".writer.lock").is_dir()


@pytest.mark.parametrize("corruption", ["intent", "receipt", "terminal", "orphan"])
def test_corrupt_state_blocks_all_new_submissions(tmp_path, corruption):
    launch(tmp_path)
    if corruption == "orphan":
        (tmp_path / "runs/orphan").mkdir()
    else:
        (tmp_path / f"runs/run-1/{corruption}.json").write_text("{incomplete")
    with pytest.raises(QueueSafetyError):
        launch(tmp_path, "run-2", output="124")
    assert not (tmp_path / "runs/run-2").exists()


def test_known_job_id_cannot_be_assigned_to_second_run(tmp_path):
    launch(tmp_path)
    with pytest.raises(QueueSafetyError, match="already known"):
        launch(tmp_path, "run-2")
    assert (tmp_path / "runs/run-2/intent.json").exists()
    assert not (tmp_path / "runs/run-2/receipt.json").exists()


def test_exact_terminal_reconciliation_releases_capacity_without_rerun(tmp_path):
    with pytest.raises(QueueSafetyError):
        launch(tmp_path, output="lost response")
    evidence = [row("123", "COMPLETED", run_id="run-1")]
    receipt = reconcile_existing(tmp_path, "run-1", "123", evidence)
    assert receipt["reconciled"] is True
    assert reconcile_existing(tmp_path, "run-1", "123", evidence) == receipt
    assert launch(tmp_path, "run-2", output="124", max_nonheld=1)["job_id"] == "124"
    assert (
        json.loads((tmp_path / "runs/run-1/terminal.json").read_text())["state"]
        == "COMPLETED"
    )


@pytest.mark.parametrize(
    "evidence",
    [
        [],
        [row("123", run_id="wrong")],
        [row("124", run_id="run-1")],
        [row("123", run_id="run-1"), row("124", run_id="run-1")],
    ],
)
def test_uncertain_reconciliation_never_resolves_intent(tmp_path, evidence):
    with pytest.raises(QueueSafetyError):
        launch(tmp_path, output="lost response")
    with pytest.raises(QueueSafetyError):
        reconcile_existing(tmp_path, "run-1", "123", evidence)
    assert not (tmp_path / "runs/run-1/receipt.json").exists()


def test_terminal_evidence_conflicting_with_live_queue_fails_closed(tmp_path):
    launch(tmp_path)
    reconcile_existing(
        tmp_path, "run-1", "123", [row("123", "COMPLETED", run_id="run-1")]
    )
    with pytest.raises(QueueSafetyError, match="conflicts"):
        launch(tmp_path, "run-2", rows=[row("123")], output="124")


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "", ".", "..", "a/b"])
def test_run_identity_cannot_escape_state_root(tmp_path, run_id):
    with pytest.raises(QueueSafetyError):
        launch(tmp_path, run_id)


def test_symlinked_state_is_not_followed(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(QueueSafetyError, match="symlink"):
        launch(link)
    assert not list(real.iterdir())
