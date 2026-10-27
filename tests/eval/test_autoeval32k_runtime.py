"""Fresh execution boundaries and real filesystem evidence, without cluster calls."""

import json
from pathlib import Path

import pytest

from eval.autoeval32k.__main__ import main, validate_settings, materialize, plan
from eval.autoeval32k.protocol import Manifest, tree_hash, render_harbor
from eval.autoeval32k.runtime import (
    stage_tasks,
    normalize_harbor,
    run,
    read_daytona_key,
    validate_vllm_versions,
)
from tests.eval import test_autoeval32k


def test_cuda_wheel_pin_checks_distribution_and_import_version():
    validate_vllm_versions(
        "0.20.1rc1.dev0+g506029539.cu128",
        {
            "vllm_distribution": "0.20.1rc1.dev0+g506029539.cu128",
            "vllm": "0.20.1rc1.dev0+g506029539",
        },
    )


@pytest.mark.parametrize(
    "observed",
    [
        {
            "vllm_distribution": "0.20.1rc1.dev0+g506029539.cu129",
            "vllm": "0.20.1rc1.dev0+g506029539",
        },
        {
            "vllm_distribution": "0.20.1rc1.dev0+g506029539.cu128",
            "vllm": "0.20.1rc1.dev0+gOTHER",
        },
        {"vllm_distribution": "0.20.1rc1.dev0+g506029539.cu128", "vllm": "dev"},
        {"vllm": "0.20.1rc1.dev0+g506029539"},
    ],
)
def test_wrong_or_unproven_vllm_build_rejects(observed):
    with pytest.raises(ValueError, match="vLLM"):
        validate_vllm_versions("0.20.1rc1.dev0+g506029539.cu128", observed)


@pytest.fixture
def manifest_data():
    return test_autoeval32k.example_manifest()


@pytest.fixture
def settings(tmp_path):
    return {
        "cluster": "tacc",
        "account": "TEST",
        "partition": "gh",
        "python": "/runtime/python",
        "serve_python": "/serving/python",
        "repo_root": str(tmp_path / "repo"),
        "harbor_root": str(tmp_path / "harbor"),
        "run_root": str(tmp_path / "runs"),
        "state_root": str(tmp_path / "state"),
        "walltime_seconds": 1800,
        "concurrency": 1,
    }


def test_dry_run_does_not_create_control_or_run_directories(
    manifest_data, settings, tmp_path, capsys
):
    manifest = tmp_path / "manifest.json"
    deployment = tmp_path / "settings.json"
    manifest.write_text(json.dumps(manifest_data))
    deployment.write_text(json.dumps(settings))
    assert main(["--manifest", str(manifest), "--settings", str(deployment)]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["run_id"] == Manifest.parse(manifest_data).run_id
    assert not Path(settings["run_root"]).exists()
    assert not Path(settings["state_root"]).exists()
    assert not any(
        arg.startswith(("--gres", "--mem", "--cpus")) for arg in rendered["command"]
    )
    assert "--no-requeue" in rendered["command"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cluster", "jupiter"),
        ("concurrency", 2),
        ("walltime_seconds", 0),
        ("account", "x\n--bad"),
        ("python", "relative"),
    ],
)
def test_unsupported_deployment_is_not_submitted(settings, field, value):
    settings[field] = value
    with pytest.raises(ValueError):
        validate_settings(settings)


def test_staging_never_changes_source_and_rejects_drift(manifest_data, tmp_path):
    task = tmp_path / "frozen"
    task.mkdir()
    (task / "instruction.md").write_text("Return OK")
    manifest_data["tasks"][0].update(path=str(task), sha256=tree_hash(task))
    manifest = Manifest.parse(manifest_data)
    staged = stage_tasks(manifest, tmp_path / "new")
    target = Path(staged[0]["path"])
    (target / "instruction.md").write_text("Only this copy changed")
    assert (task / "instruction.md").read_text() == "Return OK"
    (task / "instruction.md").write_text("drift")
    with pytest.raises(ValueError, match="changed"):
        stage_tasks(manifest, tmp_path / "another")


def test_symlink_task_is_not_followed(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "escape").symlink_to("/etc/hosts")
    with pytest.raises(ValueError, match="symlink"):
        tree_hash(task)


def test_existing_run_is_never_resumed(manifest_data, settings, tmp_path):
    manifest = Manifest.parse(manifest_data)
    old = Path(settings["run_root"]) / manifest.run_id
    old.mkdir(parents=True)
    original = old / "score.json"
    original.write_text('{"reward":0}')
    with pytest.raises(FileExistsError):
        run(manifest, settings, old)
    assert original.read_text() == '{"reward":0}'


def raw_trial(manifest, root, reward):
    trial = root / "trials" / manifest.run_id / "hello__unique" / "attempts" / "000"
    trial.mkdir(parents=True)
    cfg = render_harbor(manifest, root)
    cfg["tasks"] = [{"path": str(root / "tasks" / "hello")}]
    (root / "harbor-config.json").write_text(json.dumps(cfg))
    data = {
        "task_name": "hello",
        "finished_at": "2026-10-27T00:00:00Z",
        "config": {
            "task": cfg["tasks"][0],
            "timeout_multiplier": 1.0,
            "agent": cfg["agents"][0],
            "environment": cfg["environment"],
            "verifier": cfg["verifier"],
        },
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": {"exception_type": "AgentTimeoutError"},
    }
    (trial / "result.json").write_text(json.dumps(data))
    (trial / "verifier").mkdir()
    (trial / "verifier" / "reward.txt").write_text(str(reward))
    return trial


def staged_manifest(data, root):
    target = root / "tasks" / "hello"
    target.mkdir(parents=True)
    (target / "instruction.md").write_text("Synthetic task")
    data["tasks"][0]["sha256"] = tree_hash(target)
    return Manifest.parse(data)


def test_report_cross_checks_raw_score_and_cleanup(manifest_data, tmp_path):
    manifest = staged_manifest(manifest_data, tmp_path)
    trial = raw_trial(manifest, tmp_path, 0)
    raw_before = (trial / "result.json").read_bytes()
    (trial / "sandbox-create-intent.json").write_text("{}")
    report = normalize_harbor(manifest, tmp_path)
    assert report["numeric"] == 1 and report["zero"] == 1
    assert report["cleanup_complete"] is False
    assert (trial / "result.json").read_bytes() == raw_before


def test_reward_disagreement_stops_audit(manifest_data, tmp_path):
    manifest = staged_manifest(manifest_data, tmp_path)
    trial = raw_trial(manifest, tmp_path, 1)
    (trial / "verifier" / "reward.txt").write_text("0")
    with pytest.raises(ValueError, match="mismatch"):
        normalize_harbor(manifest, tmp_path)


@pytest.mark.parametrize(
    "mutation", ["attempt", "summarization", "force_build", "task_bytes"]
)
def test_raw_protocol_drift_is_not_reported_as_valid(manifest_data, tmp_path, mutation):
    manifest = staged_manifest(manifest_data, tmp_path)
    trial = raw_trial(manifest, tmp_path, 0)
    path = trial / "result.json"
    raw = json.loads(path.read_text())
    if mutation == "attempt":
        trial.rename(trial.with_name("001"))
    elif mutation == "summarization":
        raw["config"]["agent"]["kwargs"]["enable_summarize"] = True
        path.write_text(json.dumps(raw))
    elif mutation == "force_build":
        raw["config"]["environment"]["force_build"] = True
        path.write_text(json.dumps(raw))
    else:
        (tmp_path / "tasks/hello/instruction.md").write_text("Changed")
    with pytest.raises(ValueError):
        normalize_harbor(manifest, tmp_path)


def test_missing_cleanup_evidence_cannot_pass(manifest_data, tmp_path):
    manifest = staged_manifest(manifest_data, tmp_path)
    raw_trial(manifest, tmp_path, 1)
    report = normalize_harbor(manifest, tmp_path)
    assert report["complete"] is True  # score coverage, not execution health
    assert report["cleanup_complete"] is False


def test_compute_entry_cannot_bypass_concurrency_gate(
    manifest_data, settings, tmp_path
):
    settings["concurrency"] = 4
    with pytest.raises(ValueError, match="Concurrency"):
        run(Manifest.parse(manifest_data), settings, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_execution_source_is_frozen_and_never_overwritten(manifest_data, settings):
    planned = plan(Manifest.parse(manifest_data), settings)
    materialize(planned)
    bundle = Path(planned["bundle"])
    original = (bundle / "manifest.json").read_bytes()
    materialize(planned)
    assert (bundle / "manifest.json").read_bytes() == original
    source = bundle / "code/eval/autoeval32k/runtime.py"
    source.write_text("drift")
    with pytest.raises(ValueError, match="source changed"):
        materialize(planned)
    assert source.read_text() == "drift"


def test_credentials_are_parsed_without_executing_or_selecting_an_ambiguous_org(
    tmp_path,
):
    path = tmp_path / "runtime.env"
    marker = tmp_path / "not-created"
    path.write_text(
        f"export DAYTONA_API_KEY='test-key' # scoped\nOTHER=$(touch {marker})\n"
    )
    assert read_daytona_key(path) == "test-key"
    assert not marker.exists()
    path.write_text("DAYTONA_API_KEY=first\nDAYTONA_DATA_API_KEY=second\n")
    with pytest.raises(ValueError, match="exactly one"):
        read_daytona_key(path)


def test_early_failure_saves_evidence_without_claiming_execution(
    manifest_data, settings, monkeypatch
):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    manifest = Manifest.parse(manifest_data)
    root = Path(settings["run_root"]) / manifest.run_id
    with pytest.raises(RuntimeError, match="allocated Slurm"):
        run(manifest, settings, root)
    assert (
        json.loads((root / "failure.json").read_text())["error_type"] == "RuntimeError"
    )
    assert json.loads((root / "cleanup-audit.json").read_text())["create_intents"] == 0
    assert not (root / "execution.json").exists()


@pytest.mark.parametrize(
    "confirmation,expected", [("destroyed", True), ("not_found", True), (None, False)]
)
def test_complete_report_requires_confirmed_matching_deletion(
    manifest_data, tmp_path, confirmation, expected
):
    manifest = staged_manifest(manifest_data, tmp_path)
    trial = raw_trial(manifest, tmp_path, 1)
    identity = {"run_id": manifest.run_id, "task_id": "hello"}
    labels = {"purpose": "autoeval32k", "run_id": manifest.run_id, "trial_id": "unique"}
    for name, data in {
        "sandbox-create-intent.json": dict(identity, labels=labels),
        "sandbox.json": dict(identity, labels=labels, sandbox_id="exact-id"),
        "sandbox-cleanup.json": dict(
            identity,
            sandbox_id="exact-id",
            status="delete_completed",
            confirmation=confirmation,
        ),
    }.items():
        (trial / name).write_text(json.dumps(data))
    report = normalize_harbor(manifest, tmp_path)
    assert report["complete"] is True
    assert report["cleanup_complete"] is expected
