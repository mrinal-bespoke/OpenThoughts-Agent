"""Strict opt-in contract: no network or scheduler is needed for these tests."""

import copy
import hashlib
import json

import pytest

from eval.autoeval32k.protocol import (
    Manifest,
    render_harbor,
    serve_command,
    check_context,
)
from eval.autoeval32k.audit import audit_results


def example_manifest():
    return {
        "version": 1,
        "model": {
            "repo": "Qwen/Qwen3-8B",
            "revision": "a" * 40,
            "family": "qwen3",
            "thinking": False,
            "template_sha256": "b" * 64,
        },
        "suite": "synthetic-v1",
        "tasks": [
            {
                "id": "hello",
                "path": "/frozen/hello",
                "sha256": "c" * 64,
                "snapshot": "existing-approved",
                "snapshot_id": "snapshot-immutable-id",
            }
        ],
        "runtime": {
            "harbor_commit": "d" * 40,
            "vllm_version": "0.11.2",
            "agent": "terminus-2",
        },
        "budgets": {
            "startup_seconds": 600,
            "request_seconds": 120,
            "agent_seconds": 600,
            "verifier_seconds": 120,
            "environment_seconds": 180,
        },
    }


@pytest.fixture
def manifest_data():
    return example_manifest()


def test_contract_is_opt_in_deterministic_and_revision_specific(manifest_data):
    first = Manifest.parse(manifest_data)
    assert first.run_id == Manifest.parse(copy.deepcopy(manifest_data)).run_id
    manifest_data["model"]["revision"] = "e" * 40
    assert first.run_id != Manifest.parse(manifest_data).run_id
    assert first.as_dict()["model"]["revision"] == "a" * 40


def test_moving_identical_tasks_does_not_create_a_new_attempt(manifest_data):
    first = Manifest.parse(manifest_data)
    manifest_data["tasks"][0]["path"] = "/different-machine/hello"
    assert first.run_id == Manifest.parse(manifest_data).run_id


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["model"].update(revision="main"),
        lambda d: d.update(retries=1),
        lambda d: d["model"].update(thinking="false"),
        lambda d: d["tasks"].append(copy.deepcopy(d["tasks"][0])),
        lambda d: d["budgets"].update(agent_seconds=0),
        lambda d: d["tasks"][0].update(snapshot=""),
        lambda d: d["model"].update(family="unknown"),
    ],
)
def test_invalid_or_implicit_protocol_is_rejected(manifest_data, mutate):
    mutate(manifest_data)
    with pytest.raises(ValueError):
        Manifest.parse(manifest_data)


def test_complete_config_never_inherits_legacy_defaults(manifest_data, tmp_path):
    manifest = Manifest.parse(manifest_data)
    config = render_harbor(manifest, tmp_path, concurrency=1)
    assert config["n_attempts"] == 1
    assert config["orchestrator"]["retry"]["max_retries"] == 0
    assert config["environment"]["force_build"] is False
    kwargs = config["agents"][0]["kwargs"]
    assert kwargs["enable_summarize"] is False
    assert kwargs["model_info"]["max_input_tokens"] == 24576
    assert kwargs["model_info"]["max_output_tokens"] == 8192
    assert kwargs["temperature"] == 0.6
    assert config["datasets"] == []
    assert len(config["tasks"]) == 1
    argv = serve_command(
        manifest, "/runtime/python", "/cache/pin", port=8000, concurrency=1
    )
    assert argv[argv.index("--max-model-len") + 1] == "32768"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
    assert "--revision" not in argv  # local immutable snapshot; runtime verifies pin


def test_context_is_rejected_not_truncated():
    check_context(24576, 8192)
    for prompt, output in [(24577, 8192), (10, 8193), (-1, 1), (True, 1)]:
        with pytest.raises(ValueError):
            check_context(prompt, output)


def test_audit_preserves_zero_null_missing_and_raw_bytes(manifest_data, tmp_path):
    manifest_data["tasks"] *= 1
    for name in ("null", "missing", "win"):
        task = dict(manifest_data["tasks"][0], id=name, path="/frozen/" + name)
        manifest_data["tasks"].append(task)
    manifest = Manifest.parse(manifest_data)
    paths = []
    for name, reward, error in [
        ("hello", 0, "AgentTimeoutError"),
        ("null", None, "InfraError"),
        ("win", 1, None),
    ]:
        path = tmp_path / (name + ".json")
        path.write_text(
            json.dumps(
                {
                    "run_id": manifest.run_id,
                    "task_id": name,
                    "attempt": 0,
                    "reward": reward,
                    "error": error,
                }
            )
        )
        paths.append(path)
    before = {p: p.read_bytes() for p in paths}
    report = audit_results(manifest, paths)
    assert report["planned"] == 4
    assert report["numeric"] == 2
    assert report["zero"] == 1
    assert report["unscored"] == 1
    assert report["missing"] == ["missing"]
    assert report["successes_over_planned"] == 0.25
    assert report["mean_numeric"] == 0.5
    assert report["complete"] is False
    assert all(p.read_bytes() == before[p] for p in paths)
    assert (
        report["raw_sha256"][str(paths[0])]
        == hashlib.sha256(before[paths[0]]).hexdigest()
    )
    with pytest.raises(ValueError, match="Duplicate"):
        audit_results(manifest, paths + paths[:1])


@pytest.mark.parametrize(
    "field,value",
    [("reward", True), ("reward", float("nan")), ("attempt", 1), ("run_id", "wrong")],
)
def test_audit_fails_closed_on_wrong_evidence(manifest_data, tmp_path, field, value):
    manifest = Manifest.parse(manifest_data)
    record = {
        "run_id": manifest.run_id,
        "task_id": "hello",
        "attempt": 0,
        "reward": 0,
        "error": None,
    }
    record[field] = value
    path = tmp_path / "result.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit_results(manifest, [path])
