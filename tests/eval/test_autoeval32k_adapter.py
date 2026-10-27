"""Real Harbor classes; only HTTP and Daytona service boundaries are simulated."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError
from harbor.llms.chat import Chat
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from upath import UPath

from eval.autoeval32k import harbor_adapter as adapter
from eval.autoeval32k.protocol import Manifest, render_harbor, tree_hash


@pytest.fixture
def manifest(tmp_path):
    task = tmp_path / "tasks" / "example"
    (task / "environment").mkdir(parents=True)
    (task / "environment/Dockerfile").write_text("FROM ubuntu:22.04\n")
    (task / "instruction.md").write_text("Synthetic test fixture")
    return Manifest.parse(
        {
            "version": 1,
            "model": {
                "repo": "org/model",
                "revision": "a" * 40,
                "family": "qwen3",
                "thinking": False,
                "template_sha256": "b" * 64,
            },
            "suite": "synthetic",
            "tasks": [
                {
                    "id": "example",
                    "path": str(task),
                    "sha256": tree_hash(task),
                    "snapshot": "existing",
                    "snapshot_id": "immutable-1",
                }
            ],
            "runtime": {
                "harbor_commit": adapter.SUPPORTED_HARBOR_COMMIT,
                "vllm_version": "0.10.0",
                "agent": "terminus-2",
            },
            "budgets": {
                name: 1
                for name in (
                    "startup_seconds",
                    "request_seconds",
                    "agent_seconds",
                    "verifier_seconds",
                    "environment_seconds",
                )
            },
        }
    )


def environment(manifest, tmp_path):
    return adapter.SnapshotOnly(
        manifest_json=manifest.serialized,
        environment_dir=tmp_path / "tasks/example/environment",
        environment_name="example",
        session_id="trial",
        trial_paths=TrialPaths(trial_dir=UPath(tmp_path / "trial")),
        task_env_config=EnvironmentConfig(),
    )


class Sandbox:
    def __init__(self, labels):
        self.id = "sandbox-unique"
        self.labels = labels
        self.deletes = 0
        self.state = "started"

    async def delete(self):
        self.deletes += 1
        self.state = "destroying"

    async def refresh_data(self):
        self.state = "destroyed"


class Service:
    def __init__(self):
        self.snapshot = SimpleNamespace(get=self.get_snapshot)
        self.snapshot_value = SimpleNamespace(
            name="existing", id="immutable-1", state="active"
        )
        self.creates = 0
        self.sandbox = None

    async def get_snapshot(self, name):
        assert name == "existing"
        return self.snapshot_value

    async def create(self, *, params, timeout):
        self.creates += 1
        self.sandbox = Sandbox(params.labels)
        return self.sandbox

    async def list(self, query):
        if self.sandbox:
            yield self.sandbox


def test_existing_snapshot_one_create_and_idempotent_exact_cleanup(manifest, tmp_path):
    async def scenario():
        env = environment(manifest, tmp_path)
        service = Service()
        params = await env._resolve_start_sandbox_params(
            service, None, force_build=False
        )
        assert params.snapshot == "existing" and params.ephemeral is True
        assert params.auto_stop_interval > 0 and params.auto_delete_interval == 0
        await env._create_sandbox(params, service)
        with pytest.raises(FileExistsError):
            await env._create_sandbox(params, service)
        await env.stop()
        await env.stop()
        assert service.creates == 1 and service.sandbox.deletes == 1
        evidence = json.loads((tmp_path / "trial/sandbox-cleanup.json").read_text())
        assert evidence["status"] == "delete_completed"
        assert evidence["sandbox_id"] == "sandbox-unique"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value", [("id", "wrong"), ("name", "wrong"), ("state", "error")]
)
def test_snapshot_identity_mismatch_never_creates(manifest, tmp_path, field, value):
    async def scenario():
        env = environment(manifest, tmp_path)
        service = Service()
        setattr(service.snapshot_value, field, value)
        with pytest.raises(ValueError, match="identity"):
            await env._resolve_start_sandbox_params(service, None, force_build=False)
        assert service.creates == 0
        assert not (tmp_path / "trial/sandbox-create-intent.json").exists()

    asyncio.run(scenario())


def test_missing_snapshot_never_falls_back_to_build(manifest, tmp_path):
    async def missing(name):
        raise LookupError("Missing snapshot")

    async def scenario():
        service = Service()
        service.snapshot.get = missing
        env = environment(manifest, tmp_path)
        with pytest.raises(LookupError):
            await env._resolve_start_sandbox_params(service, None, force_build=False)
        assert service.creates == 0

    asyncio.run(scenario())


def test_task_mutation_rejected_before_environment_initialization(manifest, tmp_path):
    (tmp_path / "tasks/example/instruction.md").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        environment(manifest, tmp_path)


def test_cancelled_create_captures_late_id_and_deletes_only_it(manifest, tmp_path):
    async def scenario():
        env = environment(manifest, tmp_path)
        service = Service()
        started = asyncio.Event()
        original = service.create

        async def delayed(**kwargs):
            started.set()
            await asyncio.sleep(0.02)
            return await original(**kwargs)

        service.create = delayed
        params = await env._resolve_start_sandbox_params(
            service, None, force_build=False
        )
        task = asyncio.create_task(env._create_sandbox(params, service))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service.creates == 1 and service.sandbox.deletes == 1
        assert (
            json.loads((tmp_path / "trial/sandbox-cleanup.json").read_text())["status"]
            == "delete_completed"
        )

    asyncio.run(scenario())


def test_create_error_preserves_intent_and_unknown_cleanup(manifest, tmp_path):
    async def scenario():
        env = environment(manifest, tmp_path)
        service = Service()

        async def uncertain(**kwargs):
            service.creates += 1
            raise ConnectionError("Response lost")

        service.create = uncertain
        params = await env._resolve_start_sandbox_params(
            service, None, force_build=False
        )
        with pytest.raises(ConnectionError):
            await env._create_sandbox(params, service)
        with pytest.raises(FileExistsError):
            await env._create_sandbox(params, service)
        assert service.creates == 1
        assert (tmp_path / "trial/sandbox-create-intent.json").exists()
        assert (
            json.loads((tmp_path / "trial/sandbox-cleanup.json").read_text())["status"]
            == "unresolved"
        )

    asyncio.run(scenario())


def strict_agent(manifest, tmp_path):
    cfg = render_harbor(manifest, tmp_path)["agents"][0]
    kwargs = dict(cfg["kwargs"], manifest_json=manifest.serialized)
    return adapter.StrictTerminus2(
        logs_dir=tmp_path / "agent", model_name=cfg["model_name"], **kwargs
    )


def completion(model, *, count=10, finish="stop", content="answer", reasoning=None):
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
            }
        ],
        "usage": {
            "prompt_tokens": count,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 1},
        },
    }


def test_real_chat_history_usage_and_exact_template_payload(
    manifest, tmp_path, monkeypatch
):
    agent = strict_agent(manifest, tmp_path)
    calls = []

    async def post(url, payload, timeout):
        calls.append((url, payload, timeout))
        if url.endswith("/tokenize"):
            return {"count": 10}
        return completion(manifest.run_id)

    monkeypatch.setattr(adapter, "_post_json", post)
    chat = Chat(agent._llm)
    chat.messages.append({"role": "system", "content": "instructions"})
    result = asyncio.run(chat.chat("pending prompt"))
    assert result.content == "answer"
    assert chat.total_input_tokens == 10 and chat.total_output_tokens == 2
    assert chat.total_cache_tokens == 1
    assert [m["role"] for m in chat.messages] == ["system", "user", "assistant"]
    assert len(calls) == 2
    assert calls[0][1]["messages"] == calls[1][1]["messages"]
    assert calls[0][1]["messages"][-1]["content"] == "pending prompt"
    assert calls[0][1]["chat_template_kwargs"] == calls[1][1]["chat_template_kwargs"]
    assert calls[0][1]["add_generation_prompt"] is True
    assert calls[1][1]["max_tokens"] == 8192
    assert calls[0][2] == 1


@pytest.mark.parametrize("count", [24577, 32768])
def test_context_overflow_keeps_history_and_never_generates(
    manifest, tmp_path, monkeypatch, count
):
    agent = strict_agent(manifest, tmp_path)
    calls = []

    async def post(url, payload, timeout):
        calls.append(url)
        return {"count": count}

    monkeypatch.setattr(adapter, "_post_json", post)
    chat = Chat(agent._llm)
    chat.messages.append({"role": "system", "content": "original"})
    with pytest.raises(ContextLengthExceededError):
        asyncio.run(chat.chat("pending"))
    assert chat.messages == [{"role": "system", "content": "original"}]
    assert len(calls) == 1 and calls[0].endswith("/tokenize")


def test_failed_tokenizer_has_no_estimate_fallback_or_retry(
    manifest, tmp_path, monkeypatch
):
    agent = strict_agent(manifest, tmp_path)
    calls = []

    async def post(url, payload, timeout):
        calls.append(url)
        raise ConnectionError("failed")

    monkeypatch.setattr(adapter, "_post_json", post)
    with pytest.raises(ConnectionError):
        asyncio.run(Chat(agent._llm).chat("pending"))
    assert len(calls) == 1


def test_output_overflow_is_not_salvaged_or_retried(manifest, tmp_path, monkeypatch):
    agent = strict_agent(manifest, tmp_path)
    calls = []

    async def post(url, payload, timeout):
        calls.append(url)
        return (
            {"count": 10}
            if url.endswith("/tokenize")
            else completion(manifest.run_id, finish="length")
        )

    monkeypatch.setattr(adapter, "_post_json", post)
    context = SimpleNamespace(logging_paths=SimpleNamespace(debug=None, response=None))
    with pytest.raises(OutputLengthExceededError):
        asyncio.run(agent._query_llm(Chat(agent._llm), "pending", context))
    assert len(calls) == 2


def test_transport_budget_bounds_wait(manifest, tmp_path, monkeypatch):
    agent = strict_agent(manifest, tmp_path)
    agent._request_seconds = 0.01

    async def post(url, payload, timeout):
        await asyncio.Event().wait()

    monkeypatch.setattr(adapter, "_post_json", post)
    with pytest.raises(TimeoutError):
        asyncio.run(Chat(agent._llm).chat("pending"))


@pytest.mark.parametrize("mutation", ["model", "tool", "finish", "usage"])
def test_response_identity_shape_and_usage_fail_closed(manifest, tmp_path, mutation):
    agent = strict_agent(manifest, tmp_path)
    result = completion(manifest.run_id)
    if mutation == "model":
        result["model"] = "another"
    elif mutation == "tool":
        result["choices"][0]["message"]["tool_calls"] = [{"name": "unsafe"}]
    elif mutation == "finish":
        result["choices"][0]["finish_reason"] = "content_filter"
    else:
        result["usage"]["prompt_tokens"] = 11
    with pytest.raises(ValueError):
        agent._parse_response(result, 10)


def test_response_preserves_explicit_reasoning(manifest, tmp_path):
    agent = strict_agent(manifest, tmp_path)
    result = agent._parse_response(
        completion(manifest.run_id, reasoning="reasoning text"), 10
    )
    assert result.reasoning_content == "reasoning text"


def test_unqualified_harbor_commit_is_rejected(manifest, tmp_path):
    data = manifest.as_dict()
    data["runtime"]["harbor_commit"] = "f" * 40
    with pytest.raises(ValueError, match="qualified"):
        strict_agent(Manifest.parse(data), tmp_path)


def test_exact_input_boundary_reserves_full_output_budget(
    manifest, tmp_path, monkeypatch
):
    agent = strict_agent(manifest, tmp_path)
    calls = []

    async def post(url, payload, timeout):
        calls.append(payload)
        return (
            {"count": 24576}
            if url.endswith("/tokenize")
            else completion(manifest.run_id, count=24576)
        )

    monkeypatch.setattr(adapter, "_post_json", post)
    response = asyncio.run(Chat(agent._llm).chat("pending"))
    assert response.usage.prompt_tokens == 24576
    assert calls[1]["max_tokens"] == 8192


def test_source_drift_rejected_even_when_commit_matches(manifest, monkeypatch):
    monkeypatch.setitem(adapter._SOURCE_HASHES, adapter.Chat, "0" * 64)
    with pytest.raises(ValueError, match="fingerprint"):
        adapter.validate_runtime(manifest)


def test_cleanup_label_mismatch_never_deletes_another_sandbox(manifest, tmp_path):
    async def scenario():
        env = environment(manifest, tmp_path)
        service = Service()

        async def uncertain(**kwargs):
            service.sandbox = Sandbox({"run_id": "someone-else"})
            raise ConnectionError("lost response")

        service.create = uncertain
        params = await env._resolve_start_sandbox_params(
            service, None, force_build=False
        )
        with pytest.raises(ConnectionError):
            await env._create_sandbox(params, service)
        assert service.sandbox.deletes == 0
        assert (
            json.loads((tmp_path / "trial/sandbox-cleanup.json").read_text())["status"]
            == "unresolved"
        )

    asyncio.run(scenario())


def test_delete_returning_does_not_prove_sandbox_is_gone(manifest, tmp_path):
    async def scenario():
        env = environment(manifest, tmp_path)
        service = Service()
        params = await env._resolve_start_sandbox_params(
            service, None, force_build=False
        )
        await env._create_sandbox(params, service)

        async def still_running():
            service.sandbox.state = "started"

        service.sandbox.refresh_data = still_running
        env._budget = 0.01
        with pytest.raises(TimeoutError):
            await env.stop()
        with pytest.raises(RuntimeError, match="unresolved"):
            await env.stop()
        assert service.sandbox.deletes == 1
        proof = json.loads((tmp_path / "trial/sandbox-cleanup.json").read_text())
        assert proof["status"] == "unresolved"
        assert proof["reason"] == "deletion_not_confirmed"

    asyncio.run(scenario())


def test_exact_id_not_found_confirms_cleanup(manifest, tmp_path):
    async def scenario():
        env = environment(manifest, tmp_path)
        service = Service()
        params = await env._resolve_start_sandbox_params(
            service, None, force_build=False
        )
        await env._create_sandbox(params, service)

        async def gone():
            raise adapter.DaytonaNotFoundError("gone")

        service.sandbox.refresh_data = gone
        await env.stop()
        assert service.sandbox.deletes == 1
        proof = json.loads((tmp_path / "trial/sandbox-cleanup.json").read_text())
        assert proof["status"] == "delete_completed"
        assert proof["confirmation"] == "not_found"

    asyncio.run(scenario())
