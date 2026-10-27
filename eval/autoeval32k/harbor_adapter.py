"""Narrow, source-qualified Harbor adapters, not a vanilla-transport parity claim.

No snapshot creation/build fallback; no generation retry, history summarization,
or truncation. Live qualification is still required for every runtime deployment.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
import uuid
from pathlib import Path

import httpx
from daytona import (
    CreateSandboxFromSnapshotParams,
    DaytonaNotFoundError,
    ListSandboxesQuery,
)
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.daytona.environment import DaytonaEnvironment
from harbor.llms.base import (
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM
from harbor.models.metric import UsageInfo

from .protocol import Manifest, check_context, tree_hash
from .queue import _write_new

SUPPORTED_HARBOR_COMMIT = "84c67493d28fdd56d6d1476da1d95485d4e41b37"
_SOURCE_HASHES = {
    DaytonaEnvironment: "47a3ccc6a385574c513b3ec1064940f177e3aaf940ce7e73089e05cfa5655a02",
    Terminus2: "9f3d040e8c1af1ba267c7c1e5055279750affae8ea0ffaae895315a3aab132b4",
    LiteLLM: "ddeb85d581a1e4e9b8c6f27cc9dc198255b45c13137e6cdb9e9e46ca03f7d34d",
    Chat: "7a9b524df9ea6de3bf0537993be350a6023cfcc32b6b7ed9faa0086872f53bdc",
}


def _manifest(serialized):
    manifest = Manifest.parse(json.loads(serialized))
    validate_runtime(manifest)
    return manifest


def validate_runtime(manifest):
    """Reject unreviewed Harbor implementations, including dirty source drift."""
    if manifest.as_dict()["runtime"]["harbor_commit"] != SUPPORTED_HARBOR_COMMIT:
        raise ValueError("Harbor commit has not been qualified for this adapter")
    for cls, expected in _SOURCE_HASHES.items():
        source = inspect.getsourcefile(cls)
        if (
            not source
            or hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Harbor adapter source fingerprint mismatch")


class SnapshotOnly(DaytonaEnvironment):
    """Direct-container existing snapshots only; uncertain creates need reconciliation."""

    def __init__(self, *args, manifest_json, **kwargs):
        self._manifest = _manifest(manifest_json)
        data = self._manifest.as_dict()
        environment_dir = Path(kwargs.get("environment_dir", args[0] if args else ""))
        task_root = environment_dir.parent
        tasks = [task for task in data["tasks"] if task["id"] == task_root.name]
        if environment_dir.name != "environment" or len(tasks) != 1:
            raise ValueError("Environment does not map to exactly one frozen task")
        self._task = tasks[0]
        if tree_hash(task_root) != self._task["sha256"]:
            raise ValueError("Frozen task hash mismatch")
        if (environment_dir / "docker-compose.yaml").exists() or kwargs.get(
            "extra_docker_compose"
        ):
            raise ValueError("Compose environments are not qualified")
        forbidden = {
            "auto_snapshot",
            "snapshot_template_name",
            "assume_global_snapshot",
            "auto_stop_interval_mins",
            "auto_delete_interval_mins",
        }
        if forbidden.intersection(kwargs):
            raise ValueError("Snapshot policy cannot be overridden")
        self._budget = data["budgets"]["environment_seconds"]
        # Bounded idle expiry, not a hard lifetime; job/controller bounds remain mandatory.
        idle_minutes = (
            math.ceil(
                sum(
                    data["budgets"][k]
                    for k in (
                        "agent_seconds",
                        "verifier_seconds",
                        "environment_seconds",
                    )
                )
                / 60
            )
            + 1
        )
        super().__init__(
            *args,
            auto_snapshot=False,
            auto_stop_interval_mins=idle_minutes,
            auto_delete_interval_mins=0,
            **kwargs,
        )
        self.task_env_config = self.task_env_config.model_copy(
            update={"build_timeout_sec": self._budget}, deep=True
        )
        self._evidence = Path(str(self.trial_paths.trial_dir))
        if not self._evidence.is_absolute() or self._evidence.is_symlink():
            raise ValueError("Trial evidence must be a local absolute directory")
        self._evidence.mkdir(parents=True, exist_ok=True)
        self._labels = {
            "purpose": "autoeval32k",
            "run_id": self._manifest.run_id,
            "trial_id": uuid.uuid4().hex,
        }
        self._daytona = None

    def _record(self, name, **data):
        _write_new(
            self._evidence / name,
            {
                "schema_version": 1,
                "run_id": self._manifest.run_id,
                "task_id": self._task["id"],
                **data,
            },
        )

    async def _resolve_start_sandbox_params(self, daytona, resources, *, force_build):
        if force_build:
            raise ValueError("Building environments is forbidden")
        snapshot = await asyncio.wait_for(
            daytona.snapshot.get(self._task["snapshot"]), self._budget
        )
        state = getattr(snapshot.state, "value", snapshot.state)
        if (
            snapshot.id != self._task["snapshot_id"]
            or snapshot.name != self._task["snapshot"]
            or str(state).lower() != "active"
        ):
            raise ValueError("Snapshot identity or readiness mismatch")
        self._record(
            "snapshot-verified.json", snapshot=snapshot.name, snapshot_id=snapshot.id
        )
        return CreateSandboxFromSnapshotParams(
            snapshot=snapshot.name,
            # Harbor prepares /logs via su root. A snapshot's nonroot default
            # can block that command awaiting authentication instead of exiting.
            os_user="root",
            ephemeral=True,
            public=False,
            auto_stop_interval=self._auto_stop_interval,
            auto_delete_interval=0,
            network_block_all=self._network_block_all,
            labels=self._labels,
        )

    def _capture(self, sandbox):
        if not isinstance(sandbox.id, str) or not sandbox.id:
            raise ValueError("Missing created sandbox identity")
        if not isinstance(sandbox.labels, dict) or any(
            sandbox.labels.get(k) != v for k, v in self._labels.items()
        ):
            raise ValueError("Created sandbox labels differ from this exact intent")
        self._sandbox = sandbox
        self._record("sandbox.json", sandbox_id=sandbox.id, labels=self._labels)

    async def _create_sandbox(self, params, daytona=None):
        if daytona is None or not isinstance(params, CreateSandboxFromSnapshotParams):
            raise ValueError("Only a verified existing-snapshot create is supported")
        if (
            params.snapshot != self._task["snapshot"]
            or params.labels != self._labels
            or params.os_user != "root"
        ):
            raise ValueError("Create parameters differ from verified snapshot policy")
        if not (self._evidence / "snapshot-verified.json").exists():
            raise ValueError("Snapshot has not been verified")
        self._daytona = daytona
        self._record(
            "sandbox-create-intent.json",
            labels=self._labels,
            snapshot=self._task["snapshot"],
            snapshot_id=self._task["snapshot_id"],
            os_user=params.os_user,
        )
        task = asyncio.create_task(daytona.create(params=params, timeout=self._budget))
        try:
            sandbox = await asyncio.wait_for(asyncio.shield(task), self._budget)
            self._capture(sandbox)
        except BaseException:
            # Preserve the one in-flight creation briefly on cancellation, so an
            # assigned ID is not lost. Never start a second creation request.
            if not task.done():
                try:
                    sandbox = await asyncio.wait_for(
                        asyncio.shield(task), min(30, self._budget)
                    )
                    self._capture(sandbox)
                except BaseException:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            elif (
                not task.cancelled()
                and task.exception() is None
                and self._sandbox is None
            ):
                self._capture(task.result())
            await self._cleanup_after_failure()
            raise

    async def _cleanup_after_failure(self):
        try:
            await asyncio.wait_for(self.stop(delete=True), self._budget)
        except BaseException:
            # The create intent/cleanup intent is deliberately retained. The
            # controller must report unresolved cleanup instead of retrying work.
            pass

    async def start(self, force_build):
        if force_build:
            raise ValueError("Building environments is forbidden")
        try:
            await asyncio.wait_for(super().start(False), self._budget)
        except BaseException:
            await self._cleanup_after_failure()
            raise

    async def stop(self, delete=True):
        if not delete:
            raise ValueError("Keeping evaluation sandboxes is forbidden")
        final = self._evidence / "sandbox-cleanup.json"
        if final.exists():
            if json.loads(final.read_text())["status"] != "delete_completed":
                raise RuntimeError(
                    "Cleanup unresolved; explicit reconciliation required"
                )
            return
        if not (self._evidence / "sandbox-create-intent.json").exists():
            return
        if self._sandbox is None and self._daytona is not None:

            async def locate():
                return [
                    item
                    async for item in self._daytona.list(
                        ListSandboxesQuery(labels=self._labels)
                    )
                ]

            found = await asyncio.wait_for(locate(), self._budget)
            if len(found) != 1 or any(
                found[0].labels.get(k) != v for k, v in self._labels.items()
            ):
                self._record(
                    "sandbox-cleanup.json",
                    status="unresolved",
                    reason="ambiguous_create_outcome",
                )
                raise RuntimeError("Creation outcome unresolved; do not replay")
            self._capture(found[0])
        if self._sandbox is None:
            raise RuntimeError("No exact sandbox identity available for cleanup")
        saved = json.loads((self._evidence / "sandbox.json").read_text())
        if self._sandbox.id != saved["sandbox_id"] or saved["labels"] != self._labels:
            raise ValueError("Refusing cleanup of a different sandbox")
        self._record("sandbox-cleanup-intent.json", sandbox_id=self._sandbox.id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._budget
        confirmation = None
        try:
            try:
                await asyncio.wait_for(self._sandbox.delete(), self._budget)
            except DaytonaNotFoundError:
                confirmation = "not_found"
            while confirmation is None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError("Sandbox deletion was not confirmed")
                try:
                    await asyncio.wait_for(self._sandbox.refresh_data(), remaining)
                except DaytonaNotFoundError:
                    confirmation = "not_found"
                    break
                if self._sandbox.id != saved["sandbox_id"]:
                    raise ValueError("Refreshed sandbox identity changed")
                state = getattr(self._sandbox.state, "value", self._sandbox.state)
                if str(state).lower() == "destroyed":
                    confirmation = "destroyed"
                    break
                await asyncio.sleep(min(0.25, max(0, deadline - loop.time())))
        except BaseException:
            self._record(
                "sandbox-cleanup.json",
                status="unresolved",
                sandbox_id=saved["sandbox_id"],
                reason="deletion_not_confirmed",
            )
            raise
        self._record(
            "sandbox-cleanup.json",
            status="delete_completed",
            sandbox_id=self._sandbox.id,
            confirmation=confirmation,
        )
        self._sandbox = None


async def _post_json(url, payload, timeout):
    # No inherited proxy, automatic retry, or provider parameter fallback.
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


class StrictTerminus2(Terminus2):
    """Terminus parser/loop with a bounded, one-request local-vLLM transport."""

    def __init__(self, *args, manifest_json, **kwargs):
        manifest = _manifest(manifest_json)
        data = manifest.as_dict()
        self._request_seconds = data["budgets"]["request_seconds"]
        self._thinking = data["model"]["thinking"]
        self._served = manifest.run_id
        self._api_base = kwargs.get("api_base", "")
        if (
            re.fullmatch(r"http://127\.0\.0\.1:[1-9][0-9]{0,4}/v1", self._api_base)
            is None
        ):
            raise ValueError("Strict adapter only accepts the local vLLM endpoint")
        expected = {
            "model_name": "hosted_vllm/" + self._served,
            "temperature": 0.6,
            "enable_summarize": False,
            "proactive_summarization_threshold": 0,
        }
        if any(kwargs.get(k) != v for k, v in expected.items()):
            raise ValueError("Strict agent profile mismatch")
        if (
            kwargs.get("model_info", {}).get("max_input_tokens") != 24576
            or kwargs.get("model_info", {}).get("max_output_tokens") != 8192
        ):
            raise ValueError("Strict agent token budget mismatch")
        if kwargs.get("extra_body") != {
            "chat_template_kwargs": {"enable_thinking": self._thinking}
        }:
            raise ValueError("Strict template/thinking configuration mismatch")
        for key in (
            "llm_kwargs",
            "llm_call_kwargs",
            "turn_callback",
            "max_thinking_tokens",
            "use_responses_api",
            "collect_rollout_details",
        ):
            if kwargs.get(key):
                raise ValueError(f"Unqualified strict-agent override: {key}")
        if kwargs.get("interleaved_thinking", True) is not True:
            raise ValueError("Explicit reasoning content must remain in history")
        kwargs["interleaved_thinking"] = True
        super().__init__(*args, **kwargs)
        # Instance-local seam retains Chat's usage accounting and history. Do
        # not invoke LiteLLM.call: it has nested retries/parameter fallbacks.
        self._llm.call = self._strict_call
        self._llm.get_model_context_limit = lambda: 24576
        self._llm.get_model_output_limit = lambda: 8192
        self._llm_call_kwargs = {}

    async def _tokens(self, messages):
        payload = {
            "model": self._served,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": self._thinking},
        }
        result = await _post_json(
            self._api_base[:-3] + "/tokenize", payload, self._request_seconds
        )
        count = result.get("count")
        if type(count) is not int or count < 0:
            raise ValueError("Exact tokenizer count unavailable; no estimate fallback")
        try:
            check_context(count)
        except ValueError as exc:
            raise ContextLengthExceededError(str(exc)) from exc
        return count

    async def _check_proactive_summarization(
        self, chat, original_instruction, session, pending_prompt=""
    ):
        # Exact check occurs at the transport boundary including the pending turn.
        return None

    def _limit_output_length(self, output, max_bytes=10000):
        return output

    async def _query_llm(self, chat, prompt, request_context):
        paths = request_context.logging_paths
        return await self._dispatch_llm_request(
            chat, prompt, paths.debug, paths.response
        )

    async def _strict_call(self, prompt, message_history, logging_path=None, **kwargs):
        if any(value is not None for value in kwargs.values()):
            raise ValueError("Unqualified transport kwargs")
        messages = list(message_history) + [{"role": "user", "content": prompt}]
        if any(
            not isinstance(m, dict)
            or m.get("role") not in {"system", "user", "assistant"}
            or not isinstance(m.get("content"), str)
            or set(m) - {"role", "content", "reasoning_content"}
            for m in messages
        ):
            raise ValueError("Native tools/multimodal content are not qualified")

        async def request():
            count = await self._tokens(messages)
            payload = {
                "model": self._served,
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": 8192,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": self._thinking},
            }
            result = await _post_json(
                self._api_base + "/chat/completions", payload, self._request_seconds
            )
            return self._parse_response(result, count)

        return await asyncio.wait_for(request(), self._request_seconds)

    def _parse_response(self, result, count):
        if result.get("model") != self._served or len(result.get("choices", [])) != 1:
            raise ValueError("Unexpected served identity or choice count")
        choice = result["choices"][0]
        message = choice.get("message", {})
        if choice.get("finish_reason") == "length":
            raise OutputLengthExceededError(
                "Response exceeded strict 8192-token budget"
            )
        if (
            choice.get("finish_reason") != "stop"
            or message.get("tool_calls")
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("Unsupported completion shape or finish reason")
        if message.get("role") != "assistant" or any(
            value is not None
            and key not in {"role", "content", "reasoning_content", "tool_calls"}
            for key, value in message.items()
        ):
            raise ValueError("Unsupported assistant response fields")
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("Invalid reasoning content")
        usage = result.get("usage", {})
        completion = usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if (
            type(usage.get("prompt_tokens")) is not int
            or usage.get("prompt_tokens") != count
            or type(completion) is not int
            or not 0 <= completion <= 8192
            or type(cached) is not int
            or not 0 <= cached <= count
        ):
            raise ValueError("Token usage differs from exact request budget")
        return LLMResponse(
            content=message["content"],
            reasoning_content=reasoning,
            model_name=self._served,
            usage=UsageInfo(
                prompt_tokens=count,
                completion_tokens=completion,
                cache_tokens=cached,
                cost_usd=0,
            ),
        )
