"""Versioned evaluation identity, independent of legacy listener defaults."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

PROFILE = {
    "name": "terminus2-32k-v1",
    "context_tokens": 32768,
    "input_tokens": 24576,
    "output_tokens": 8192,
    "temperature": 0.6,
    "summarize": False,
    "attempts": 1,
    "retries": 0,
    "dtype": "bfloat16",
    "tensor_parallel": 1,
}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label}: expected exactly {sorted(expected)}")


def token(value, pattern, label):
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError(f"Invalid {label}")


@dataclass(frozen=True)
class Manifest:
    """Canonical JSON prevents callers mutating the identity after validation."""

    serialized: str

    @classmethod
    def parse(cls, data):
        keys(
            data,
            {"version", "model", "suite", "tasks", "runtime", "budgets"},
            "manifest",
        )
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("Unsupported manifest version")
        model = data["model"]
        keys(
            model,
            {"repo", "revision", "family", "thinking", "template_sha256"},
            "model",
        )
        token(model["repo"], r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", "model repo")
        token(model["revision"], r"[0-9a-f]{40}", "immutable model revision")
        token(model["template_sha256"], r"[0-9a-f]{64}", "template hash")
        if model["family"] != "qwen3" or type(model["thinking"]) is not bool:
            raise ValueError("MVP requires explicit Qwen3 thinking mode")
        token(data["suite"], r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", "suite")
        tasks = data["tasks"]
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("An explicit nonempty task inventory is required")
        seen = set()
        paths = set()
        for task in tasks:
            keys(task, {"id", "path", "sha256", "snapshot", "snapshot_id"}, "task")
            token(task["id"], r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", "task ID")
            token(task["sha256"], r"[0-9a-f]{64}", "task hash")
            for field in ("snapshot", "snapshot_id"):
                token(task[field], r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", field)
            path = task["path"]
            if (
                not isinstance(path, str)
                or not Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                raise ValueError("Task paths must be absolute and normalized")
            if task["id"] in seen or path in paths:
                raise ValueError("Duplicate task identity/path")
            seen.add(task["id"])
            paths.add(path)
        runtime = data["runtime"]
        keys(runtime, {"harbor_commit", "vllm_version", "agent"}, "runtime")
        token(runtime["harbor_commit"], r"[0-9a-f]{40}", "Harbor commit")
        token(
            runtime["vllm_version"],
            r"[A-Za-z0-9][A-Za-z0-9_.+\-]{0,127}",
            "vLLM version",
        )
        if runtime["agent"] != "terminus-2":
            raise ValueError("Only Terminus-2 is qualified by this profile")
        keys(
            data["budgets"],
            {
                "startup_seconds",
                "request_seconds",
                "agent_seconds",
                "verifier_seconds",
                "environment_seconds",
            },
            "budgets",
        )
        for name, value in data["budgets"].items():
            if type(value) is not int or not 1 <= value <= 86400:
                raise ValueError(f"Invalid {name}")
        return cls(canonical(data))

    def as_dict(self):
        return json.loads(self.serialized)

    @property
    def run_id(self):
        # Placement is not a new attempt: the same manifest cannot be replayed by
        # changing clusters or concurrency. Task order is not an identity change.
        data = self.as_dict()
        data["tasks"] = sorted(data["tasks"], key=lambda t: t["id"])
        for task in data["tasks"]:
            del task["path"]
        return "ae32-" + digest({"manifest": data, "profile": PROFILE})[:32]


def check_context(prompt_tokens, output_tokens=8192):
    if (
        type(prompt_tokens) is not int
        or type(output_tokens) is not int
        or not 0 <= prompt_tokens <= PROFILE["input_tokens"]
        or not 1 <= output_tokens <= PROFILE["output_tokens"]
        or prompt_tokens + output_tokens > PROFILE["context_tokens"]
    ):
        raise ValueError("Context overflow: no truncation or summarization permitted")


def render_harbor(manifest, run_dir, concurrency=1, port=8000):
    if type(concurrency) is not int or not 1 <= concurrency <= 8:
        raise ValueError("Concurrency outside 1..8; promotion needs qualification")
    data = manifest.as_dict()
    budgets = data["budgets"]
    return {
        "job_name": manifest.run_id,
        "jobs_dir": str(Path(run_dir) / "trials"),
        "n_attempts": 1,
        "timeout_multiplier": 1.0,
        "orchestrator": {
            "type": "local",
            "n_concurrent_trials": concurrency,
            "retry": {"max_retries": 0},
        },
        "environment": {
            "import_path": "eval.autoeval32k.harbor_adapter:SnapshotOnly",
            "force_build": False,
            "delete": True,
            "kwargs": {"manifest_json": manifest.serialized},
        },
        "agents": [
            {
                "import_path": "eval.autoeval32k.harbor_adapter:StrictTerminus2",
                "model_name": "hosted_vllm/" + manifest.run_id,
                "override_timeout_sec": budgets["agent_seconds"],
                "override_setup_timeout_sec": budgets["environment_seconds"],
                "kwargs": {
                    "manifest_json": manifest.serialized,
                    "api_base": f"http://127.0.0.1:{port}/v1",
                    "temperature": 0.6,
                    "enable_summarize": False,
                    "proactive_summarization_threshold": 0,
                    "record_terminal_session": False,
                    "model_info": {
                        "max_input_tokens": 24576,
                        "max_output_tokens": 8192,
                        "input_cost_per_token": 0,
                        "output_cost_per_token": 0,
                    },
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": data["model"]["thinking"]
                        }
                    },
                },
            }
        ],
        "verifier": {"override_timeout_sec": budgets["verifier_seconds"]},
        "datasets": [],
        "tasks": [{"path": task["path"]} for task in data["tasks"]],
    }


def serve_command(manifest, python, snapshot, port=8000, concurrency=1):
    return [
        str(python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(snapshot),
        "--tokenizer",
        str(snapshot),
        "--served-model-name",
        manifest.run_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        str(concurrency),
        "--gpu-memory-utilization",
        "0.9",
    ]


def tree_hash(root):
    """Hash all task bytes; symlinks are forbidden to prevent out-of-tree drift."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Task must be a real directory")
    records = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Task symlinks are not allowed")
        if path.is_file():
            records[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    if not records:
        raise ValueError("Empty task")
    return digest(records)
