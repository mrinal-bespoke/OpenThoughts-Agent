"""Fresh compute-node runner. No resume, image builds, uploads or registry hooks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .audit import audit_results
from .protocol import Manifest, render_harbor, serve_command, tree_hash


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def read_daytona_key(path):
    """Read literal assignments only; ambiguous org keys require a scoped file."""
    import shlex

    values = set()
    for line in Path(path).read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if parts and parts[0] == "export":
            parts = parts[1:]
        if len(parts) == 1 and "=" in parts[0]:
            key, value = parts[0].split("=", 1)
            if key in {"DAYTONA_API_KEY", "DAYTONA_DATA_API_KEY"} and value:
                values.add(value)
    if len(values) != 1:
        raise ValueError("Credential file must identify exactly one Daytona org key")
    return values.pop()


def stage_tasks(manifest, run_dir):
    """Copy, verify and freeze inputs before Harbor can touch any task directory."""
    staged = []
    for task in manifest.as_dict()["tasks"]:
        source = Path(task["path"])
        if tree_hash(source) != task["sha256"]:
            raise ValueError(f"Frozen task changed: {task['id']}")
        target = Path(run_dir) / "tasks" / task["id"]
        shutil.copytree(source, target, symlinks=True)
        if tree_hash(target) != task["sha256"]:
            raise ValueError("Task changed during staging")
        staged.append({"path": str(target)})
    return staged


def check_runtime(manifest, harbor_root):
    """Require the installed Harbor package to come from the exact clean checkout."""
    root = Path(harbor_root).resolve()
    # Reuse installed dependencies with a private, pinned source checkout.
    # This changes only this process, never the shared installed environment.
    sys.path.insert(0, str(root / "src"))
    import harbor

    if not Path(harbor.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Installed Harbor is not from the configured checkout")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], timeout=20, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        timeout=20,
        text=True,
    )
    if commit != manifest.as_dict()["runtime"]["harbor_commit"] or dirty:
        raise ValueError("Harbor checkout differs from frozen runtime pin")
    from .harbor_adapter import validate_runtime

    validate_runtime(manifest)


def normalize_harbor(manifest, run_dir):
    """Cross-check terminal Harbor reward against verifier output, retaining raw hashes."""
    run_dir = Path(run_dir)
    planned = {
        str(run_dir / "tasks" / t["id"]): t["id"] for t in manifest.as_dict()["tasks"]
    }
    expected = render_harbor(manifest, run_dir)
    expected["tasks"] = [{"path": path} for path in planned]
    if json.loads((run_dir / "harbor-config.json").read_text()) != expected:
        raise ValueError("Frozen run configuration changed")
    for task in manifest.as_dict()["tasks"]:
        if tree_hash(run_dir / "tasks" / task["id"]) != task["sha256"]:
            raise ValueError("Staged task changed")
    output = run_dir / "outcomes"
    output.mkdir()
    paths, seen = [], set()
    for path in sorted((run_dir / "trials" / manifest.run_id).rglob("result.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        if "task_name" not in data:
            # Only the known job-level aggregate is not a trial.
            if path.parent != run_dir / "trials" / manifest.run_id:
                raise ValueError("Unrecognized result artifact")
            continue
        config = data["config"]
        relative_parts = path.relative_to(run_dir / "trials" / manifest.run_id).parts
        if (
            "attempts" in relative_parts
            and relative_parts[relative_parts.index("attempts") + 1] != "000"
        ):
            raise ValueError("Unexpected retry/attempt")
        if config.get("timeout_multiplier") != 1.0:
            raise ValueError("Raw timeout protocol mismatch")
        for section, wanted in (
            ("agent", expected["agents"][0]),
            ("environment", expected["environment"]),
            ("verifier", expected["verifier"]),
        ):
            if any(
                config.get(section, {}).get(key) != value
                for key, value in wanted.items()
            ):
                raise ValueError("Raw protocol mismatch: " + section)
        task_id = planned.get(config["task"]["path"])
        if task_id is None or task_id in seen or not data.get("finished_at"):
            raise ValueError("Unexpected, duplicate, or unfinished terminal record")
        if data["task_name"] != task_id:
            raise ValueError("Raw task name differs from frozen inventory")
        seen.add(task_id)
        if config["agent"]["model_name"] != "hosted_vllm/" + manifest.run_id:
            raise ValueError("Raw model identity mismatch")
        rewards = (data.get("verifier_result") or {}).get("rewards")
        if rewards is not None and set(rewards) != {"reward"}:
            raise ValueError("Unexpected verifier metric")
        reward = rewards.get("reward") if rewards is not None else None
        reward_file = path.parent / "verifier" / "reward.txt"
        reward_json = path.parent / "verifier" / "reward.json"
        if reward_file.exists() and reward_json.exists():
            raise ValueError("Ambiguous verifier reward files")
        verifier_reward = None
        if reward_file.exists():
            verifier_reward = float(reward_file.read_text().strip())
        elif reward_json.exists():
            contents = json.loads(reward_json.read_text())
            if set(contents) != {"reward"}:
                raise ValueError("Unexpected raw verifier metric")
            verifier_reward = contents["reward"]
        if reward != verifier_reward:
            raise ValueError("Verifier/raw reward mismatch")
        if verifier_reward is not None and (
            type(verifier_reward) not in (int, float)
            or not math.isfinite(verifier_reward)
            or verifier_reward not in (0, 1)
        ):
            raise ValueError("Invalid verifier reward")
        dest = output / (task_id + ".json")
        write_new(
            dest,
            {
                "run_id": manifest.run_id,
                "task_id": task_id,
                "attempt": 0,
                "reward": reward,
                "error": (data.get("exception_info") or {}).get("exception_type"),
                "raw_path": str(path),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
            },
        )
        paths.append(dest)
    report = audit_results(manifest, paths)
    cleanup = cleanup_evidence(manifest, run_dir)
    report["cleanup_complete"] = cleanup["complete"] and cleanup["task_ids"] == sorted(
        seen
    )
    report["sandbox_create_intents"] = cleanup["create_intents"]
    write_new(run_dir / "audit.json", report)
    return report


def cleanup_evidence(manifest, run_dir):
    intents = list((Path(run_dir) / "trials").rglob("sandbox-create-intent.json"))
    seen = set()
    complete = True
    for path in intents:
        try:
            intent = json.loads(path.read_text())
            created = json.loads((path.parent / "sandbox.json").read_text())
            deleted = json.loads((path.parent / "sandbox-cleanup.json").read_text())
            task = intent["task_id"]
            valid = (
                task not in seen
                and task in {t["id"] for t in manifest.as_dict()["tasks"]}
                and all(
                    d["run_id"] == manifest.run_id and d["task_id"] == task
                    for d in (intent, created, deleted)
                )
                and intent["labels"] == created["labels"]
                and isinstance(created["sandbox_id"], str)
                and bool(created["sandbox_id"])
                and deleted["sandbox_id"] == created["sandbox_id"]
                and deleted["status"] == "delete_completed"
                and deleted.get("confirmation") in {"destroyed", "not_found"}
            )
            complete = complete and valid
            seen.add(task)
        except (OSError, ValueError, KeyError, TypeError):
            complete = False
    return {
        "complete": complete,
        "create_intents": len(intents),
        "task_ids": sorted(seen),
    }


def fetch_json(url, payload=None, timeout=30):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    # Local model traffic must never traverse a login proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=timeout
    ) as response:
        return json.load(response)


def terminate_owned(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def run(manifest, settings, run_dir):
    from .__main__ import validate_settings

    settings = validate_settings(settings)
    run_dir = Path(run_dir).absolute()
    if run_dir != Path(settings["run_root"]) / manifest.run_id:
        raise ValueError("Execution directory does not match immutable run identity")
    run_dir.mkdir(parents=True, exist_ok=False)  # never resume or overwrite
    write_new(run_dir / "manifest.json", manifest.as_dict())
    write_new(
        run_dir / "settings.json", settings
    )  # paths only, never credential values
    serving = None
    started = time.monotonic()
    try:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Compute execution requires an allocated Slurm job")
        if os.environ.get("SLURM_JOB_NAME") != manifest.run_id:
            raise RuntimeError("Allocation name differs from run identity")
        check_runtime(manifest, settings["harbor_root"])
        # Only the two Daytona names are accepted; do not execute a shell env file.
        if settings.get("credential_file"):
            os.environ["DAYTONA_API_KEY"] = read_daytona_key(
                settings["credential_file"]
            )
        if not os.environ.get("DAYTONA_API_KEY"):
            raise RuntimeError("Missing Daytona org credential")
        for key in list(os.environ):
            if key.startswith(("SUPABASE_", "WANDB_", "HARBOR_TELEMETRY_")):
                os.environ.pop(key)
        os.environ.update(
            DO_NOT_TRACK="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            VLLM_NO_USAGE_STATS="1",
            PYTHONDONTWRITEBYTECODE="1",
            TOKENIZERS_PARALLELISM="false",
        )
        tasks = stage_tasks(manifest, run_dir)
        config = render_harbor(manifest, run_dir, settings["concurrency"])
        config["tasks"] = tasks
        from harbor.models.job.config import JobConfig

        parsed = JobConfig.model_validate(config)
        # Require effective values; older schemas silently ignore unknown fields.
        if (
            parsed.n_attempts != 1
            or parsed.retry.max_retries != 0
            or parsed.n_concurrent_trials != settings["concurrency"]
        ):
            raise ValueError("Harbor ignored required protocol fields")
        write_new(run_dir / "harbor-config.json", config)
        model = manifest.as_dict()["model"]
        # Tokenizer dependencies already live in the serving environment. Bound
        # cache verification/download too, before allocating time to vLLM startup.
        checkpoint_command = [
            settings["serve_python"],
            "-B",
            "-m",
            "eval.autoeval32k.checkpoint",
            "--manifest",
            str(run_dir / "manifest.json"),
            "--run-dir",
            str(run_dir),
        ]
        if settings.get("checkpoint_cache"):
            checkpoint_command += ["--cache", settings["checkpoint_cache"]]
        with (run_dir / "checkpoint.log").open("x") as log:
            subprocess.run(
                checkpoint_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=manifest.as_dict()["budgets"]["startup_seconds"],
            )
        snapshot = Path(
            json.loads((run_dir / "checkpoint.json").read_text())["snapshot"]
        ).absolute()
        if (
            snapshot.name != model["revision"]
            or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name
            != "models--" + model["repo"].replace("/", "--")
        ):
            raise ValueError("Snapshot path does not match exact HF repo/revision")
        # vLLM version and GPU math are checked in its actual Python environment.
        probe = subprocess.run(
            [
                settings["serve_python"],
                "-c",
                "import json,torch,vllm; assert torch.cuda.is_available(); "
                "x=torch.ones((16,16),device='cuda',dtype=torch.bfloat16); y=x@x; "
                "torch.cuda.synchronize(); assert bool((y==16).all()); "
                "print(json.dumps({'vllm':vllm.__version__,'bf16':True,'gpu':torch.cuda.get_device_name(0)}))",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=manifest.as_dict()["budgets"]["startup_seconds"],
        )
        gpu = json.loads(probe.stdout.splitlines()[-1])
        if gpu["vllm"] != manifest.as_dict()["runtime"]["vllm_version"]:
            raise ValueError("vLLM runtime pin mismatch")
        write_new(run_dir / "gpu.json", gpu)
        env = dict(
            os.environ,
            HF_HOME=str(run_dir / "hf-cache"),
            XDG_CACHE_HOME=str(run_dir / "cache"),
            TRITON_CACHE_DIR=str(run_dir / "cache" / "triton"),
        )
        for key in ("DAYTONA_API_KEY", "DAYTONA_DATA_API_KEY"):
            env.pop(key, None)
        with (run_dir / "vllm.log").open("x") as log:
            serving = subprocess.Popen(
                serve_command(
                    manifest,
                    settings["serve_python"],
                    snapshot,
                    concurrency=settings["concurrency"],
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            deadline = (
                time.monotonic() + manifest.as_dict()["budgets"]["startup_seconds"]
            )
            while True:
                if serving.poll() is not None:
                    raise RuntimeError("Serving process exited during startup")
                try:
                    models = fetch_json("http://127.0.0.1:8000/v1/models", timeout=5)
                    if any(
                        m.get("id") == manifest.run_id for m in models.get("data", [])
                    ):
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Serving readiness deadline")
                time.sleep(2)
            completion = fetch_json(
                "http://127.0.0.1:8000/v1/chat/completions",
                {
                    "model": manifest.run_id,
                    "messages": [{"role": "user", "content": "Reply OK."}],
                    "max_tokens": 32,
                    "temperature": 0.6,
                    "chat_template_kwargs": {"enable_thinking": model["thinking"]},
                },
                timeout=manifest.as_dict()["budgets"]["request_seconds"],
            )
            if not completion["choices"][0]["message"].get("content"):
                raise ValueError("Serving probe produced no text")
            write_new(
                run_dir / "serving.json",
                {
                    "passed": True,
                    "model": manifest.run_id,
                    "usage": completion.get("usage"),
                },
            )
            from harbor.job import Job

            async def execute():
                job = await Job.create(parsed)
                return await job.run()

            asyncio.run(execute())
        report = normalize_harbor(manifest, run_dir)
        passed = report["complete"] and report["cleanup_complete"]
        write_new(
            run_dir / "execution.json",
            {"finished": True, "passed": passed, "seconds": time.monotonic() - started},
        )
        return 0 if passed else 1
    except BaseException as exc:
        # Exception strings may contain credentials/URLs from dependencies.
        write_new(
            run_dir / "failure.json",
            {"error_type": type(exc).__name__, "seconds": time.monotonic() - started},
        )
        raise
    finally:
        try:
            terminate_owned(serving)
        finally:
            write_new(
                run_dir / "cleanup-audit.json", cleanup_evidence(manifest, run_dir)
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = Manifest.parse(json.loads(args.manifest.read_text()))
    settings = json.loads(args.settings.read_text())
    bundle = args.manifest.parent
    lock = json.loads((bundle / "code-lock.json").read_text())
    actual = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    if actual != lock:
        raise ValueError("Execution source differs from submitted bundle")

    def terminate(signum, frame):
        raise KeyboardInterrupt("Allocation stopping")

    signal.signal(signal.SIGTERM, terminate)
    return run(manifest, settings, args.run_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Dependency exceptions can embed authenticated URLs; retain type only.
        print(
            "Execution failed: " + type(exc).__name__ + "; inspect private run evidence"
        )
        raise SystemExit(1) from None
