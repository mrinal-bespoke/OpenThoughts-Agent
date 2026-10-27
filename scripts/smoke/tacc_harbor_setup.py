"""One fresh, bounded real Harbor setup diagnostic, without model or benchmark."""

import argparse
import asyncio
import json
import os
import socket
import time
import tomllib
from pathlib import Path

from eval.autoeval32k.protocol import Manifest
from eval.autoeval32k.runtime import (
    check_runtime,
    read_daytona_key,
    stage_tasks,
    write_new,
)


async def diagnose(args):
    root = args.root.resolve()
    root.mkdir(exist_ok=False)
    data = json.loads(args.manifest.read_text())
    data["suite"] = root.name
    manifest = Manifest.parse(data)
    assert data["budgets"]["environment_seconds"] == 60
    assert len(data["tasks"]) == 1 and data["tasks"][0]["id"] == "hello"
    check_runtime(manifest, args.harbor_root)
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.config import ServiceVolumeConfig
    from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

    from eval.autoeval32k.harbor_adapter import SnapshotOnly

    os.environ["DAYTONA_API_KEY"] = read_daytona_key(args.credential_file)
    os.environ["DAYTONA_API_URL"] = "https://app.daytona.io/api"
    write_new(root / "manifest.json", data)
    task = Path(stage_tasks(manifest, root)[0]["path"])
    paths = TrialPaths(root / "trial")
    paths.mkdir()
    targets = EnvironmentPaths()
    mounts = [
        ServiceVolumeConfig(
            type="bind",
            source=str(getattr(paths, name).resolve()),
            target=str(getattr(targets, name)),
        )
        for name in ("verifier_dir", "agent_dir", "artifacts_dir")
    ]
    config = tomllib.loads((task / "task.toml").read_text())
    env = SnapshotOnly(
        environment_dir=task / "environment",
        environment_name="hello",
        session_id=root.name,
        trial_paths=paths,
        task_env_config=EnvironmentConfig.model_validate(config["environment"]),
        mounts=mounts,
        manifest_json=manifest.serialized,
    )
    started = time.monotonic()
    counts = {}

    def event(method, phase, **fields):
        with (root / "stages.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        method=method,
                        phase=phase,
                        elapsed=time.monotonic() - started,
                        **fields,
                    )
                )
                + "\n"
            )

    def trace(method):
        original = getattr(env, method)

        async def wrapped(*a, **kw):
            counts[method] = counts.get(method, 0) + 1
            count = counts[method]
            # Bound log volume while retaining early calls and periodic progress.
            logged = count <= 5 or count % 100 == 0
            if logged:
                event(method, "enter", count=count)
            try:
                result = await original(*a, **kw)
            except BaseException as exc:
                event(method, "error", count=count, error_type=type(exc).__name__)
                raise
            if logged:
                event(
                    method,
                    "return",
                    count=count,
                    exit_code=getattr(result, "exit_code", None),
                    return_code=getattr(result, "return_code", None),
                )
            return result

        setattr(env, method, wrapped)

    for method in (
        "_resolve_start_sandbox_params",
        "_create_sandbox",
        "ensure_dirs",
        "_ensure_exec_session",
        "_execute_session_command_with_retry",
        "_get_session_command_with_retry",
        "_get_session_command_logs_with_retry",
        "_upload_environment_dir_after_start",
    ):
        trace(method)
    write_new(
        root / "started.json",
        dict(
            hostname=socket.gethostname(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            run_id=manifest.run_id,
            model_loaded=False,
            environment_seconds=60,
        ),
    )
    passed = False
    try:
        await env.start(False)  # Real adapter's unchanged 60-second setup bound.
        write_new(
            root / "setup.json", dict(passed=True, elapsed=time.monotonic() - started)
        )
        result = await asyncio.wait_for(
            env.exec("printf HARBOR_SETUP_OK", timeout_sec=5), 10
        )
        assert result.return_code == 0 and result.stdout.strip() == "HARBOR_SETUP_OK"
        write_new(root / "command.json", dict(exit_code=0, output_matches=True))
        passed = True
    except BaseException as exc:
        write_new(root / "failure.json", dict(error_type=type(exc).__name__))
    finally:
        try:
            await asyncio.wait_for(env.stop(True), 60)
        except BaseException as exc:
            write_new(root / "cleanup-error.json", dict(error_type=type(exc).__name__))
        cleanup = paths.trial_dir / "sandbox-cleanup.json"
        cleaned = (
            cleanup.exists()
            and json.loads(cleanup.read_text())["status"] == "delete_completed"
        )
        write_new(
            root / "verdict.json",
            dict(
                passed=passed and cleaned,
                setup_and_command_passed=passed,
                cleanup_passed=cleaned,
                elapsed=time.monotonic() - started,
                calls=counts,
                model_loaded=False,
            ),
        )
        if env._client_manager is not None:
            await asyncio.wait_for(env._client_manager._cleanup(), 5)
    return passed and cleaned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "manifest", "harbor-root", "credential-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    return 0 if asyncio.run(diagnose(args)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
