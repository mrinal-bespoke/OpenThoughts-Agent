"""One bounded, credential-safe transport diagnostic; no model or benchmark."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path


async def probe(root, credential_file):
    from daytona import (
        AsyncDaytona,
        CreateSandboxFromSnapshotParams,
        DaytonaConfig,
        DaytonaNotFoundError,
        SessionExecuteRequest,
    )
    from eval.autoeval32k.runtime import read_daytona_key

    purpose = root.name
    snapshot_id = "dad882b3-a047-4715-8713-bddec50bb7ca"
    started = time.monotonic()
    sandbox = None
    attempted = False
    passed = False
    cleaned = False

    def record(name, **data):
        with (root / name).open("x") as stream:
            json.dump(data, stream)

    async def step(name, awaitable, seconds=4):
        with (root / "stages.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "stage": name,
                        "event": "enter",
                        "elapsed": time.monotonic() - started,
                    }
                )
                + "\n"
            )
        result = await asyncio.wait_for(awaitable, seconds)
        with (root / "stages.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "stage": name,
                        "event": "pass",
                        "elapsed": time.monotonic() - started,
                    }
                )
                + "\n"
            )
        return result

    async with AsyncDaytona(
        DaytonaConfig(
            api_key=read_daytona_key(credential_file),
            api_url="https://app.daytona.io/api",
        )
    ) as client:
        try:
            async with asyncio.timeout(24):
                snapshot = await step("snapshot", client.snapshot.get("daytona-small"))
                assert snapshot.id == snapshot_id
                record("snapshot.json", id=snapshot.id, name=snapshot.name)
                record("create-intent.json", purpose=purpose, snapshot_id=snapshot_id)
                attempted = True
                sandbox = await step(
                    "create",
                    client.create(
                        CreateSandboxFromSnapshotParams(
                            snapshot="daytona-small",
                            name=purpose,
                            labels={"purpose": purpose},
                            ephemeral=True,
                            public=False,
                            auto_stop_interval=5,
                            auto_delete_interval=0,
                            ttl_minutes=10,
                        ),
                        timeout=8,
                    ),
                    seconds=10,
                )
                assert sandbox.labels.get("purpose") == purpose
                record("sandbox.json", id=sandbox.id, purpose=purpose)
                direct = await step(
                    "direct_exec",
                    sandbox.process.exec("printf AUTOEVAL_DAYTONA_OK", timeout=3),
                )
                assert (
                    direct.exit_code == 0
                    and direct.result.strip() == "AUTOEVAL_DAYTONA_OK"
                )
                record("direct.json", exit_code=direct.exit_code, output_matches=True)
                session_id = "autoeval-transport"
                await step(
                    "session_create",
                    sandbox.process.create_session(session_id, request_timeout=3),
                )
                command = await step(
                    "session_execute",
                    sandbox.process.execute_session_command(
                        session_id,
                        SessionExecuteRequest(
                            command="mkdir -p /tmp/autoeval-session-probe && printf AUTOEVAL_SESSION_OK",
                            run_async=True,
                        ),
                        timeout=3,
                    ),
                )
                assert command.cmd_id
                record("session.json", session_id=session_id, command_id=command.cmd_id)
                details = await step(
                    "session_get",
                    sandbox.process.get_session_command(
                        session_id, command.cmd_id, request_timeout=3
                    ),
                )
                # No command replay. Only query the original command until terminal.
                while details.exit_code is None:
                    await asyncio.sleep(0.1)
                    details = await step(
                        "session_poll",
                        sandbox.process.get_session_command(
                            session_id, command.cmd_id, request_timeout=3
                        ),
                    )
                logs = await step(
                    "session_logs",
                    sandbox.process.get_session_command_logs(
                        session_id, command.cmd_id, request_timeout=3
                    ),
                )
                assert (
                    details.exit_code == 0
                    and logs.stdout.strip() == "AUTOEVAL_SESSION_OK"
                )
                record(
                    "session-result.json",
                    exit_code=details.exit_code,
                    output_matches=True,
                )
                passed = True
        except Exception as exc:
            record("failure.json", error_type=type(exc).__name__)
        finally:
            cleanup = {"attempted_create": attempted, "confirmed_absent": False}
            try:
                async with asyncio.timeout(10):
                    if attempted:
                        if sandbox is None:
                            sandbox = await client.get(purpose, request_timeout=3)
                        assert sandbox.labels.get("purpose") == purpose
                        cleanup["sandbox_id"] = sandbox.id
                        await client.delete(sandbox, timeout=3)
                        while True:
                            try:
                                remaining = await client.get(
                                    sandbox.id, request_timeout=3
                                )
                                if (
                                    str(
                                        getattr(
                                            remaining.state, "value", remaining.state
                                        )
                                    ).lower()
                                    == "destroyed"
                                ):
                                    cleaned = True
                                    break
                            except DaytonaNotFoundError:
                                cleaned = True
                                break
                            await asyncio.sleep(0.2)
                    else:
                        cleaned = True
            except Exception as exc:
                cleanup["error_type"] = type(exc).__name__
            cleanup["confirmed_absent"] = cleaned
            record("cleanup.json", **cleanup)
            record(
                "verdict.json",
                passed=passed and cleaned,
                transport_passed=passed,
                elapsed_seconds=time.monotonic() - started,
                location="TACC login node",
                compute_validated=False,
                model_loaded=False,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.root.mkdir(exist_ok=False)  # Never replay an uncertain diagnostic.
    asyncio.run(probe(args.root, args.credential_file))


if __name__ == "__main__":
    main()
