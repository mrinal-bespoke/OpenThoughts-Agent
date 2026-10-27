"""One synthetic 32K Harbor/Daytona smoke. No benchmark, snapshot build or retry."""

import asyncio
import faulthandler
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from lifecycle import validate_lifetime
import urllib.request
from pathlib import Path

BASE = Path("/e/scratch/reformo/kumar37/jupiter-vista-parity-20261018")
ROOT = Path(__file__).resolve().parent
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ALIAS = "qwen8-instruct-b968826d9c46"
SNAPSHOT = "daytona-small"


def record(name, value):
    with (ROOT / name).open("x") as f:
        json.dump(value, f, indent=2)


def request(path, body=None):
    req = urllib.request.Request(
        "http://127.0.0.1:18291" + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90 if body else 5) as response:
        assert response.status == 200
        return json.load(response)


def model_check():
    started = time.monotonic()
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)

    def stage(name):
        print(
            json.dumps(
                {"preflight_stage": name, "elapsed_s": time.monotonic() - started}
            ),
            flush=True,
        )

    stage("python_ready")
    import torch

    stage("torch_imported")
    from transformers import AutoTokenizer

    stage("transformers_imported")
    config = json.loads((BASE / "qwen8-instruct.model.json").read_text())
    assert config["revision"] == REVISION
    assert Path(config["snapshot"]).name == REVISION
    tokenizer = AutoTokenizer.from_pretrained(config["snapshot"], local_files_only=True)
    stage("tokenizer_loaded")
    assert (
        hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
        == config["template_sha256"]
    )
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    stage("cuda_available_single_gpu")
    a = torch.ones((256, 256), device="cuda", dtype=torch.bfloat16)
    assert bool(((a @ a) == 256).all())
    stage("bf16_matmul_passed")
    faulthandler.cancel_dump_traceback_later()
    record(
        "gpu.json",
        {
            "device": torch.cuda.get_device_name(),
            "revision": REVISION,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "bf16_matmul": True,
            "elapsed_s": time.monotonic() - started,
        },
    )


async def agent_smoke():
    from daytona import CreateSandboxFromSnapshotParams
    from harbor.environments.daytona.environment import DaytonaEnvironment
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths
    from harbor.models.agent.context import AgentContext
    from harbor.agents.terminus_2.terminus_2 import Terminus2

    class ExistingSnapshotOnly(DaytonaEnvironment):
        async def _resolve_start_sandbox_params(
            self, daytona, resources, *, force_build
        ):
            if force_build:
                raise RuntimeError("Build forbidden")
            snap = await daytona.snapshot.get(SNAPSHOT)
            assert snap.id == "dad882b3-a047-4715-8713-bddec50bb7ca", (
                "Snapshot identity changed"
            )
            record(
                "sdk-snapshot.json",
                {"name": snap.name, "id": snap.id, "via_proxy": True},
            )
            return CreateSandboxFromSnapshotParams(
                snapshot=SNAPSHOT,
                name="jupiter-smoke-" + os.environ["SLURM_JOB_ID"],
                labels={
                    "purpose": "jupiter-smoke-v3a-20261027",
                    "slurm_job": os.environ["SLURM_JOB_ID"],
                },
                ephemeral=True,
                auto_stop_interval=10,
                auto_delete_interval=0,
                ttl_minutes=15,
            )

        async def _create_sandbox(self, params, daytona=None):
            assert (
                isinstance(params, CreateSandboxFromSnapshotParams)
                and params.snapshot == SNAPSHOT
            )
            record(
                "sandbox-create-intent.json",
                {"job": os.environ["SLURM_JOB_ID"], "snapshot": SNAPSHOT},
            )
            self._sandbox = await daytona.create(params, timeout=120)
            record("sandbox.json", {"id": self._sandbox.id, "snapshot": SNAPSHOT})

    trial = ROOT / "synthetic"
    trial.mkdir()
    envdir = ROOT / "synthetic-env"
    envdir.mkdir()
    # Definition is required by Harbor but can never enter a build path above.
    (envdir / "Dockerfile").write_text("FROM scratch\n")
    env = ExistingSnapshotOnly(
        environment_dir=envdir,
        environment_name="jupiter-infra-smoke",
        session_id="jupiter-smoke-" + os.environ["SLURM_JOB_ID"],
        trial_paths=TrialPaths(trial),
        task_env_config=EnvironmentConfig(build_timeout_sec=120),
        snapshot_template_name=SNAPSHOT,
        auto_snapshot=False,
        auto_stop_interval_mins=10,
        ttl_minutes=15,
    )
    complete = False
    try:
        await asyncio.wait_for(env.start(force_build=False), 180)
        reply = await asyncio.wait_for(
            env.exec("printf JUPITER_NETWORK_OK", timeout_sec=20), 40
        )
        assert reply.return_code == 0 and "JUPITER_NETWORK_OK" in reply.stdout
        record(
            "sdk-command.json",
            {"return_code": reply.return_code, "stdout": reply.stdout},
        )
        agent = Terminus2(
            logs_dir=trial / "agent",
            model_name="hosted_vllm/" + ALIAS,
            api_base="http://127.0.0.1:18291/v1",
            max_turns=4,
            temperature=0.6,
            enable_summarize=False,
            record_terminal_session=False,
            model_info={
                "max_input_tokens": 24576,
                "max_output_tokens": 8192,
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        await asyncio.wait_for(agent.setup(env), 90)
        context = AgentContext()
        instruction = (
            "Infrastructure smoke test only. Use the terminal to write exactly the text "
            "JUPITER_AGENT_OK into /tmp/jupiter_agent_smoke.txt. Read the file back to verify it. "
            "Then mark the task complete. Do not install packages or access any other files or networks."
        )
        await asyncio.wait_for(agent.run(instruction, env, context), 180)
        reply = await asyncio.wait_for(
            env.exec("cat /tmp/jupiter_agent_smoke.txt", timeout_sec=20), 40
        )
        assert reply.return_code == 0 and reply.stdout.strip() == "JUPITER_AGENT_OK"
        record(
            "agent-result.json",
            {
                "passed": True,
                "artifact": reply.stdout.strip(),
                "context": context.model_dump(mode="json"),
            },
        )
        complete = True
    finally:
        try:
            await asyncio.wait_for(env.stop(delete=True), 90)
            record("sandbox-cleanup.json", {"delete_completed": True})
        except BaseException as exc:
            record(
                "sandbox-cleanup.json",
                {"delete_completed": False, "error_type": type(exc).__name__},
            )
            raise
    assert complete


def controller():
    private = json.loads((ROOT / "private.json").read_text())
    ready = json.loads((ROOT / "proxy-ready.json").read_text())
    assert ready["expires_epoch"] - time.time() > 780, "Proxy lifetime insufficient"
    proxy = f"http://smoke:{private['password']}@{ready['host']}:{ready['port']}"
    os.environ.update(
        HTTPS_PROXY=proxy,
        HTTP_PROXY=proxy,
        https_proxy=proxy,
        http_proxy=proxy,
        NO_PROXY="localhost,127.0.0.1",
        no_proxy="localhost,127.0.0.1",
        DAYTONA_API_KEY=private["daytona_key"],
        DAYTONA_API_URL="https://app.daytona.io/api",
        DAYTONA_USE_DEPRECATED_POLLING="true",
        DAYTONA_OTEL_ENABLED="false",
    )
    del private
    with urllib.request.urlopen("https://app.daytona.io/", timeout=20) as response:
        record("network.json", {"status": response.status, "proxy_host": ready["host"]})
    asyncio.run(agent_smoke())


def run():
    if (ROOT / "run-started.json").exists():
        raise RuntimeError("Never replay this smoke directory")
    record(
        "run-started.json",
        {
            "job": os.environ["SLURM_JOB_ID"],
            "context": 32768,
            "benchmark": False,
            "snapshot_creation_allowed": False,
            "time_limit_minutes": 30,
        },
    )
    ready = json.loads((ROOT / "proxy-ready.json").read_text())
    validate_lifetime(ready, time.time())
    config = json.loads((BASE / "qwen8-instruct.model.json").read_text())
    assert config["revision"] == REVISION and config["alias"] == ALIAS
    ids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        timeout=20,
    ).splitlines()
    assert ids
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=ids[0].strip(),
        HF_HOME=str(BASE / "hf-cache"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        XDG_CACHE_HOME=str(ROOT / "cache"),
        TRITON_CACHE_DIR=str(ROOT / "cache/triton"),
        FLASHINFER_WORKSPACE_BASE=str(ROOT / "cache/flashinfer"),
        FLASHINFER_EXTRA_CUDAFLAGS="-I"
        + str(BASE / "serving/lib/python3.12/site-packages/nvidia/curand/include"),
        FLASHINFER_EXTRA_LDFLAGS="-L"
        + str(BASE / "toolchain/targets/sbsa-linux/lib/stubs"),
        LD_PRELOAD=":".join(
            str(BASE / "toolchain/lib" / n) for n in ("libstdc++.so.6", "libgcc_s.so.1")
        ),
    )
    preflight_started = time.monotonic()
    try:
        with (ROOT / "preflight.log").open("x") as preflight_log:
            subprocess.run(
                [str(BASE / "serving/bin/python"), "-u", __file__, "model-check"],
                env=env,
                check=True,
                timeout=300,
                stdout=preflight_log,
                stderr=subprocess.STDOUT,
            )
    finally:
        record(
            "preflight-timing.json",
            {"elapsed_s": time.monotonic() - preflight_started, "limit_s": 300},
        )
    command = [
        str(BASE / "serving/bin/vllm"),
        "serve",
        config["snapshot"],
        "--tokenizer",
        config["snapshot"],
        "--served-model-name",
        ALIAS,
        "--host",
        "127.0.0.1",
        "--port",
        "18291",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.85",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--enable-prefix-caching",
    ]
    server = None

    def stop_signal(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        with (ROOT / "vllm.log").open("x") as logfile:
            server = subprocess.Popen(
                command,
                env=env,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        serving_started = time.monotonic()
        deadline = serving_started + 600
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("Model server stopped")
            try:
                if [m["id"] for m in request("/v1/models")["data"]] == [ALIAS]:
                    break
            except OSError:
                pass
            time.sleep(5)
        else:
            raise TimeoutError("Serving readiness exceeded 600 seconds")
        record(
            "serving-startup-timing.json",
            {"elapsed_s": time.monotonic() - serving_started, "limit_s": 600},
        )
        body = {
            "model": ALIAS,
            "messages": [{"role": "user", "content": "Reply with the number 5."}],
            "temperature": 0.6,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = request("/v1/chat/completions", body)
        assert response["choices"][0]["message"]["content"].strip()
        record("serving.json", {"request": body, "response": response})
        harness_env = dict(os.environ, PYTHONPATH=str(BASE / "harbor/src"))
        subprocess.run(
            [
                str(BASE / "proxychains/bin/proxychains4"),
                "-f",
                str(ROOT / "proxychains.conf"),
                str(BASE / "harness/bin/python"),
                __file__,
                "controller",
            ],
            env=harness_env,
            check=True,
            timeout=660,
        )
        record(
            "verdict.json",
            {"passed": True, "context": 32768, "benchmark_runs": 0, "new_snapshots": 0},
        )
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


if __name__ == "__main__":
    try:
        {"model-check": model_check, "controller": controller, "run": run}[
            sys.argv[1]
        ]()
    except BaseException as exc:
        # Preserve diagnosis without leaking proxy/API credentials in exceptions.
        detail = traceback.format_exc()
        if (ROOT / "private.json").exists():
            private = json.loads((ROOT / "private.json").read_text())
            for key in ("password", "daytona_key"):
                detail = detail.replace(private[key], "[REDACTED]")
        name = "failure-" + sys.argv[1] + ".json"
        if not (ROOT / name).exists():
            record(name, {"error_type": type(exc).__name__, "traceback": detail})
        print("SMOKE_FAILED", type(exc).__name__, flush=True)
        raise SystemExit(1)
