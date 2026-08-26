import subprocess
from types import SimpleNamespace

from hpc.vllm import start_vllm_ray_controller as controller


def _args(**overrides):
    values = {
        "model": "/models/checkpoint",
        "host": "127.0.0.1",
        "port": 8000,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 2,
        "ray_address": "127.0.0.1:6379",
        "served_model_name": "example/model",
        "verbose": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cli_discovery_failure_assumes_required_cuda_flags_are_supported(monkeypatch):
    monkeypatch.delenv("VLLM_SKIP_FLAG_DISCOVERY", raising=False)
    monkeypatch.setattr(controller, "_supported_flags", lambda: None)

    assert controller._flag_supported("--data-parallel-size")


def test_cli_discovery_timeout_returns_unknown(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(controller.subprocess, "run", timeout)

    assert controller._discover_cli_flags() is None


def test_cli_discovery_failure_does_not_forward_ray_address(monkeypatch):
    monkeypatch.delenv("VLLM_SKIP_FLAG_DISCOVERY", raising=False)
    monkeypatch.setattr(controller, "_supported_flags", lambda: None)

    cmd = controller.build_vllm_command(_args(), [])

    assert "--data-parallel-size" in cmd
    assert "--ray-address" not in cmd


def test_ray_address_is_forwarded_when_cli_explicitly_supports_it(monkeypatch):
    monkeypatch.delenv("VLLM_SKIP_FLAG_DISCOVERY", raising=False)
    monkeypatch.setattr(
        controller,
        "_supported_flags",
        lambda: {"--data-parallel-size", "--ray-address"},
    )

    cmd = controller.build_vllm_command(_args(), [])

    index = cmd.index("--ray-address")
    assert cmd[index + 1] == "127.0.0.1:6379"


def test_ray_address_is_not_forwarded_when_cli_explicitly_omits_it(monkeypatch):
    monkeypatch.delenv("VLLM_SKIP_FLAG_DISCOVERY", raising=False)
    monkeypatch.setattr(controller, "_supported_flags", lambda: {"--data-parallel-size"})

    cmd = controller.build_vllm_command(_args(), [])

    assert "--ray-address" not in cmd


def test_skip_discovery_keeps_required_dp_and_drops_ray_address(monkeypatch):
    monkeypatch.setenv("VLLM_SKIP_FLAG_DISCOVERY", "1")

    cmd = controller.build_vllm_command(_args(), [])

    assert "--data-parallel-size" in cmd
    assert "--ray-address" not in cmd


def test_tpu_guard_drops_ray_address_even_if_discovery_mentions_it(monkeypatch):
    monkeypatch.delenv("VLLM_SKIP_FLAG_DISCOVERY", raising=False)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-8")
    monkeypatch.setattr(
        controller,
        "_supported_flags",
        lambda: {"--data-parallel-size", "--ray-address"},
    )

    cmd = controller.build_vllm_command(_args(), [])

    assert "--data-parallel-size" in cmd
    assert "--ray-address" not in cmd


def test_cli_discovery_failure_keeps_required_pipeline_parallel_size(monkeypatch):
    monkeypatch.delenv("VLLM_SKIP_FLAG_DISCOVERY", raising=False)
    monkeypatch.setattr(controller, "_supported_flags", lambda: None)

    cmd = controller.build_vllm_command(_args(pipeline_parallel_size=2), [])

    assert "--pipeline-parallel-size" in cmd


def test_subprocess_environment_always_contains_ray_address(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "stale.example:1234")

    env = controller.build_subprocess_env(_args())

    assert env["RAY_ADDRESS"] == "127.0.0.1:6379"
